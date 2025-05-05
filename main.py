import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
import torch.multiprocessing as mp

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
from mast3r_slam.evaluate import prepare_savedir, save_traj, save_reconstruction, save_keyframes
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization

# Move this function definition to module level (outside of any other function)
def relocalize(frame, keyframes, factor_graph, retriever):
    with keyframes.lock:
        candidates = retriever.update(frame, add_after_query=False,
                                      k=config["retrieval"]["k"],
                                      min_thresh=config["retrieval"]["min_thresh"])
        if not candidates:
            return False

        keyframes.append(frame)
        idx = len(keyframes) - 1
        if factor_graph.add_factors([idx] * len(candidates), list(candidates),
                                     config["reloc"]["min_match_frac"],
                                     is_reloc=config["reloc"]["strict"]):
            retriever.update(frame, add_after_query=True,
                             k=config["retrieval"]["k"],
                             min_thresh=config["retrieval"]["min_thresh"])
            keyframes.T_WC[idx] = keyframes.T_WC[candidates[0]].clone()
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
            return True

        keyframes.pop_last()
        return False

# Define backend_loop at module level
def backend_loop(cfg, model, states, keyframes, K):
    set_global_config(cfg)
    factor_graph = FactorGraph(model, keyframes, K, keyframes.device)
    retriever = load_retriever(model)

    while (mode := states.get_mode()) != Mode.TERMINATED:
        if mode in (Mode.INIT, ) or states.is_paused():
            time.sleep(0.01)
            continue

        if mode == Mode.RELOC:
            frame = states.get_frame()
            if relocalize(frame, keyframes, factor_graph, retriever):
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue

        with states.lock:
            if not states.global_optimizer_tasks:
                time.sleep(0.01)
                continue
            idx = states.global_optimizer_tasks[0]

        neighbors = [idx - 1 - i for i in range(min(1, idx))]
        frame = keyframes[idx]
        retrieved = retriever.update(frame, add_after_query=True,
                                     k=config["retrieval"]["k"],
                                     min_thresh=config["retrieval"]["min_thresh"])
        neighbors += retrieved

        neighbors = list(set(neighbors) - {idx})
        if neighbors:
            factor_graph.add_factors(neighbors, [idx] * len(neighbors),
                                     config["local_opt"]["min_match_frac"])

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()
            states.global_optimizer_tasks.pop(0)

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

# Define main function at module level
def main(dataset_path, config_path, save_name, no_viz=False, calib_path=None ):
    # Make sure to set start method if not already set
    
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"

    load_config(config_path)
    dataset = load_dataset(dataset_path)
    dataset.subsample(config["dataset"]["subsample"])

    if calib_path:
        with open(calib_path) as f:
            intrinsics = yaml.safe_load(f)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"])
    
    height, width = dataset.get_img_shape()[0]
    manager = mp.Manager()
    to_viz = new_queue(manager, no_viz)
    from_viz = new_queue(manager, no_viz)
    keyframes = SharedKeyframes(manager, height, width, buffer=config.get("buffer_size",256))
    print(f"Dataset length: {len(dataset)}")
    print(f"Buffer length: {keyframes.buffer}")
   
    states = SharedStates(manager, height, width)
    
    if not no_viz:
        viz_process = mp.Process(target=run_visualization,
                   args=(config, states, keyframes, to_viz, from_viz))
        viz_process.daemon = True  # This ensures the process is terminated when the main program exits
        viz_process.start()
  
    model = load_mast3r(device=device)
    model.share_memory()
   
    if config["use_calib"] and not dataset.has_calib():
        print("[Warning] Missing calibration!")
        sys.exit(1)

    K = torch.tensor(dataset.camera_intrinsics.K_frame,
                     dtype=torch.float32, device=device) if config["use_calib"] else None
    if K is not None:
        keyframes.set_intrinsics(K)

    if dataset.save_results:
        save_dir, seq_name = prepare_savedir(save_name, dataset)
        for ext in (".txt", ".ply"):
            path = save_dir / f"{seq_name}{ext}"
            if path.exists():
                path.unlink()

    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()
    
    # Create backend process
    backend_process = mp.Process(target=backend_loop,
               args=(config, model, states, keyframes, K))
    backend_process.daemon = True  # This ensures the process is terminated when the main program exits
    backend_process.start()

    frame_idx = 0
    fps_start = time.time()
    frames = []
    save_frames = False
    
    try:
        while frame_idx < len(dataset):
            
            
            if len(keyframes) >= config["buffer_size"]-1:
                print("Buffer full, exiting...")
                break
            msg = try_get_msg(from_viz)
            last_msg = msg or last_msg

            if last_msg.is_terminated:
                states.set_mode(Mode.TERMINATED)
                break
            if last_msg.is_paused and not last_msg.next:
                states.pause()
                time.sleep(0.01)
                continue
            if not last_msg.is_paused:
                states.unpause()

            timestamp, image = dataset[frame_idx]
            if save_frames:
                frames.append(image)

            T_WC = lietorch.Sim3.Identity(1, device=device) if frame_idx == 0 else states.get_frame().T_WC
            frame = create_frame(frame_idx, image, T_WC,
                                img_size=dataset.img_size, device=device)

            match states.get_mode():
                case Mode.INIT:
                
                    X, C = mast3r_inference_mono(model, frame)
                    frame.update_pointmap(X, C)
                    keyframes.append(frame)
                    states.queue_global_optimization(len(keyframes) - 1)
                    states.set_mode(Mode.TRACKING)
                    states.set_frame(frame)
                    frame_idx += 1
                
                case Mode.TRACKING:
                
                    add_kf, _, needs_reloc = tracker.track(frame)
                    if needs_reloc:
                        states.set_mode(Mode.RELOC)
                    states.set_frame(frame)

                    if add_kf:
                        keyframes.append(frame)
                        states.queue_global_optimization(len(keyframes) - 1)
                        if config["single_thread"]:
                            while states.global_optimizer_tasks:
                                time.sleep(0.01)

                case Mode.RELOC:
                    
                    X, C = mast3r_inference_mono(model, frame)
                    frame.update_pointmap(X, C)
                    states.set_frame(frame)
                    states.queue_reloc()
                    if config["single_thread"]:
                        while states.reloc_sem.value:
                            time.sleep(0.01)

            if frame_idx % 30 == 0:
                fps = frame_idx / (time.time() - fps_start)
                print(f"FPS: {fps:.2f}")
                progress = (frame_idx / len(dataset)) * 100
                print(f"Progress: {progress:.2f}%")
                print("current number of keyframes: ", len(keyframes), "max number of keyframes: ", config["buffer_size"])  

            frame_idx += 1
            
                
            

        if dataset.save_results:
            save_dir, seq_name = prepare_savedir(save_name, dataset)
            save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
            save_reconstruction(save_dir, f"{seq_name}.ply", keyframes, last_msg.C_conf_threshold)
            save_keyframes(save_dir / "keyframes", dataset.timestamps, keyframes)

        if save_frames:
            path = pathlib.Path(f"logs/frames/{datetime.datetime.now().isoformat()}")
            path.mkdir(parents=True, exist_ok=True)
            for i, img in tqdm.tqdm(enumerate(frames), total=len(frames)):
                img = (img * 255).clip(0, 255).astype("uint8")
                cv2.imwrite(str(path / f"{i:05}.png"), img)
        
    
    finally:
        # Ensure we clean up processes
        states.set_mode(Mode.TERMINATED)
        # Give processes time to terminate gracefully
        time.sleep(0.5)
        
        # Force terminate if still running
        if backend_process.is_alive():
            backend_process.terminate()
        if not no_viz and viz_process.is_alive():
            viz_process.terminate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("config")
    parser.add_argument("--save_as", default="output")
    # parser.add_argument("--no_viz", action="store_true", default = False)
    parser.add_argument("--no_viz",  default = False)
    parser.add_argument("--calib")
    args = parser.parse_args()

    # call main() with parsed args
    main(dataset_path=args.dataset, 
         config_path=args.config, 
         save_name=args.save_as, 
         no_viz=args.no_viz, 
         calib_path=args.calib, 
    )