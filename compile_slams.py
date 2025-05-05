import os 
import re
import argparse
import shutil
import numpy as np
import re



def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions
    q1, q2: quaternions in format [x, y, z, w]
    Returns quaternion product q1*q2
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([x, y, z, w])

def quaternion_inverse(q):
    """
    Calculate the inverse of a quaternion
    q: quaternion in format [x, y, z, w]
    Returns inverse quaternion
    """
    x, y, z, w = q
    norm_sq = x*x + y*y + z*z + w*w
    
    if norm_sq < 1e-10:
        return np.array([0, 0, 0, 1])  # Default to identity if norm is too small
    
    # Conjugate divided by norm squared
    return np.array([-x/norm_sq, -y/norm_sq, -z/norm_sq, w/norm_sq])

def normalize_quaternion(q):
    """
    Normalize a quaternion to unit length
    q: quaternion in format [x, y, z, w]
    Returns normalized quaternion
    """
    norm = np.sqrt(np.sum(q * q))
    
    if norm < 1e-10:
        return np.array([0, 0, 0, 1])  # Default to identity if norm is too small
    
    return q / norm

def rotate_vector_by_quaternion(v, q):
    """
    Rotate a 3D vector v by quaternion q
    v: 3D vector [x, y, z]
    q: quaternion [x, y, z, w]
    Returns rotated vector
    """
    # Convert vector to pure quaternion form (0 as real part)
    v_quat = np.array([v[0], v[1], v[2], 0])
    
    # q * v_quat * q^(-1)
    q_inv = quaternion_inverse(q)
    
    # First multiply q * v_quat
    temp = quaternion_multiply(q, v_quat)
    
    # Then multiply by q_inv
    result_quat = quaternion_multiply(temp, q_inv)
    
    # Extract the vector part
    result_vector = result_quat[:3]
    
    return result_vector

def process_slams(folder_slams, chunk_folders, overlap, max_frames, max_fps):
    """
    Process SLAM data from multiple chunk folders and merge them into a consolidated trajectory.
    
    Args:
        folder_slams (str): Path to the folder containing chunk-specific SLAM data folders
        chunk_folders (list): List of chunk folder names to process
        overlap (float): Proportion of overlapping frames between consecutive chunks (0.0 to 1.0)
        max_frames (int): Maximum number of frames per chunk
        max_fps (float): Maximum frames per second
        
    Output:
        Creates a consolidated folder with merged trajectory data and keyframes
    """
    if not chunk_folders:
        print("No chunk folders provided.")
        return
    
    # Extract video name from the first chunk folder
    match = re.match(r'(.+)_chunk_\d+', chunk_folders[0])
    if not match:
        print(f"Could not extract video name from folder: {chunk_folders[0]}")
        return
    
    video_name = match.group(1)
    output_folder = os.path.join(folder_slams, video_name)
    output_keyframes_folder = os.path.join(output_folder, "keyframes")
    
    # Create output directories
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_keyframes_folder, exist_ok=True)
    
    # Calculate chunk duration in seconds (except potentially the last chunk)
    chunk_duration = max_frames / max_fps
    overlap_duration = chunk_duration * overlap
    
    # Process each chunk and reconcile consecutive ones
    all_poses = []  # Will store (time, x, y, z, qx, qy, qz, qw) for all chunks
    keyframe_mappings = {}  # Will store {original_time: adjusted_time} for keyframes
    
    # For transformation between coordinate systems
    prev_transform = None
    
    for i, chunk_name in enumerate(chunk_folders):
        chunk_folder = os.path.join(folder_slams, chunk_name)
        slam_file = os.path.join(chunk_folder, f"{chunk_name}.txt")
        keyframes_folder = os.path.join(chunk_folder, "keyframes")
        
        
        if not os.path.exists(slam_file):
            print(f"SLAM file not found: {slam_file}")
            continue
        
        # Calculate time offset for this chunk
        # First chunk starts at t=0, subsequent chunks need time adjustment
        time_offset = 0 if i == 0 else (
            i * chunk_duration - i * overlap_duration
        )
        
        # Read SLAM data from file
        chunk_poses = []
        with open(slam_file, 'r') as f:
            for line in f:
               
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) == 8:  # t,x,y,z,qx,qy,qz,qw
                    t, x, y, z, qx, qy, qz, qw = map(float, parts)
                    chunk_poses.append((t, x, y, z, qx, qy, qz, qw))
        
        # If this is not the first chunk, we need to align it with the previous chunk
        if i > 0 and chunk_poses and prev_chunk_poses:
            # Find overlapping portion in time
            overlap_start_prev = chunk_duration - overlap_duration
            overlap_end_prev = chunk_duration
            
            # Get poses from previous chunk in the overlap region
            prev_overlap_poses = [pose for pose in prev_chunk_poses 
                                if overlap_start_prev <= pose[0] <= overlap_end_prev]
            
            # Get poses from current chunk in the overlap region
            current_overlap_poses = [pose for pose in chunk_poses 
                                   if 0 <= pose[0] <= overlap_duration]
            
            # Check if we have enough data for alignment
            if prev_overlap_poses and current_overlap_poses:
                # Simple alignment - find transformation between middle points in overlap
                index_prev , index_curr = 0,0
                min_dist_time = float('inf')
                for i1 in range(len(prev_overlap_poses)):
                    for i2 in range(len(current_overlap_poses)):
                        if abs(prev_overlap_poses[i1][0] -current_overlap_poses[i2][0])< min_dist_time:
                            min_dist_time = abs(prev_overlap_poses[i1][0] -current_overlap_poses[i2][0])
                            index_prev = i1
                            index_curr = i2
                
                mid_idx_prev = index_prev #len(prev_overlap_poses) // 2
                mid_idx_curr = index_curr #len(current_overlap_poses) // 2
                
                # Get representative poses from each chunk
                _, px, py, pz, pqx, pqy, pqz, pqw = prev_overlap_poses[mid_idx_prev]
                _, cx, cy, cz, cqx, cqy, cqz, cqw = current_overlap_poses[mid_idx_curr]
                
                # Calculate proper transformation using quaternions and positions
                # First, compute relative rotation and translation between coordinate systems
                
                # Quaternion representation of previous and current orientations
                prev_quat = np.array([pqx, pqy, pqz, pqw])
                curr_quat = np.array([cqx, cqy, cqz, cqw])
                
                # Compute rotation that takes current to previous
                rot_quat = quaternion_multiply(prev_quat, quaternion_inverse(curr_quat))
                
                # Previous and current positions
                prev_pos = np.array([px, py, pz])
                curr_pos = np.array([cx, cy, cz])
                
                # Rotate current position using the rotation quaternion
                rotated_curr_pos = rotate_vector_by_quaternion(curr_pos, rot_quat)
                
                # Calculate translation after rotation
                translation = prev_pos - rotated_curr_pos
                
                # Transform all poses in current chunk
                transformed_poses = []
                for t, x, y, z, qx, qy, qz, qw in chunk_poses:
                    # Apply rotation to position
                    pos = np.array([x, y, z])
                    quat = np.array([qx, qy, qz, qw])
                    
                    # Rotate the position vector
                    rotated_pos = rotate_vector_by_quaternion(pos, rot_quat)
                    
                    # Apply translation
                    new_pos = rotated_pos + translation
                    
                    # Apply rotation to orientation quaternion
                    new_quat = quaternion_multiply(rot_quat, quat)
                    new_quat = normalize_quaternion(new_quat)
                    
                    transformed_poses.append((
                        t, 
                        new_pos[0], new_pos[1], new_pos[2], 
                        new_quat[0], new_quat[1], new_quat[2], new_quat[3]
                    ))
                
                chunk_poses = transformed_poses
        
        # Store current chunk poses for alignment with next chunk
        prev_chunk_poses = chunk_poses
        
        # Adjust timestamps and add to consolidated trajectory
        for t, x, y, z, qx, qy, qz, qw in chunk_poses:
            adjusted_time = time_offset + t
            
            # For the overlap regions, we want to include:
            # - First half of overlap from the first chunk
            # - Second half of overlap from the second chunk
            if i > 0 and t < overlap_duration:
                # This is in the overlap region of the current chunk
                normalized_pos_in_overlap = t / overlap_duration
                
                # Only include if in the second half of the overlap
                if normalized_pos_in_overlap >= 0.5:
                    all_poses.append((adjusted_time, x, y, z, qx, qy, qz, qw))
                    
                    # Check and record if this is a keyframe
                    keyframe_path = os.path.join(keyframes_folder, f"{t}.png")
                    if os.path.exists(keyframe_path):
                        keyframe_mappings[keyframe_path] = adjusted_time
            elif i < len(chunk_folders) - 1 and t > (chunk_duration - overlap_duration):
                # This is in the overlap region with the next chunk
                normalized_pos_in_overlap = (t - (chunk_duration - overlap_duration)) / overlap_duration
                
                # Only include if in the first half of the overlap
                if normalized_pos_in_overlap < 0.5:
                    all_poses.append((adjusted_time, x, y, z, qx, qy, qz, qw))
                    
                    # Check and record if this is a keyframe
                    keyframe_path = os.path.join(keyframes_folder, f"{t}.png")
                  
                    if os.path.exists(keyframe_path):
                        keyframe_mappings[keyframe_path] = adjusted_time
            else:
                # This is not in any overlap region, include it
                all_poses.append((adjusted_time, x, y, z, qx, qy, qz, qw))
                
                # Check and record if this is a keyframe
                keyframe_path = os.path.join(keyframes_folder, f"{t}.png")
                if os.path.exists(keyframe_path):
                    keyframe_mappings[keyframe_path] = adjusted_time
    
    # Sort poses by adjusted time
    all_poses.sort(key=lambda x: x[0])
    
    # Write consolidated trajectory to file
    output_slam_file = os.path.join(output_folder, f"{video_name}.txt")
    with open(output_slam_file, 'w') as f:
        for t, x, y, z, qx, qy, qz, qw in all_poses:
            f.write(f"{t},{x},{y},{z},{qx},{qy},{qz},{qw}\n")
    
    # Copy and rename keyframes
    for src_keyframe_path, adjusted_time in keyframe_mappings.items():
        
        if os.path.exists(src_keyframe_path):
            dst_keyframe_path = os.path.join(output_keyframes_folder, f"{adjusted_time}.png")
            shutil.copy2(src_keyframe_path, dst_keyframe_path)
    
    print(f"Processed {len(chunk_folders)} chunks into {video_name}")
    print(f"Generated trajectory with {len(all_poses)} poses and {len(keyframe_mappings)} keyframes")





def main(config):
    root = config["general"]["root"]
    folder_slams=str(root)+ config["slam"]["log_path"]
    overlap= config["preprocess"]["overlap"]
    max_frames=config["preprocess"]["max_frames_per_video"]
    max_fps= config["preprocess"]["max_fps"]
    list_folders = os.listdir(folder_slams)
    for folder in list_folders:
        match = re.match(r"(.*)_chunk_\d+$", folder)
        if match:
            # get the raw name of the folder without _chunk_i
            raw_name = match.group(1)
        
            chunk_folders = [
                (int(re.search(r"_chunk_(\d+)$", f).group(1)), f)
                for f in list_folders
                if re.match(rf"{raw_name}_chunk_\d+$", f)
            ]
            print("found files for ", raw_name)
            chunk_folders.sort(key=lambda x: x[0])
            chunk_folders = [f[1] for f in chunk_folders]
            print("chunk files: ", chunk_folders) 
            process_slams(
                folder_slams, chunk_folders, overlap, max_frames, max_fps) 
                
        else:
           continue  
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process chunked slam files")
    parser.add_argument(
        "--folder_slams",
        type=str,
        default="",
        help="Path to the folder containing the SLAM folders",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap fraction between chunks (0.0 to 1.0)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=1500,
        help="Maximum number of frames per chunk",
    )
    parser.add_argument(
        "--max_fps",
        type=float,
        default=15.0,
        help="Maximum frames per second",
    )
    args = parser.parse_args()
    main(args.folder_slams, args.overlap)    
    