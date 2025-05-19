import os 
import re
import argparse
import shutil
import numpy as np
from scipy.spatial import procrustes
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA


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

def is_chunk_robust(chunk_poses, keyframes_folder, max_frames, max_fps, is_last_chunk=False):
    """
    Check if a SLAM chunk is robust based on criteria:
    1. Has keyframes with time > 80% of max_frames (except for last chunk)
    2. No gap of more than 15s without keyframes
    
    Returns: (is_robust, reason)
    """
    if not chunk_poses:
        return False, "No poses in chunk"
    
    # Get all keyframe times
    keyframe_times = []
    for file in os.listdir(keyframes_folder):
        if file.endswith('.png'):
            try:
                time = float(file.replace('.png', ''))
                keyframe_times.append(time)
            except ValueError:
                pass
    
    if not keyframe_times:
        return False, "No keyframes found"
    
    keyframe_times.sort()
    
    # Check if we have keyframes with time > 80% of max_frames
    if not is_last_chunk:
        max_expected_time = max_frames / max_fps
        has_late_keyframe = any(t > 0.8 * max_expected_time for t in keyframe_times)
        if not has_late_keyframe:
            return False, f"No keyframe found after {0.8 * max_expected_time:.1f}s (80% of max frames)"
    
    # Check for gaps > 15s between keyframes
    for i in range(1, len(keyframe_times)):
        gap = keyframe_times[i] - keyframe_times[i-1]
        if gap > 15.0:
            return False, f"Gap of {gap:.1f}s between keyframes (max allowed: 15s)"
    
    # If we reach here, the chunk is robust
    return True, "Chunk is robust"

def write_trajectory_file(output_folder, video_name, trajectory_idx, poses, keyframe_mappings):
    """
    Write a trajectory file and copy corresponding keyframes
    """
    # Create trajectory-specific folders
    trajectory_folder = f"{output_folder}_trajectory_{trajectory_idx}"
    keyframes_folder = os.path.join(trajectory_folder, "keyframes")
    trajectory_file = os.path.join(trajectory_folder, f"{video_name}_trajectory_{trajectory_idx}.txt")

    
    os.makedirs(trajectory_folder, exist_ok=True)
    os.makedirs(keyframes_folder, exist_ok=True)
    
    # Write trajectory file
    with open(trajectory_file, 'w') as f:
        for t, x, y, z, qx, qy, qz, qw in poses:
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
    
    # Copy keyframes
    copied_keyframes = 0
    for src_keyframe_path, adjusted_time in keyframe_mappings.items():
        if os.path.exists(src_keyframe_path):
            dst_keyframe_path = os.path.join(keyframes_folder, f"{adjusted_time}.png")
            shutil.copy2(src_keyframe_path, dst_keyframe_path)
            copied_keyframes += 1
    
    return trajectory_file, len(poses), copied_keyframes
import numpy as np

def find_rigid_transform_2d(source_points, target_points):
    """
    Find a rigid transformation (rotation + translation, no scaling) that best maps
    source_points to target_points in 2D using Procrustes analysis.

    Args:
        source_points: Array of shape (n, 2) containing source point coordinates
        target_points: Array of shape (n, 2) containing target point coordinates

    Returns:
        tuple: (rotation_matrix, translation_vector, rmse) where:
            - rotation_matrix: 2x2 rotation matrix
            - translation_vector: 2x1 translation vector
            - rmse: root mean squared error of the alignment
    """
    source_points = np.array(source_points)
    target_points = np.array(target_points)

    if len(source_points) < 2:
        print("Warning: Need at least 2 points for reliable rigid transform in 2D. Using identity.")
        return np.eye(2), np.zeros(2), float('inf')

    # Compute centroids
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # Center the points
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Compute covariance matrix
    covariance = target_centered.T @ source_centered

    # SVD
    U, _, Vt = np.linalg.svd(covariance)

    # Handle reflection
    det = np.linalg.det(U @ Vt)
    # If det is negative, we need to reflect the points
    if det < 0:
        Vt[1, :] *= -1
        
    R = U @ np.diag([1, np.sign(det)]) @ Vt

    # Translation
    t = target_centroid - (R @ source_centroid)

    # Compute RMSE
    transformed = (R @ source_points.T).T + t
    rmse = np.sqrt(np.mean(np.sum((transformed - target_points)**2, axis=1)))

    return R, t, rmse


def interpolate_poses(poses, target_times):
    """
    Linearly interpolate poses at specific target times
    
    Args:
        poses: List of (t, x, y, z, qx, qy, qz, qw) tuples 
        target_times: List of times at which to interpolate
    
    Returns:
        List of interpolated (t, x, y, z, qx, qy, qz, qw) tuples
    """
    if len(poses)==0 or len(target_times)==0:
        print("No poses or target times provided for interpolation.")
        return []
    
    # Extract times and separate components
    times = np.array([p[0] for p in poses])
    positions = np.array([(p[1], p[2], p[3]) for p in poses])  # x, y, z
    quaternions = np.array([(p[4], p[5], p[6], p[7]) for p in poses])  # qx, qy, qz, qw
    
    # Create interpolation functions for positions
    if len(poses) > 1:
        interp_x = interp1d(times, positions[:, 0], bounds_error=False, fill_value="extrapolate")
        interp_y = interp1d(times, positions[:, 1], bounds_error=False, fill_value="extrapolate")
        interp_z = interp1d(times, positions[:, 2], bounds_error=False, fill_value="extrapolate")
        
        # For quaternions, use linear interpolation and renormalize
        interp_qx = interp1d(times, quaternions[:, 0], bounds_error=False, fill_value="extrapolate")
        interp_qy = interp1d(times, quaternions[:, 1], bounds_error=False, fill_value="extrapolate")
        interp_qz = interp1d(times, quaternions[:, 2], bounds_error=False, fill_value="extrapolate")
        interp_qw = interp1d(times, quaternions[:, 3], bounds_error=False, fill_value="extrapolate")
        
        interpolated_poses = []
        for t in target_times:
            # Interpolate position components
            x = float(interp_x(t))
            y = float(interp_y(t))
            z = float(interp_z(t))
            
            # Interpolate quaternion components and normalize
            qx = float(interp_qx(t))
            qy = float(interp_qy(t))
            qz = float(interp_qz(t))
            qw = float(interp_qw(t))
            
            # Normalize quaternion
            qnorm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
            if qnorm > 1e-10:
                qx /= qnorm
                qy /= qnorm
                qz /= qnorm
                qw /= qnorm
            else:
                qx, qy, qz, qw = 0, 0, 0, 1
            
            interpolated_poses.append((t, x, y, z, qx, qy, qz, qw))
    else:
        # If we have just one pose, return it for all target times
        interpolated_poses = [(t, poses[0][1], poses[0][2], poses[0][3], 
                              poses[0][4], poses[0][5], poses[0][6], poses[0][7]) 
                             for t in target_times]
    
    return interpolated_poses

def process_slams(folder_slams, chunk_folders, overlap, max_frames, max_fps):
    """
    Process SLAM data from multiple chunk folders and merge them into consolidated trajectories.
    Performs PCA on each chunk to reduce to 2D and uses z=0 for all points.
    Creates separate trajectories when encountering non-robust chunks.
    
    Args:
        folder_slams (str): Path to the folder containing chunk-specific SLAM data folders
        chunk_folders (list): List of chunk folder names to process
        overlap (float): Proportion of overlapping frames between consecutive chunks (0.0 to 1.0)
        max_frames (int): Maximum number of frames per chunk
        max_fps (float): Maximum frames per second
        
    Output:
        Creates one or more trajectory files depending on robustness of chunks
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
    
    # Calculate chunk duration in seconds (except potentially the last chunk)
    chunk_duration = max_frames / max_fps
    overlap_duration = chunk_duration * overlap
    
    # Process each chunk and create separate trajectories when needed
    trajectories = []  # List of (poses, keyframe_mappings)
    current_trajectory_poses = []
    current_keyframe_mappings = {}
    
    prev_chunk_poses = []
    last_successful_chunk_idx = -1
    skipped_chunks = []
    
    # Transformations applied to each chunk (scaling, rotation, translation)
    chunk_transforms = [None] * len(chunk_folders)
    
    # Store PCA models for each chunk
    pca_models = {}
    
    for i, chunk_name in enumerate(chunk_folders):
        chunk_folder = os.path.join(folder_slams, chunk_name)
        slam_file = os.path.join(chunk_folder, f"{chunk_name}.txt")
        keyframes_folder = os.path.join(chunk_folder, "keyframes")
        
        if not os.path.exists(slam_file):
            print(f"SLAM file not found: {slam_file}")
            continue
        
        # Read SLAM data from file
        chunk_poses = []
        with open(slam_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                # Handle both comma and space separated values
                parts = re.split(r'[,\s]+', line.strip())
                if len(parts) == 8:  # t,x,y,z,qx,qy,qz,qw
                    t, x, y, z, qx, qy, qz, qw = map(float, parts)
                    chunk_poses.append((t, x, y, z, qx, qy, qz, qw))
        
        # Check if this chunk is robust
        is_last_chunk = (i == len(chunk_folders) - 1)
        is_robust, reason = is_chunk_robust(chunk_poses, keyframes_folder, max_frames, max_fps, is_last_chunk)
        
        if not is_robust:
            print(f"Chunk {chunk_name} is not robust: {reason}")
            skipped_chunks.append(chunk_name)
            
            # Save current trajectory if it has any data
            if current_trajectory_poses:
                trajectories.append((current_trajectory_poses, current_keyframe_mappings))
                current_trajectory_poses = []
                current_keyframe_mappings = {}
                prev_chunk_poses = []
                last_successful_chunk_idx = -1
            continue
        
        # Extract positions for PCA
        positions = np.array([(p[1], p[2], p[3]) for p in chunk_poses])
        
        # Perform PCA to reduce to 2D
        if len(positions) >= 2:  # Need at least 2 points for PCA
            pca = PCA(n_components=2)
            positions_2d = pca.fit_transform(positions)
            
            # Store the PCA model for this chunk
            pca_models[i] = pca
            
            # Replace original poses with PCA-transformed poses (z=0)
            pca_poses = []
            for j, (t, _, _, _, qx, qy, qz, qw) in enumerate(chunk_poses):
                # Use the 2D coordinates from PCA with z=0
                pca_poses.append((t, positions_2d[j][0], positions_2d[j][1], 0.0, qx, qy, qz, qw))
            
            chunk_poses = pca_poses
        else:
            print(f"Warning: Chunk {chunk_name} has fewer than 2 poses. Cannot perform PCA.")
            # Just set z=0 for these poses
            pca_poses = [(t, x, y, 0.0, qx, qy, qz, qw) for t, x, y, z, qx, qy, qz, qw in chunk_poses]
            chunk_poses = pca_poses
        
        # Calculate time offset for this chunk
        # If starting a new trajectory or it's the first chunk, start at t=0
        # Otherwise, calculate offset based on previous successful chunks
        if last_successful_chunk_idx == -1:
            time_offset = 0
        else:
            time_offset = (i - last_successful_chunk_idx) * chunk_duration - (i - last_successful_chunk_idx) * overlap_duration + last_time_offset
        
        # ----- ALIGNMENT APPROACH FOR 2D POSITIONS AFTER PCA -----
        # If we need to align with previous chunk
        if last_successful_chunk_idx >= 0 and prev_chunk_poses:
            # Define overlap regions in time
            overlap_start_prev = chunk_duration - overlap_duration
            overlap_end_prev = chunk_duration
            
            # Get poses from previous chunk in the overlap region
            prev_overlap_poses = [pose for pose in prev_chunk_poses 
                                 if overlap_start_prev <= pose[0] <= overlap_end_prev]
            
            # Get poses from current chunk in the overlap region
            current_overlap_poses = [pose for pose in chunk_poses 
                                    if 0 <= pose[0] <= overlap_duration]
            
            # If either overlap region has no poses, use endpoints
            if not prev_overlap_poses:
                prev_overlap_poses = [prev_chunk_poses[-1]]
            if not current_overlap_poses:
                current_overlap_poses = [chunk_poses[0]]
                
            # Create a set of corresponding points for alignment
            # First, create evenly spaced time points in the overlap region
            num_samples = min(20, len(prev_overlap_poses), len(current_overlap_poses))
            if num_samples < 2:
                num_samples = 2  # Minimum for 2D
                
            # Sample times in the overlap region
            overlap_times = np.linspace(0, overlap_duration, num_samples)
            
            # Get interpolated poses at these times
            prev_times = [t + overlap_start_prev for t in overlap_times]
            curr_times = overlap_times
            prev_interp_poses = interpolate_poses(prev_overlap_poses, prev_times)
            curr_interp_poses = interpolate_poses(current_overlap_poses, curr_times)
            
            # Extract 2D positions for alignment
            prev_positions_2d = np.array([(p[1], p[2]) for p in prev_interp_poses])
            curr_positions_2d = np.array([(p[1], p[2]) for p in curr_interp_poses])
            
            # Find the 2D similarity transformation (scale, rotation, translation)
            rotation_2d, translation_2d, rmse = find_rigid_transform_2d(
                curr_positions_2d, prev_positions_2d)
            scale = 1 # force similitude to avoid deriving scale from slam
            # Store the transform for this chunk
            chunk_transforms[i] = (scale, rotation_2d, translation_2d)
            
            print(f"Chunk {i} 2D alignment RMSE: {rmse:.4f}, Scale: {scale:.4f}")
            
            # Apply the transformation to all poses in this chunk
            transformed_poses = []
            for t, x, y, z, qx, qy, qz, qw in chunk_poses:
                # Transform the position (scale * R * p + t)
                position_2d = np.array([x, y])
                new_position_2d = scale * (rotation_2d @ position_2d) + translation_2d
                
                # For orientation quaternions, we need to derive a 3D rotation from our 2D rotation
                # The 2D rotation is in the XY plane, so we can extend it to 3D
                # by adding a z-axis that doesn't rotate
                cos_theta = rotation_2d[0, 0]  # Cosine of rotation angle
                sin_theta = rotation_2d[1, 0]  # Sine of rotation angle
                
                # Create quaternion for rotation around Z axis
                rot_angle = np.arctan2(sin_theta, cos_theta)  # Extract rotation angle
                rot_x = 0
                rot_y = 0
                rot_z = np.sin(rot_angle / 2)
                rot_w = np.cos(rot_angle / 2)
                
                rot_quat = normalize_quaternion(np.array([rot_x, rot_y, rot_z, rot_w]))
                pose_quat = np.array([qx, qy, qz, qw])
                
                # Apply rotation to orientation quaternion
                new_quat = quaternion_multiply(rot_quat, pose_quat)
                new_quat = normalize_quaternion(new_quat)
                
                transformed_poses.append((
                    t, 
                    new_position_2d[0], new_position_2d[1], 0.0,  # z is always 0
                    new_quat[0], new_quat[1], new_quat[2], new_quat[3]
                ))
            
            chunk_poses = transformed_poses
        
        # Store current chunk poses for alignment with next chunk
        prev_chunk_poses = chunk_poses
        last_successful_chunk_idx = i
        last_time_offset = time_offset
        
        # Adjust timestamps and add to current trajectory
        for t, x, y, z, qx, qy, qz, qw in chunk_poses:
            adjusted_time = time_offset + t
            
            # For the overlap regions, we want to include:
            # - First half of overlap from the first chunk
            # - Second half of overlap from the second chunk
            include_pose = True
            
            if i > last_successful_chunk_idx and t < overlap_duration:
                # This is in the overlap region of the current chunk
                normalized_pos_in_overlap = t / overlap_duration
                # Only include if in the second half of the overlap
                include_pose = normalized_pos_in_overlap >= 0.5
            elif i < len(chunk_folders) - 1 and t > (chunk_duration - overlap_duration):
                # This is in the overlap region with the next chunk
                normalized_pos_in_overlap = (t - (chunk_duration - overlap_duration)) / overlap_duration
                # Only include if in the first half of the overlap
                include_pose = normalized_pos_in_overlap < 0.5
            
            if include_pose:
                # Make sure z is always 0
                current_trajectory_poses.append((adjusted_time, x, y, 0.0, qx, qy, qz, qw))
                
                # Check and record if this is a keyframe
                keyframe_file = f"{t}.png"
                keyframe_path = os.path.join(keyframes_folder, keyframe_file)
                if os.path.exists(keyframe_path):
                    current_keyframe_mappings[keyframe_path] = adjusted_time
    
    # Add the last trajectory if it has any data
    if current_trajectory_poses:
        trajectories.append((current_trajectory_poses, current_keyframe_mappings))
    
    # Write all trajectories
    print(f"\nProcessing complete for {video_name}:")
    print(f"Total chunks: {len(chunk_folders)}")
    print(f"Skipped chunks: {len(skipped_chunks)} ({', '.join(skipped_chunks)})")
    print(f"Number of trajectories created: {len(trajectories)}")
    
    for idx, (poses, keyframe_mappings) in enumerate(trajectories):
        # Sort poses by adjusted time
        poses.sort(key=lambda x: x[0])
        
        # Ensure Z coordinate is always 0
        poses = [(t, x, y, 0.0, qx, qy, qz, qw) for t, x, y, z, qx, qy, qz, qw in poses]
        
        # Write trajectory file
        trajectory_file, num_poses, num_keyframes = write_trajectory_file(
            output_folder, video_name, idx, poses, keyframe_mappings
        )
        
        print(f"  Trajectory {idx}: {num_poses} poses, {num_keyframes} keyframes")
        print(f"    Saved to: {trajectory_file}")


def main():
    """Main function to process SLAM data from command line arguments"""
    parser = argparse.ArgumentParser(description="Process chunked slam files with PCA")
    parser.add_argument(
        "--folder_slams",
        type=str,
        required=True,
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
    
    folder_slams = args.folder_slams
    overlap = args.overlap
    max_frames = args.max_frames
    max_fps = args.max_fps
    
    list_folders = os.listdir(folder_slams)
    processed_videos = set()
    
    for folder in list_folders:
        match = re.match(r"(.*)_chunk_\d+$", folder)
        if match and folder not in processed_videos:
            # get the raw name of the folder without _chunk_i
            raw_name = match.group(1)
            
            # Find all chunks for this video
            chunk_folders = [
                (int(re.search(r"_chunk_(\d+)$", f).group(1)), f)
                for f in list_folders
                if re.match(rf"{raw_name}_chunk_\d+$", f)
            ]
            print("Found files for", raw_name)
            chunk_folders.sort(key=lambda x: x[0])
            chunk_folders = [f[1] for f in chunk_folders]
            print("Chunk files:", chunk_folders) 
            
            process_slams(folder_slams, chunk_folders, overlap, max_frames, max_fps)
            
            # Mark all these chunks as processed
            processed_videos.update(chunk_folders)


def main_with_config(config):
    """Alternative main function that accepts a config dictionary"""
    root = config["general"]["root"]
    folder_slams = str(root) + config["slam"]["log_path"]
    overlap = config["preprocess"]["overlap"]
    max_frames = config["preprocess"]["max_frames_per_video"]
    max_fps = config["preprocess"]["max_fps"]
    
    list_folders = os.listdir(folder_slams)
    processed_videos = set()
    
    for folder in list_folders:
        match = re.match(r"(.*)_chunk_\d+$", folder)
        if match and folder not in processed_videos:
            # get the raw name of the folder without _chunk_i
            raw_name = match.group(1)
            
            # Find all chunks for this video
            chunk_folders = [
                (int(re.search(r"_chunk_(\d+)$", f).group(1)), f)
                for f in list_folders
                if re.match(rf"{raw_name}_chunk_\d+$", f)
            ]
            print("Found files for", raw_name)
            chunk_folders.sort(key=lambda x: x[0])
            chunk_folders = [f[1] for f in chunk_folders]
            print("Chunk files:", chunk_folders) 
            
            process_slams(folder_slams, chunk_folders, overlap, max_frames, max_fps)
            
            # Mark all these chunks as processed
            processed_videos.update(chunk_folders)


if __name__ == "__main__":
    main()