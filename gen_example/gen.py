import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Your existing BODY_COARSE_POSE dictionary here...
BODY_COARSE_POSE = {
    "LEFT_ARM_UPPER": range(2, 4),
    "LEFT_ARM_LOWER": range(3, 5),
    "RIGHT_ARM_UPPER": range(5, 7),
    "RIGHT_ARM_LOWER": range(6, 8),
    "HEAD": range(25, 95),
    "LEFT_THUMB_LOWER": [95, 96, 97],
    "LEFT_THUMB_UPPER": [97, 98, 99],
    "LEFT_INDEX_LOWER": [95, 100, 101],
    "LEFT_INDEX_UPPER": [101, 102, 103],
    "LEFT_MIDDLE_LOWER": [95, 104, 105],
    "LEFT_MIDDLE_UPPER": [105, 106, 107],
    "LEFT_RING_LOWER": [95, 108, 109],
    "LEFT_RING_UPPER": [109, 110, 111],
    "LEFT_LITTLE_LOWER": [95, 112, 113],
    "LEFT_LITTLE_UPPER": [113, 114, 115],
    "RIGHT_THUMB_LOWER": [116, 117, 118],
    "RIGHT_THUMB_UPPER": [118, 119, 120],
    "RIGHT_INDEX_LOWER": [116, 121, 122],
    "RIGHT_INDEX_UPPER": [122, 123, 124],
    "RIGHT_MIDDLE_LOWER": [116, 125, 126],
    "RIGHT_MIDDLE_UPPER": [126, 127, 128],
    "RIGHT_RING_LOWER": [116, 129, 130],
    "RIGHT_RING_UPPER": [130, 131, 132],
    "RIGHT_LITTLE_LOWER": [116, 133, 134],
    "RIGHT_LITTLE_UPPER": [134, 135, 136],
}

# Define connections for OpenPose Body25 model
BODY_25_PAIRS = [
    (1, 0), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (8, 12),
    (11, 12), (12, 13), (0, 14), (0, 15), (14, 16), (15, 17), (0, 16), (16, 18)
]

# Define connections for hand model
HAND_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# Define connections for face model (simplified)
FACE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16)
    # Add more facial feature connections as needed
]


def extract_coarse_pose(pose_data):
    coarse_pose = {}
    for body_part, indices in BODY_COARSE_POSE.items():
        part_coords = [pose_data[i:i + 2] for i in range(0, len(pose_data), 3) if i // 3 in indices]
        if part_coords:
            coarse_pose[body_part] = np.mean(part_coords, axis=0)
    return coarse_pose


def scale_pose(pose, scale_factor):
    scaled_pose = {}
    for part, coord in pose.items():
        scaled_pose[part] = [coord[0] * scale_factor, coord[1] * scale_factor]
    return scaled_pose


def extract_raw_pose(pose_data):
    raw_pose = {}
    for i in range(0, len(pose_data), 3):
        if pose_data[i] != 0 or pose_data[i + 1] != 0:  # Only include non-zero coordinates
            raw_pose[f"point_{i // 3}"] = [pose_data[i], pose_data[i + 1]]
    return raw_pose


def plot_pose(pose, connections=None, is_raw=False):
    if connections:
        for start, end in connections:
            if is_raw:
                start_key = f"point_{start}"
                end_key = f"point_{end}"
            else:
                start_key = start
                end_key = end

            if start_key in pose and end_key in pose:
                plt.plot([pose[start_key][0], pose[end_key][0]],
                         [pose[start_key][1], pose[end_key][1]], 'b-', linewidth=8)
    for part, coord in pose.items():
        plt.plot(coord[0], coord[1], 'ro', markersize=16)


def process_file(file_path, coarse_output_dir, raw_output_dir):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract keypoint data
    person = data['people'][0]
    pose_data = person['pose_keypoints_2d'] + person['face_keypoints_2d'] + \
                person['hand_left_keypoints_2d'] + person['hand_right_keypoints_2d']

    # Extract coarse and raw poses
    coarse_pose = extract_coarse_pose(pose_data)
    raw_pose = extract_raw_pose(pose_data)

    # Calculate scaling factor
    original_width = max(coord[0] for coord in coarse_pose.values()) - min(coord[0] for coord in coarse_pose.values())
    desired_width = 400
    scale_factor = desired_width / original_width

    # Scale poses
    scaled_coarse_pose = scale_pose(coarse_pose, scale_factor)
    scaled_raw_pose = scale_pose(raw_pose, scale_factor)

    # Define connections for coarse pose
    coarse_connections = [
        ("LEFT_ARM_UPPER", "LEFT_ARM_LOWER"),
        ("RIGHT_ARM_UPPER", "RIGHT_ARM_LOWER"),
        ("LEFT_THUMB_LOWER", "LEFT_THUMB_UPPER"),
        ("LEFT_INDEX_LOWER", "LEFT_INDEX_UPPER"),
        ("LEFT_MIDDLE_LOWER", "LEFT_MIDDLE_UPPER"),
        ("LEFT_RING_LOWER", "LEFT_RING_UPPER"),
        ("LEFT_LITTLE_LOWER", "LEFT_LITTLE_UPPER"),
        ("RIGHT_THUMB_LOWER", "RIGHT_THUMB_UPPER"),
        ("RIGHT_INDEX_LOWER", "RIGHT_INDEX_UPPER"),
        ("RIGHT_MIDDLE_LOWER", "RIGHT_MIDDLE_UPPER"),
        ("RIGHT_RING_LOWER", "RIGHT_RING_UPPER"),
        ("RIGHT_LITTLE_LOWER", "RIGHT_LITTLE_UPPER"),
    ]

    # Combine all connections for raw pose
    raw_connections = BODY_25_PAIRS + [(p[0] + 25, p[1] + 25) for p in FACE_PAIRS] + \
                      [(p[0] + 25 + 70, p[1] + 25 + 70) for p in HAND_PAIRS] + \
                      [(p[0] + 25 + 70 + 21, p[1] + 25 + 70 + 21) for p in HAND_PAIRS]

    # Calculate figure dimensions
    fig_width = desired_width / 72  # Convert pixels to inches (assuming 72 DPI)
    fig_height = fig_width * (max(coord[1] for coord in scaled_coarse_pose.values()) -
                              min(coord[1] for coord in scaled_coarse_pose.values())) / desired_width

    # Plot and save coarse pose
    plt.figure(figsize=(fig_width, fig_height))
    # plt.gca().set_facecolor('lightgray')  # 设置背景颜色
    # plt.gcf().patch.set_facecolor('lightgray')  # 设置figure背景颜色
    plot_pose(scaled_coarse_pose, coarse_connections, is_raw=False)
    plt.gca().invert_yaxis()
    plt.axis('off')
    coarse_output_file = os.path.join(coarse_output_dir,
                                      f"{os.path.splitext(os.path.basename(file_path))[0]}_coarse.png")
    plt.savefig(coarse_output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Plot and save raw pose
    plt.figure(figsize=(fig_width, fig_height))
    plot_pose(scaled_raw_pose, raw_connections, is_raw=True)
    plt.gca().invert_yaxis()
    plt.axis('off')
    raw_output_file = os.path.join(raw_output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_raw.png")
    plt.savefig(raw_output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    input_dir = r"E:\Project\Ham2Pose\data\hamnosys\keypoints\10334"
    coarse_output_dir = r".\output_coarse"
    raw_output_dir = r".\output_raw"

    # Create output directories if they don't exist
    os.makedirs(coarse_output_dir, exist_ok=True)
    os.makedirs(raw_output_dir, exist_ok=True)

    # Process all JSON files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            process_file(file_path, coarse_output_dir, raw_output_dir)
            print(f"Processed {filename}")


if __name__ == "__main__":
    main()
