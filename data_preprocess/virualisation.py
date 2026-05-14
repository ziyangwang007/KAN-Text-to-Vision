import json
import numpy as np
import matplotlib.pyplot as plt
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer


with open("../data/hamnosys/keypoints/2209/2209_000000000000_keypoints.json", "r", encoding="utf-8") as f:
    pose_data = json.load(f)
    keypoints = pose_data["people"][0]

    # 提取关键点
    pose_keypoints = keypoints["pose_keypoints_2d"][0: 8 * 3]
    hand_left_keypoints = keypoints["hand_left_keypoints_2d"]
    hand_right_keypoints = keypoints["hand_right_keypoints_2d"]
    face_keypoints = keypoints["face_keypoints_2d"]

    keypoints = keypoints["pose_keypoints_2d"] + keypoints["hand_left_keypoints_2d"] + keypoints[
        "hand_right_keypoints_2d"] + keypoints["face_keypoints_2d"]

    # remove 0s
    keypoints = [i for i in keypoints if i != 0]

    # flip Y axis
    y_max = max(keypoints[1::3])
    keypoints[1::3] = [y_max - y for y in keypoints[1::3]]

    # visualize the keypoints
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(np.array(keypoints[0::3]), np.array(keypoints[1::3]), c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
