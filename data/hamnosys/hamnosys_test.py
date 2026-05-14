"""hamnosys dataset."""
import pickle

import tensorflow_datasets as tfds
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_header import PoseHeader
from pose_format.utils.reader import BufferReader
from tqdm import tqdm

from data.tfds_dataset import process_datum
# import os
# import sys
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
from hamnosys import Hamnosys, SignDatasetConfig, _POSE_HEADERS


# from hamnosys import hamnosys
# from .config import SignDatasetConfig


def pose_normalization_info(pose_header: PoseHeader):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(
            p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
            p2=("POSE_LANDMARKS", "LEFT_SHOULDER")
        )

    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(
            p1=("BODY_135", "RShoulder"),
            p2=("BODY_135", "LShoulder")
        )

    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(
            p1=("pose_keypoints_2d", "RShoulder"),
            p2=("pose_keypoints_2d", "LShoulder")
        )

    raise ValueError("Unknown pose header schema for normalization")





# class HamnosysTest(tfds.testing.DatasetBuilderTestCase):
#   """Tests for hamnosys dataset."""
#   DATASET_CLASS = hamnosys.Hamnosys
#   # SPLITS = {
#   #     'train': 3,  # Number of fake train example
#   #     'test': 1,  # Number of fake test example
#   # }
#   name = "hamnosys"
#   config = SignDatasetConfig(name=name,
#                              version="1.0.0",  # Specific version
#                              include_video=False,  # Download and load dataset videos
#                              fps=25,  # Load videos at constant fps
#                              include_pose="openpose")  # Download and load openpose estimation
#   tfds_dataset = tfds.load(name=name, builder_kwargs=dict(config=config),
#                            split="train")
#
#   with open(hamnosys._POSE_HEADERS["openpose"], "rb") as f:
#       pose_header = PoseHeader.read(BufferReader(f.read()))
#
#   normalization_info = pose_normalization_info(pose_header)
#   data = [process_datum(datum, pose_header, normalization_info)
#           for datum in tqdm(tfds_dataset)]
#   h = 0


if __name__ == '__main__':
    # tfds.testing.test_main()
    name = "hamnosys"
    config = SignDatasetConfig(name="hamnosys",
                               version="1.0.0",  # Specific version
                               include_video=False,  # Download and load dataset videos
                               fps=25,  # Load videos at constant fps
                               include_pose="openpose")  # Download and load openpose estimation
    # tfds_dataset = tfds.load(name="hamnosys", builder_kwargs=dict(config=config),
    #                          split="train")
    with open("E:\\Project\\Ham2Pose\\temp\\train\\train_dataset.pkl", "rb") as f:
        tfds_dataset = pickle.load(f)
    print(tfds_dataset)
    with open(_POSE_HEADERS["openpose"], "rb") as f:
        pose_header = PoseHeader.read(BufferReader(f.read()))

    normalization_info = pose_normalization_info(pose_header)
    # data = [process_datum(datum, pose_header, normalization_info)
    #         for datum in tqdm(tfds_dataset)]
    # print(tfds_dataset[0])
    # print(tfds_dataset[0]["pose"]["data"][0].shape)
    # print(process_datum(tfds_dataset[0], pose_header, normalization_info))
    # print(tfds_dataset[1]["pose"]["data"].shape)
    # print(tfds_dataset[2]["pose"]["data"].shape)
    # print(tfds_dataset[3]["pose"]["data"].shape)
    print(tfds_dataset)
    data = [process_datum(datum, pose_header, normalization_info)
            for datum in tqdm(tfds_dataset)]
    print(data)