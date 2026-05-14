#!/usr/bin/env python
# -*-coding:utf-8 -*-
# 根据原始数据删除视频的fade in效果

import json
import os
import tqdm

currrnt_path = os.path.dirname(os.path.abspath(__file__))

old_keypoints = os.listdir("keypoints_openpose_provide_by_author")
new_keypoints = os.listdir("keypoints_new")

os.makedirs("keypoints_remove_fade_in", exist_ok=True)
dir_path = os.path.join(currrnt_path, "keypoints_remove_fade_in")
new_not_exist = [keypoint for keypoint in old_keypoints if keypoint not in new_keypoints]
print(f"new_not_exist: {new_not_exist}")

for keypoint in tqdm.tqdm(old_keypoints):
    if keypoint in new_not_exist:
        continue
    keypoint_path = os.path.join(currrnt_path, "keypoints", keypoint)
    new_keypoint_path = os.path.join(dir_path, keypoint)
    os.makedirs(new_keypoint_path, exist_ok=True)
    k = os.listdir(keypoint_path)
    for i in range(len(k)):
        file_name = k[i]
        if not file_name.startswith(keypoint):
            print(f"Pass {file_name}")
            continue
        expected_file_name = f"{keypoint}_{'_'.join(file_name.split('_')[-2:])}"
        file_path = os.path.join(keypoint_path, expected_file_name)
        dwpose_file_path = os.path.join(currrnt_path, "keypoints_new", keypoint, f"{keypoint}_{'_'.join(file_name.split('_')[-2:])}")
        try:
            with open(file_path, "r", encoding='utf8') as f:
                data = json.load(f)

            if not len(data["people"]) >= 1:  # 如果没有检测到人
                with open(os.path.join(new_keypoint_path, file_name), "w", encoding='utf8') as f:
                    json.dump({"version": 1.3, "people": []}, f)
            else:  # 如果检测到人
                with open(dwpose_file_path, "r", encoding='utf8') as f:
                    dwpose_data = json.load(f)
                with open(os.path.join(new_keypoint_path, file_name), "w", encoding='utf8') as f:
                    json.dump(dwpose_data, f)
        except Exception as e:
            print(f"Error in {file_path}: {e}")