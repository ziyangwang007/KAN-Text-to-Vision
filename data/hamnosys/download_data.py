#!/usr/bin/env python
# -*-coding:utf-8 -*-

import json
import os
import requests
import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Create a directory to save the downloaded videos
os.makedirs('videos', exist_ok=True)

downloaded_files = os.listdir("videos")

time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def download_video(entry, name):

    video_url = entry['video_frontal']
    video_id = name
    if video_url:
        try:
            response = requests.get(video_url, stream=True)
            response.raise_for_status()

            # Define the local file path
            file_path = os.path.join('videos', f"{video_id}.mp4")
            if f"{video_id}.mp4" in downloaded_files:
                return

            # Write the content to a file
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {video_id}: {e}")
    else:
        print(f"No video URL for ID {video_id}")


def main(data, max_workers=5):
    # Filter out entries without a valid video URL

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of future tasks
        future_to_entry = {executor.submit(download_video, data[d], d): d for d in data}

        # Use tqdm to display the progress bar
        for future in tqdm.tqdm(as_completed(future_to_entry), total=len(future_to_entry)):
            pass


# with open("data.json", "r", encoding="utf8") as f:
#     data = json.load(f)
#
# # Example usage
#
# main(data, max_workers=3)  # You can adjust the number of workers for parallel downloads
# failed_files.close()

max_frame = 0
dirs = os.listdir("keypoints")
frame_number_list = []
for d in dirs:
    files = len(os.listdir(f"keypoints/{d}"))
    frame_number_list.append(files)

import matplotlib.pyplot as plt
# 画直方图
plt.hist(frame_number_list, bins=20, alpha=0.75, color='b')
plt.xlabel('Frame Number')
plt.ylabel('Frequency')
plt.title('Frame Number Distribution')
plt.show()

