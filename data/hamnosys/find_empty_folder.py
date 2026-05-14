import os

def find_directories_with_few_files(path, threshold=10):
    few_files_dirs = []
    for root, dirs, files in os.walk(path):
        file_count = len([f for f in files if os.path.isfile(os.path.join(root, f))])
        if file_count < threshold:
            few_files_dirs.append((root, file_count))
    return few_files_dirs

# 替换为您的 keypoints 目录的实际路径
keypoints_path = "./keypoints"

directories_with_few_files = find_directories_with_few_files(keypoints_path)

if directories_with_few_files:
    print(f"以下是包含少于10个文件的目录:")
    for dir, count in directories_with_few_files:
        print(f"{dir}: {count} 个文件")
else:
    print("未找到包含少于10个文件的目录。")

print(f"\n总共找到 {len(directories_with_few_files)} 个目录包含少于10个文件。")