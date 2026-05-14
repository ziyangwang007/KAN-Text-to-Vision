import re
from collections import Counter

with open("failed.log", "r", encoding="utf8") as f:
    failed_files = f.read()

matches = re.findall(r'keypoints\\(\w+\d+)', failed_files)

# 字典统计出现次数
counter = Counter(matches)
print(counter)  # Counter({'gsl_861': 71, 'gsl_546': 65, 'gsl_562': 64, 'gsl_470': 59, 'gsl_836': 1})

# multiscale graph
