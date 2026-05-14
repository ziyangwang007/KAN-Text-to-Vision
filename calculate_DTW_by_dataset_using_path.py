#!/usr/bin/env python
# -*-coding:utf-8 -*-
from statistics import mean, median
import sys
import json
# pjm
# gsl
# number
# UPPER

if len(sys.argv) < 2:
    print("path not exist")
    exit(-1)

argv = sys.argv[1]


with open(argv, "r") as f:
    log_input = f.read().strip().split("\n")



ndtw_data = json.loads(log_input[-4].replace("'", '"'))
dtw_data = json.loads(log_input[-3].replace("'", '"'))

print("dtw mean of result", mean(dtw_data.values()))
print("dtw median of result", median(dtw_data.values()))


print("ndtw mean of result", mean(ndtw_data.values()))
print("ndtw median of result", median(ndtw_data.values()))



# 将结果按数据集分组
pjm_scores = []
gsl_scores = []
number_scores = []
upper_scores = []

for key, value in dtw_data.items():
    if key.startswith('pjm_'):
        pjm_scores.append(value)
    elif key.startswith('gsl_'):
        gsl_scores.append(value)
    elif key.isupper():
        upper_scores.append(value)
    elif key.isdigit():
        number_scores.append(value)
    else:
        raise ValueError(f"Unknown dataset: {key}")

# 计算每个数据集的平均分
pjm_average = mean(pjm_scores)
gsl_average = mean(gsl_scores)
number_average = mean(number_scores)
upper_average = mean(upper_scores)

# 打印结果
print(f"DTW PJM dataset average score: {pjm_average:.2f}")
print(f"DTW GSL dataset average score: {gsl_average:.2f}")
print(f"DTW Number (DGS)  dataset average score: {number_average:.2f}")
print(f"DTW UPPER (LSF)  dataset average score: {upper_average:.2f}")

result = f"{pjm_average:.2f}&{gsl_average:.2f}&{number_average:.2f}&{upper_average:.2f}&"

pjm_scores = []
gsl_scores = []
number_scores = []
upper_scores = []

for key, value in ndtw_data.items():
    if key.startswith('pjm_'):
        pjm_scores.append(value)
    elif key.startswith('gsl_'):
        gsl_scores.append(value)
    elif key.isupper():
        upper_scores.append(value)
    elif key.isdigit():
        number_scores.append(value)
    else:
        raise ValueError(f"Unknown dataset: {key}")

# 计算每个数据集的平均分
pjm_average = mean(pjm_scores)
gsl_average = mean(gsl_scores)
number_average = mean(number_scores)
upper_average = mean(upper_scores)
# 打印结果
print(f"nDTW PJM dataset average score: {pjm_average:.2f}")
print(f"nDTW GSL dataset average score: {gsl_average:.2f}")
print(f"nDTW Number (DGS)  dataset average score: {number_average:.2f}")
print(f"nDTW UPPER (LSF)  dataset average score: {upper_average:.2f}")

result += f"{pjm_average:.2f}&{gsl_average:.2f}&{number_average:.2f}&{upper_average:.2f}"
print(result)