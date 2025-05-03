#!/bin/bash

# 启动 torchrun 多进程分布式推理
# 默认使用 4 个进程（你可以根据 CPU 核心数量自行调整）

WORLD_SIZE=4
MODEL_DIR="./gpt2_student_v2"

torchrun --nproc_per_node=$WORLD_SIZE dist_infer_demo.py
