# GPT-2 分布式推理 Demo 项目 (v7)

本项目基于 GPT-2 蒸馏后的小模型 `gpt2_student_v2`，实现了 PyTorch 多进程分布式推理（Distributed Inference），  
适用于 CPU + GPU 混合场景，验证多进程并行加速能力。

🚀 项目亮点：
- 使用 `torch.distributed` + `torchrun` 实现多进程分布式推理
- 在单卡（GPU）设备中，支持 GPU+CPU 混合推理（rank 0 用 GPU，其余 rank 用 CPU）
- 演示多进程输出分配、时间统计与进程间通信（`all_gather_object`）
- 可扩展为分布式推理服务或嵌入 Web 服务中

---

## 📁 项目结构

```plaintext
python7_dist_infer_demo/
├── dist_infer_demo.py           # 多进程推理主程序（torchrun 执行）
├── dist_infer_demo_v2.py        # 加入进程通信版本（all_gather_object）
├── launch.sh                    # 启动脚本（WSL/Linux 下执行）
├── launch_v2.sh                 # 通信增强版启动脚本
└── gpt2_student_v2/             # ✅ 使用的微调小模型
```
📌 以下目录为历史项目，仅保留参考，**本项目未使用**：

- `go_api/`：Go语言服务实验
- `python1_basic_training/`：最早的 GPT-2 训练项目
- `python2_onnx_tensorrt_infer/`：ONNX + TensorRT 推理加速
- `python3_distillation/`：模型蒸馏项目
- `python4_pruning/`：模型剪枝项目
- `python5_quantization/`：模型量化项目
- `python6_k8s_deploy/`：基于 Kubernetes 的自动扩缩容部署

---

## ⚙️ 环境准备（WSL）

建议在 Linux / WSL 中运行，使用 Conda 或 venv 虚拟环境。

### 安装依赖

```bash
pip install -r requirements_dist_infer.txt --index-url https://download.pytorch.org/whl/cu118
```

requirements_dist_infer.txt:
```bash
torch==2.6.0+cu118       # ✅ 使用 PyTorch 官方 CUDA 11.8 版本
transformers==4.36.2     # ✅ 与 HuggingFace GPT-2 兼容
flask==2.2.5             # ✅ 若后续添加 Web 接口服务
numpy==1.26.4            # ✅ 推理输出处理所需
```

---

## 🚀 分布式推理运行（多进程）

### 方式一：基本多进程推理

```bash
bash launch.sh
```

### 方式二：带通信输出的推理（推荐）

```bash
bash launch_v2.sh
```
运行后，每个进程将输出自己处理的 prompt 及推理耗时，rank=0 会汇总所有输出：

示例输出：
```bash
[Rank 0] 📝 Prompt: Hello world
[Rank 0] 🔹 Predicted Token ID: 198
[Rank 0] 🔹 Predicted Token Text:
[Rank 0] ⏱ 推理耗时: 473.54 ms
...
```

---

## 🔧 可扩展方向

- 部署为 Flask Web 接口：`dist_infer_api_server.py`
- 模拟 HPA 弹性部署（可与 `python6_k8s_deploy` 联动）
- 加入 FP16 或 int8 推理模块（结合剪枝/量化项目）

---

## 📜 License

This project is licensed under the MIT License.
