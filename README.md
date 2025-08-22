# MultiwayQwen：基于 Qwen2.5VL 改进的视觉问答（VQA）系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.50-yellow)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 项目简介

**MultiwayQwen** 是一个旨在提升多模态大语言模型（MLLM）视觉问答（VQA）能力的研究与工程实践项目。

本项目的核心是基于开源的 **Qwen2.5VL (3B)** 模型，设计并实现了一种名为 **`Multiway` 的新型融合模块**。该模块借鉴了 `VLMo/MOME` 思想，通过模态专属处理（视觉MoE与文本MoE）和共享注意力（MLA）机制，来促进视觉与语言特征的深度交互。为探索模型效率与性能的边界，本项目还实验性地集成了 **Multi-Head Latent Attention (MLA)** 和 **DeepSeekMoE** 等前沿Transformer组件。

除了模型层面的创新，本项目还包含一个**完整的前后端分离VQA应用**。后端采用 **FastAPI** 构建，实现了高效的异步推理和流式响应；前端则基于 **React** 构建，提供了一个支持图文混合输入的用户友好对话界面。

## 🚀 快速开始

### 1. 克隆仓库

```
git clone https://github.com/Dncpeq/Visual-Question-Answering-System.git
cd Visual-Question-Answering-System
```

### 2. 环境安装

建议使用 `conda` 或 `venv` 创建独立的Python环境。

```
pip install -r requirements.txt
```

### 3. 模型准备

1. **下载基础模型**: 从Hugging Face Hub下载预训练的 `Qwen/Qwen2.5-VL-3B-Instruct` 模型权重。
2. **放置模型**: 将模型移至项目根目录下并确保文件夹名称为`Qwen2.5-VL-3B-Instruct`
3. **准备自定义模型目录**: 在项目根目录下创建一个名为 `MultiwayQwen` 的空文件夹，用于存放初始化后的新模型。

最终目录结构应如下所示：

```
Visual-Question-Answering-System/
├── MultiwayQwen/              # (空) 用于存放初始化后的新模型
├── Qwen2.5-VL-3B-Instruct/    # 存放下载的Qwen2.5VL基础模型
├── datasets/                  # 存放数据集 (见后续步骤)
├── init_model.py
├── main.py
├── training.py
└── ...
```

### 4. 数据集准备

1. **创建目录**: 在项目根目录下创建一个 `datasets` 文件夹。
2. **下载数据**: 根据 `training.py` 脚本中的配置，下载所需的公开数据集，并放置在 `datasets` 文件夹下。本项目使用的主要数据集包括：
   - `jackyhate/text-to-image-2M`
   - `HuggingFaceM4/Docmatix`
   - `Congliu/Chinese-DeepSeek-R1-Distill-data-110k`
   - `oscar-corpus/mOSCAR`
3. **修改路径**: 如有需要，请在 `training.py` 中修改对应的数据加载路径。

### 5. 模型初始化

在开始训练之前，我们需要运行一个初始化脚本。这个脚本会将预训练好的 `Qwen2.5VL` 权重加载到我们新的 `MultiwayQwen` 模型架构中，为接下来的微调做好准备。

1. **配置脚本**: 打开 `init_model.py` 文件。

2. **修改路径**: 确保脚本顶部的 `model_dir` 变量指向您下载的基础模型路径 (`"Qwen2.5-VL-3B-Instruct"`)，并配置 `weight_files` 列表包含该目录下所有的 `.safetensors` 权重文件。同时，确保脚本最后的 `model.save_pretrained()` 保存路径指向您创建的 `MultiwayQwen` 文件夹。

3. **运行脚本**:

   ```
   python init_model.py
   ```

   运行成功后，`MultiwayQwen/` 目录下将会生成一个完整的、可以开始训练的模型（包含配置文件和权重文件）。

### 6. 模型训练

本项目使用 `transformers.Trainer` 和 `accelerate` 进行模型训练。

1. **配置训练**: 打开 `training.py` 文件，您可以根据需要调整超参数，如 `LEARNING_RATE`, `TRAINING_STEPS`, `PER_DEVICE_TRAIN_BATCH_SIZE` 等。确保 `MODEL_DIR` 变量指向 `MultiwayQwen`。

2. **开始训练**: 使用 `accelerate` 启动训练脚本。如果您有多张GPU，它将自动处理分布式训练。

   ```
   # 根据你的机器配置accelerate
   accelerate config
   
   # 启动训练
   accelerate launch training.py
   ```

   训练脚本配置为仅训练 `multiway` 模块的参数。训练好的模型检查点（checkpoints）将保存在 `MultiwayQwen/training_output/` 目录下。

### 7. 服务启动

训练完成后，可以启动后端API服务来启动VQA的后端服务。

```
# 确保你的模型权重路径正确
# main.py 默认会从 MultiwayQwen/ 目录加载模型
uvicorn main:app --host 0.0.0.0 --port 8000
```

服务启动后，打开浏览器访问 `http://localhost:8000` 即可与您的VQA系统进行交互。

## 📜 许可证

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 开源。

## 🙌 致谢

- 感谢**阿里巴巴通义千问团队**开源的强大 **Qwen2.5-VL** 模型。
- 感谢 **DeepSeek AI** 在 **MLA** 和 **MoE** 技术上的探索与开源。
- 本项目的模型设计思路受到了 **VLMo** 等优秀研究工作的启发。