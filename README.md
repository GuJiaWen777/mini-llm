# MiniLLM

一个轻量级的 LLM（大型语言模型）实现，基于 PyTorch 和 Transformers 构建。支持预训练、微调和推理。

## 功能特性

- **完整的 Transformer 架构**
  - RMSNorm 归一化
  - RoPE 旋转位置编码（支持 YaRN 长文本扩展）
  - Flash Attention 加速
  - KV Cache 推理优化

- **模型变体**
  - 标准 Dense 模型
  - MoE（混合专家）架构
  - LoRA 微调支持

- **训练特性**
  - 分布式训练（DistributedDataParallel）
  - 混合精度训练（bfloat16/float16）
  - 断点续训
  - 梯度累积与裁剪
  - 实验跟踪（WandB/SwanLab）

- **推理特性**
  - 流式输出
  - 多种采样策略（temperature、top_p）
  - 对话模板支持

## 环境要求

- Python >= 3.11
- PyTorch >= 2.6.0
- Transformers >= 4.57.1

## 安装

```bash
# 安装依赖
uv pip install -r requirements.txt
```

## 快速开始

### 预训练

```bash
cd trainer
uv run python trainer_pretrain.py
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 1 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--learning_rate` | 5e-4 | 学习率 |
| `--hidden_size` | 512 | 隐藏层维度 |
| `--num_hidden_layers` | 8 | 层数 |
| `--max_seq_len` | 512 | 最大序列长度 |
| `--use_moe` | 0 | 是否使用 MoE (0/1) |
| `--data_path` | ../dataset/pretrain_hq.jsonl | 训练数据路径 |
| `--save_dir` | ../out | 模型保存目录 |
| `--use_wandb` | False | 启用实验跟踪 |

### 推理/对话

```bash
python main.py
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hidden_size` | 512 | 模型隐藏层维度 |
| `--num_hidden_layers` | 8 | 模型层数 |
| `--use_moe` | 0 | 是否使用 MoE |
| `--weight` | full_sft | 权重类型 (pretrain/full_sft/rlhf/reason) |
| `--max_new_tokens` | 8192 | 最大生成长度 |
| `--temperature` | 0.85 | 生成温度 |
| `--top_p` | 0.85 | nucleus 采样阈值 |
| `--device` | cuda | 运行设备 |

## 模型规格

| 模型 | 参数量 | Hidden Size | Layers | Heads | KV Heads |
|------|--------|-------------|--------|-------|-----------|
| Small | ~26M | 512 | 8 | 8 | 2 |
| Base | ~104M | 768 | 16 | 12 | 4 |
| MoE | ~145M | 640 | 8 | 8 | 2 |

## 项目结构

```
mini-llm/
├── model/
│   └── MiniLLM.py       # 模型定义 (配置、Attention、MLP、Layer、LM Head)
├── trainer/
│   ├── trainer_pretrain.py  # 预训练脚本
│   └── trainer_utils.py     # 训练工具函数
├── dataset/
│   └── lm_dataset.py     # 数据集加载
├── checkpoints/         # 训练检查点
├── out/                 # 训练产出模型权重
├── main.py              # 推理入口
├── eval.py              # 评估脚本
└── README.md
```

## 使用示例

### 训练自定义数据

```bash
cd trainer
python trainer_pretrain.py \
    --data_path ../dataset/your_data.jsonl \
    --epochs 3 \
    --batch_size 32 \
    --hidden_size 768 \
    --num_hidden_layers 16 \
    --save_weight your_model
```

### 加载自定义权重推理

```bash
python main.py \
    --load_from model \
    --save_dir out \
    --weight your_model \
    --hidden_size 768
```

## License

MIT