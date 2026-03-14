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
uv run python train_pretrain.py 
uv run python train_pretrain.py --use_moe 1
uv run python train_sft.py
uv run python train_lora.py
uv run python train_ppo.py
uv run python train_grpo.py
```
数据集
- pretrain_hq.jsonl
- sft_mini_512.jsonl
- lora_identiy.jsonl
- rlaif-mini.jsonl
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
python eval.py
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

## License

MIT