# MiniLLM

一个轻量级的 LLM（大型语言模型）实现，基于 PyTorch 和 Transformers 构建。支持预训练、监督微调（SFT）、LoRA 微调、PPO、GRPO 以及推理。

> 感谢 [minimind](https://github.com/jingyaogong/minimind) 项目提供的参考实现

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

- **训练范式**
  - 预训练 (Pretraining)
  - 监督微调 (SFT)
  - LoRA 高效微调
  - PPO 强化学习微调
  - GRPO 强化学习微调

## 项目结构

```
mini-llm/
├── dataset/                 # 训练数据集
│   ├── pretrain_hq.jsonl   # 预训练数据
│   ├── sft_mini_512.jsonl  # SFT 数据
│   ├── lora_identiy.jsonl  # LoRA 数据
│   └── rlaif-mini.jsonl    # 强化学习数据
├── out/                     # 模型输出目录
├── trainer/                 # 训练脚本
│   ├── train_pretrain.py   # 预训练
│   ├── train_sft.py        # SFT 微调
│   ├── train_lora.py       # LoRA 微调
│   ├── train_ppo.py        # PPO 训练
│   ├── train_grpo.py       # GRPO 训练
│   └── eval.py             # 推理评估
├── model/                   # 模型定义
│   ├── config.py           # 配置参数
│   ├── architecture.py     # 模型架构
│   └── layer.py            # 层定义
├── tokenizer/              # 分词器
└── utils/                  # 工具函数
```

## 环境要求

- Python >= 3.11
- PyTorch >= 2.6.0
- Transformers >= 4.57.1

## 安装

```bash
# 克隆项目
git clone https://github.com/your-repo/mini-llm.git
cd mini-llm

# 安装依赖
uv pip install -r requirements.txt

# 或使用 pip
pip install -r requirements.txt
```

## 数据准备

项目使用 JSONL 格式的数据集，每行一个 JSON 对象：

**预训练数据格式** (`pretrain_hq.jsonl`):
```json
{"text": "文本内容..."}
```

**SFT 数据格式** (`sft_mini_512.jsonl`):
```json
{"messages": [{"role": "user", "content": "用户问题"}, {"role": "assistant", "content": "助手回答"}]}
```

下载公开数据集并转换为 JSONL 格式放到 `dataset/` 目录下。

## 快速开始

### 预训练

```bash
cd trainer

# 标准 Dense 模型预训练
uv run python train_pretrain.py

# MoE 模型预训练
uv run python train_pretrain.py --use_moe 1
```

### 监督微调 (SFT)

```bash
cd trainer
uv run python train_sft.py
```

### LoRA 微调

```bash
cd trainer
uv run python train_lora.py
```

### 强化学习微调

```bash
cd trainer
uv run python train_ppo.py
uv run python train_grpo.py
```

### 推理/对话

```bash
cd trainer
python eval.py
```

## 常用参数

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 1 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--learning_rate` | 5e-4 | 学习率 |
| `--hidden_size` | 512 | 隐藏层维度 |
| `--num_hidden_layers` | 8 | 层数 |
| `--num_attention_heads` | 8 | 注意力头数 |
| `--intermediate_size` | 1408 | FFN 中间维度 |
| `--max_seq_len` | 512 | 最大序列长度 |
| `--use_moe` | 0 | 是否使用 MoE (0/1) |
| `--data_path` | ../dataset/xxx.jsonl | 训练数据路径 |
| `--save_dir` | ../out | 模型保存目录 |
| `--use_wandb` | False | 启用 wandb 实验跟踪 |

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hidden_size` | 512 | 模型隐藏层维度 |
| `--num_hidden_layers` | 8 | 模型层数 |
| `--use_moe` | 0 | 是否使用 MoE |
| `--weight` | full_sft | 权重类型 (pretrain/full_sft/rlhf/reason) |
| `--max_new_tokens` | 8192 | 最大生成长度 |
| `--temperature` | 0.85 | 生成温度 |
| `--top_p` | 0.85 | nucleus 采样阈值 |
| `--device` | cuda | 运行设备 (cuda/cpu) |

## 模型架构

### Dense 模型

- 隐藏层维度: 512
- 层数: 8
- 注意力头数: 8
- FFN 中间维度: 1408
- 词汇表大小: 200064

### MoE 模型

- 专家数量: 8
- 激活专家数: 2
- 其余参数与 Dense 模型相同

### 核心技术

1. **RMSNorm**: 比 LayerNorm 更高效的去中心化归一化
2. **RoPE**: 旋转位置编码，支持长上下文扩展
3. **Flash Attention**: 内存高效的注意力计算
4. **KV Cache**: 推理时缓存 Key-Value，减少重复计算

## 训练技巧

### 学习率调度

使用余弦退火调度器：
- 预热步数: 100
- 最小学习率: 1e-5

### 训练稳定性

- 梯度裁剪: max_norm=1.0
- 权重衰减: 0.1
- 混合精度训练 (FP16/BF16)


## 后续计划

- [ ] 完善 GRPO 训练代码
- [ ] 添加 DeepSpeed 支持
- [ ] 支持更多的 MoE 变体
- [ ] 完善推理优化

## License

MIT