# MiniMind 代码详解（一）：LLM 基础概念

> 本教程基于 MiniMind 项目代码，帮助初学者理解大语言模型（LLM）的核心概念。

## 1. 什么是大语言模型（LLM）？

大语言模型本质上是一个**"下一个词预测器"**。给它一段文字，它会预测接下来最可能出现的词是什么。

```
输入: "今天天气真"
模型预测: "好" (概率最高)
```

通过不断预测下一个词，模型就能生成一整段流畅的文字。这就是 ChatGPT 等聊天机器人的基本原理。

## 2. Token：模型看到的"单词"

### 2.1 什么是 Token？

模型不直接处理文字，而是把文字切分成更小的单元，叫做 **Token**。

```python
# MiniMind 使用的分词器（tokenizer）
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./model')

# 示例：把文字转成 Token
text = "你好，世界"
tokens = tokenizer(text)
print(tokens.input_ids)  # 输出类似：[1, 234, 567, 89, 2]
```

每个 Token 对应一个数字（ID），模型实际处理的就是这些数字序列。

### 2.2 MiniMind 的词表

在 `model/model_minimind.py` 中可以看到配置：

```python
class MiniMindConfig(PretrainedConfig):
    def __init__(self, vocab_size: int = 6400, ...):
        self.vocab_size = vocab_size  # 词表大小：6400个词
```

**词表大小 = 6400** 意味着模型"认识"6400个不同的 Token。相比之下：
- GPT-4 词表：约 100,000
- LLaMA 词表：约 32,000

MiniMind 故意用小词表来保持模型轻量。

## 3. Embedding：把数字变成向量

### 3.1 为什么需要 Embedding？

数字本身没有"含义"。我们需要把每个 Token ID 转换成一个**向量**（一串浮点数），这样模型才能学习词与词之间的关系。

```python
# model/model_minimind.py 第381行
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
# vocab_size=6400, hidden_size=512
# 创建一个 6400 x 512 的查找表
```

### 3.2 Embedding 的工作原理

```
Token ID: 234
           ↓
    查找 Embedding 表的第 234 行
           ↓
向量: [0.12, -0.34, 0.56, ..., 0.78]  (512维)
```

**hidden_size = 512** 就是每个词向量的维度。这个数字越大，模型能表达的信息越丰富，但计算量也越大。

## 4. 模型的整体结构

MiniMind 采用 **Decoder-only Transformer** 架构，这是 GPT 系列的标准架构：

```
输入文本: "今天天气"
    ↓
[Tokenizer] 分词
    ↓
Token IDs: [1, 234, 567, 89]
    ↓
[Embedding] 查表得到向量
    ↓
向量序列: [[...], [...], [...], [...]]  每个512维
    ↓
[Transformer Block] × 8层  ← 核心计算
    ↓
隐藏状态: [[...], [...], [...], [...]]
    ↓
[LM Head] 线性层，映射到词表大小
    ↓
预测分布: 6400个词的概率
    ↓
选择概率最高的词: "好"
```

在 `model/model_minimind.py` 中：

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        self.model = MiniMindModel(config)      # Transformer 主体
        self.lm_head = nn.Linear(hidden_size, vocab_size)  # 输出层
```

## 5. 损失函数：模型如何学习

### 5.1 交叉熵损失

模型通过比较"预测的词"和"实际的词"来学习：

```python
# model/model_minimind.py 第456-459行
if labels is not None:
    shift_logits = logits[..., :-1, :].contiguous()   # 预测值
    shift_labels = labels[..., 1:].contiguous()        # 真实值
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1), ignore_index=-100)
```

### 5.2 为什么要"shift"（错位）？

```
输入:    [今] [天] [天] [气]
标签:    [天] [天] [气] [好]
          ↑    ↑    ↑    ↑
        预测1 预测2 预测3 预测4
```

模型看到"今"要预测"天"，看到"今天"要预测"天"（第二个），依此类推。这就是**自回归**（Autoregressive）的含义。

### 5.3 ignore_index=-100 的含义

```python
labels[input_ids == tokenizer.pad_token_id] = -100
```

有些位置是填充（padding），不应该计算损失。设为 -100 告诉损失函数"忽略这个位置"。

## 6. LLM 训练的三个阶段

MiniMind 完整展示了 LLM 训练的三个阶段：

### 6.1 预训练（Pretrain）

```
目标：学习语言知识（语法、事实、常识）
数据：大量无标注文本
方法：预测下一个词
```

对应代码：`trainer/train_pretrain.py`

### 6.2 监督微调（SFT - Supervised Fine-Tuning）

```
目标：学习对话格式和指令遵循
数据：问答对（人类标注）
方法：在预训练模型基础上继续训练
```

对应代码：`trainer/train_full_sft.py`

### 6.3 强化学习（RLHF/RLAIF）

```
目标：让模型输出更符合人类偏好
数据：人类偏好数据（哪个回答更好）
方法：DPO、PPO、GRPO 等算法
```

对应代码：`trainer/train_dpo.py`、`trainer/train_grpo.py`

## 7. 关键参数解读

在 `model/model_minimind.py` 的配置中：

```python
class MiniMindConfig(PretrainedConfig):
    hidden_size: int = 512          # 隐藏层维度（词向量维度）
    num_attention_heads: int = 8    # 注意力头数
    num_hidden_layers: int = 8      # Transformer 层数
    num_key_value_heads: int = 2    # KV 头数（GQA技术）
    vocab_size: int = 6400          # 词表大小
    max_position_embeddings: int = 32768  # 最大序列长度
```

**模型大小估算**：
- Embedding 层：6400 × 512 = 3.3M 参数
- 8层 Transformer：约 20M 参数
- **总计：约 25M 参数**

这就是为什么 MiniMind 只有 25.8M 参数，却能实现基本的对话功能。

## 8. 总结

| 概念 | 解释 | MiniMind 中的体现 |
|------|------|------------------|
| Token | 文本的最小单位 | vocab_size=6400 |
| Embedding | Token 到向量的映射 | hidden_size=512 |
| Transformer | 核心计算模块 | num_hidden_layers=8 |
| 自回归 | 预测下一个词 | shift_logits/shift_labels |
| 损失函数 | 衡量预测与真实的差距 | cross_entropy |

下一篇我们将深入 Transformer 架构，理解注意力机制（Attention）的工作原理。

---

[下一篇：模型架构详解 →](./02_模型架构详解.md)
