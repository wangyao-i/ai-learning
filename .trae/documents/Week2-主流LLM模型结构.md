# Week 2: 主流LLM模型结构

## 本周目标

- 理解GPT系列结构演进
- 掌握LLaMA系列结构改进
- 理解Attention变体(MQA/GQA/FlashAttention)
- 了解MoE架构和LongContext技术

***

## Day 8: GPT系列结构演进

### 学习任务

| 时间      | 任务            | 目标               |
| ------- | ------------- | ---------------- |
| 工作中     | 回顾GPT系列结构     | 了解演进历程           |
| 晚上 1.5h | GPT-1/2/3/4演进 | 理解Decoder-only优势 |

### 学习材料

#### 必读论文

- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - GPT-1
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3

#### 推荐阅读

- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
- [The Annotated GPT-2](https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html)

### 核心知识点

#### GPT系列演进对比

| 特性        | GPT-1  | GPT-2     | GPT-3    | GPT-4        |
| --------- | ------ | --------- | -------- | ------------ |
| **参数量**   | 117M   | 1.5B      | 175B     | \~1.8T (推测)  |
| **层数**    | 12     | 48        | 96       | \~120 (推测)   |
| **隐藏维度**  | 768    | 1600      | 12288    | \~15000 (推测) |
| **注意力头数** | 12     | 25        | 96       | \~120 (推测)   |
| **训练数据**  | \~5GB  | 40GB      | 570GB    | \~10TB (推测)  |
| **核心创新**  | 预训练+微调 | Zero-shot | Few-shot | 多模态+RLHF     |

#### GPT架构特点

**Decoder-only架构**：

```
输入 → Token Embedding + Position Embedding
              ↓
        ┌─────────────────┐
        │  Decoder Block  │ × N
        │                 │
        │  Layer Norm     │
        │       ↓         │
        │  Masked Self-   │
        │  Attention      │
        │       ↓         │
        │  Add & Norm     │
        │       ↓         │
        │  FFN            │
        │       ↓         │
        │  Add & Norm     │
        └─────────────────┘
              ↓
        Layer Norm → Linear → Softmax
```

#### 为什么Decoder-only成为主流？

| 优势                | 说明               |
| ----------------- | ---------------- |
| **自回归生成**         | 天然适合文本生成任务       |
| **参数效率**          | 不需要Encoder，参数更集中 |
| **训练简单**          | 单向注意力，训练目标明确     |
| **扩展性好**          | 规模扩展时效果持续提升      |
| **Zero/Few-shot** | 大规模预训练后具备泛化能力    |

#### GPT vs BERT架构对比

| 特性    | GPT (Decoder-only) | BERT (Encoder-only) |
| ----- | ------------------ | ------------------- |
| 注意力方向 | 单向 (从左到右)          | 双向                  |
| 预训练任务 | 语言模型 (预测下一个词)      | Masked LM + NSP     |
| 适用任务  | 文本生成               | 文本理解                |
| 典型应用  | ChatGPT, 文本生成      | 分类, NER, 问答         |

### 自测题

#### 问题

1. GPT系列从GPT-1到GPT-4的主要演进点是什么？
2. 为什么Decoder-only架构在LLM中成为主流？
3. GPT和BERT的架构有什么区别？各自适用于什么场景？
4. GPT-3的Few-shot能力是如何实现的？
5. 请画出GPT的Decoder Block结构图。

#### 答案

**1. GPT系列从GPT-1到GPT-4的主要演进点是什么？**

| 版本        | 主要演进                                          |
| --------- | --------------------------------------------- |
| **GPT-1** | 首次提出预训练+微调范式，证明无监督预训练的有效性                     |
| **GPT-2** | 扩大规模(1.5B)，提出Zero-shot概念，无需微调直接完成任务           |
| **GPT-3** | 大幅扩展规模(175B)，Few-shot能力涌现，In-context Learning |
| **GPT-4** | 多模态能力，RLHF对齐，更强的推理和代码能力                       |

核心趋势：

- **规模扩展**：参数量从117M到\~1.8T
- **数据扩展**：训练数据从GB级到TB级
- **能力涌现**：Zero-shot → Few-shot → 多模态

**2. 为什么Decoder-only架构在LLM中成为主流？**

原因：

- **生成任务适配**：自回归生成是LLM的核心任务
- **训练效率**：单向注意力计算更简单，训练更快
- **参数效率**：所有参数都用于生成，没有Encoder的冗余
- **扩展性**：Scaling Laws表明Decoder-only扩展效果最好
- **涌现能力**：大规模Decoder-only模型涌现出Few-shot能力

**3. GPT和BERT的架构有什么区别？各自适用于什么场景？**

架构区别：

```
GPT (Decoder-only):
- 单向注意力 (Masked Self-Attention)
- 从左到右，只能看到之前的内容
- 预训练：预测下一个token

BERT (Encoder-only):
- 双向注意力 (Self-Attention)
- 可以看到整个序列
- 预训练：Masked LM + Next Sentence Prediction
```

适用场景：

| GPT  | BERT      |
| ---- | --------- |
| 文本生成 | 文本分类      |
| 对话系统 | 命名实体识别    |
| 代码生成 | 情感分析      |
| 翻译   | 问答系统(抽取式) |

**4. GPT-3的Few-shot能力是如何实现的？**

实现机制：

- **In-context Learning**：通过上下文学习，无需梯度更新
- **模式识别**：模型学会了从示例中识别任务模式
- **规模效应**：175B参数量下涌现出这种能力

工作原理：

```
输入:
"Translate to French:
Hello → Bonjour
Goodbye → Au revoir
Thank you →"

输出:
"Merci"
```

模型通过少量示例理解任务，并生成正确答案。

**5. 请画出GPT的Decoder Block结构图。**

```
输入 x (seq_len, d_model)
        ↓
┌─────────────────────────────┐
│      Layer Norm 1           │
│           ↓                 │
│   Masked Multi-Head         │
│   Self-Attention            │
│           ↓                 │
│      Add (残差连接)          │
│           ↓                 │
│      Layer Norm 2           │
│           ↓                 │
│   Feed-Forward Network      │
│           ↓                 │
│      Add (残差连接)          │
└─────────────────────────────┘
        ↓
输出 (seq_len, d_model)
```

注意：GPT使用Pre-LN (LayerNorm在子层之前)

***

## Day 9: LLaMA系列结构

### 学习任务

| 时间      | 任务            | 目标               |
| ------- | ------------- | ---------------- |
| 工作中     | 分析LLaMA系列     | 了解结构改进           |
| 晚上 1.5h | LLaMA/2/3结构改进 | 理解RMSNorm/SwiGLU |

### 学习材料

#### 必读论文

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

#### 推荐阅读

- [LLaMA源码解读](https://github.com/facebookresearch/llama)

### 核心知识点

#### LLaMA系列对比

| 特性         | LLaMA          | LLaMA 2        | LLaMA 3          |
| ---------- | -------------- | -------------- | ---------------- |
| **参数规模**   | 7B/13B/33B/65B | 7B/13B/34B/70B | 8B/70B/405B      |
| **上下文长度**  | 2048           | 4096           | 8192/128K        |
| **训练数据**   | 1T tokens      | 2T tokens      | 15T tokens       |
| **GQA**    | 无              | 34B/70B使用      | 全部使用             |
| **Chat版本** | 无              | LLaMA 2 Chat   | LLaMA 3 Instruct |

#### LLaMA架构改进

**相比原始Transformer的改进**：

| 组件    | 原始Transformer | LLaMA       |
| ----- | ------------- | ----------- |
| 归一化   | LayerNorm     | **RMSNorm** |
| 激活函数  | ReLU          | **SwiGLU**  |
| 位置编码  | 绝对位置编码        | **RoPE**    |
| 归一化位置 | Post-LN       | **Pre-LN**  |

#### LLaMA Block结构

```
输入 x
    ↓
┌─────────────────────────┐
│     RMSNorm             │
│         ↓               │
│   Grouped Query         │
│   Attention (GQA)       │
│         ↓               │
│   RoPE位置编码          │
│         ↓               │
│   Add (残差连接)         │
│         ↓               │
│     RMSNorm             │
│         ↓               │
│   SwiGLU FFN            │
│         ↓               │
│   Add (残差连接)         │
└─────────────────────────┘
    ↓
输出
```

#### SwiGLU FFN详解

```
SwiGLU(x) = (Swish(xW_gate) ⊗ xW_up) W_down

其中:
- W_gate, W_up: (d_model, d_ff)
- W_down: (d_ff, d_model)
- Swish(x) = x * sigmoid(x)
- ⊗: 逐元素乘法
```

#### LLaMA 2 vs LLaMA 1 改进

| 改进点        | 说明                               |
| ---------- | -------------------------------- |
| **训练数据**   | 2T tokens (LLaMA 1: 1T)          |
| **上下文长度**  | 4096 (LLaMA 1: 2048)             |
| **GQA**    | 34B/70B使用Grouped Query Attention |
| **Chat版本** | RLHF训练的对话版本                      |
| **安全训练**   | 加入安全相关的训练数据                      |

### 自测题

#### 问题

1. LLaMA相比原始Transformer做了哪些改进？为什么做这些改进？
2. RMSNorm相比LayerNorm有什么优势？
3. SwiGLU激活函数的计算过程是什么？为什么比ReLU更好？
4. LLaMA 2相比LLaMA 1有哪些改进？
5. 请画出LLaMA的Decoder Block结构图，标注关键组件。

#### 答案

**1. LLaMA相比原始Transformer做了哪些改进？为什么做这些改进？**

| 改进                     | 原因              |
| ---------------------- | --------------- |
| **RMSNorm替代LayerNorm** | 计算更简单，效果相当，训练更快 |
| **SwiGLU替代ReLU**       | 门控机制，性能更好       |
| **RoPE替代绝对位置编码**       | 相对位置感知，长度外推能力   |
| **Pre-LN替代Post-LN**    | 训练更稳定，不需要warmup |

这些改进都是经过大量实验验证的，在大规模模型中效果更好。

**2. RMSNorm相比LayerNorm有什么优势？**

```
LayerNorm: y = γ * (x - μ) / σ + β
RMSNorm:   y = γ * x / RMS(x)

其中 RMS(x) = sqrt(mean(x²) + ε)
```

优势：

- **计算更简单**：不需要计算均值，只计算均方根
- **无偏移参数**：去除β参数，模型更简洁
- **效果相当**：实验表明效果与LayerNorm相当甚至更好
- **训练更快**：计算量减少约10-15%

**3. SwiGLU激活函数的计算过程是什么？为什么比ReLU更好？**

计算过程：

```python
def SwiGLU(x, W_gate, W_up, W_down):
    gate = x @ W_gate           # (d_model, d_ff)
    up = x @ W_up               # (d_model, d_ff)
    
    swish_gate = swish(gate)    # swish(x) = x * sigmoid(x)
    hidden = swish_gate * up    # 逐元素乘法
    
    output = hidden @ W_down    # (d_ff, d_model)
    return output
```

比ReLU更好的原因：

- **门控机制**：通过逐元素乘法实现信息筛选
- **平滑激活**：Swish比ReLU更平滑，梯度更稳定
- **非单调性**：对负值不完全截断，保留部分信息
- **实验验证**：在LLM中性能优于ReLU/GELU

**4. LLaMA 2相比LLaMA 1有哪些改进？**

| 改进点    | LLaMA 1   | LLaMA 2      |
| ------ | --------- | ------------ |
| 训练数据   | 1T tokens | 2T tokens    |
| 上下文长度  | 2048      | 4096         |
| GQA    | 无         | 34B/70B使用    |
| Chat版本 | 无         | LLaMA 2 Chat |
| 安全训练   | 无         | 有            |

**5. 请画出LLaMA的Decoder Block结构图，标注关键组件。**

```
输入 x (seq_len, d_model)
        ↓
┌─────────────────────────────────┐
│         RMSNorm                 │
│           ↓                     │
│   Grouped Query Attention (GQA) │
│   - Q: n_heads                  │
│   - K,V: n_kv_heads (较少)      │
│           ↓                     │
│   RoPE位置编码                   │
│           ↓                     │
│   Add (残差连接)                 │
│           ↓                     │
│         RMSNorm                 │
│           ↓                     │
│   SwiGLU FFN                    │
│   - gate = Swish(x @ W_gate)    │
│   - up = x @ W_up               │
│   - out = (gate * up) @ W_down  │
│           ↓                     │
│   Add (残差连接)                 │
└─────────────────────────────────┘
        ↓
输出 (seq_len, d_model)
```

***

## Day 10: Attention实现变体

### 学习任务

| 时间      | 任务                       | 目标           |
| ------- | ------------------------ | ------------ |
| 工作中     | 查看Attention实现变体          | 了解优化方法       |
| 晚上 1.5h | 整理FlashAttention/MQA/GQA | 理解KV Cache复用 |

### 学习材料

#### 必读论文

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) - MQA

#### 推荐阅读

- [FlashAttention详解](https://zhuanlan.zhihu.com/p/617521833)

### 核心知识点

#### Multi-Head Attention (MHA)

```
标准MHA:
- 每个head有独立的Q, K, V
- n_heads个Q, n_heads个K, n_heads个V
- KV Cache: n_heads * seq_len * d_head
```

#### Multi-Query Attention (MQA)

```
MQA:
- 每个head有独立的Q
- 所有head共享同一个K, V
- n_heads个Q, 1个K, 1个V
- KV Cache: 1 * seq_len * d_head (减少n_heads倍)
```

**优势**：

- KV Cache大幅减少
- 推理速度提升
- 质量略有下降

#### Grouped Query Attention (GQA)

```
GQA:
- n_heads个Q
- n_kv_groups个K, V (n_kv_groups < n_heads)
- 例如: 32个Q, 8个K, 8个V
- KV Cache: n_kv_groups * seq_len * d_head
```

**优势**：

- 平衡MHA的质量和MQA的效率
- LLaMA 2/3使用GQA

#### 三种Attention对比

| 特性       | MHA      | MQA      | GQA           |
| -------- | -------- | -------- | ------------- |
| Q数量      | n\_heads | n\_heads | n\_heads      |
| K,V数量    | n\_heads | 1        | n\_kv\_groups |
| KV Cache | 100%     | \~3%     | \~25%         |
| 质量       | 最好       | 略差       | 接近MHA         |
| 速度       | 最慢       | 最快       | 中等            |

#### FlashAttention

**核心思想**：通过优化内存访问模式加速Attention计算

**传统Attention的问题**：

```
1. Q @ K^T: 写入HBM (n²大小的矩阵)
2. Softmax: 从HBM读取，计算，写入HBM
3. @ V: 从HBM读取，计算
内存访问量大，成为瓶颈
```

**FlashAttention优化**：

```
1. 分块计算: 将Q, K, V分成小块
2. 在SRAM中完成Softmax
3. 避免存储n²的中间矩阵
4. 内存访问量从O(n²)降到O(n)
```

**效果**：

- 内存使用: O(n²) → O(n)
- 速度提升: 2-4倍
- 数值精度: 保持不变

### 自测题

#### 问题

1. MQA和GQA的区别是什么？为什么GQA更常用？
2. FlashAttention是如何优化Attention计算的？
3. 为什么MQA/GQA能减少KV Cache？这对推理有什么影响？
4. LLaMA 2/3为什么选择GQA而不是MQA？
5. 请比较MHA、MQA、GQA的KV Cache大小和计算复杂度。

#### 答案

**1. MQA和GQA的区别是什么？为什么GQA更常用？**

区别：

```
MQA: n_heads个Q, 1组K/V
GQA: n_heads个Q, n_kv_groups组K/V (1 < n_kv_groups < n_heads)
```

GQA更常用的原因：

- **质量更好**：GQA的质量接近MHA，优于MQA
- **灵活权衡**：可以通过调整n\_kv\_groups平衡质量和效率
- **实验验证**：LLaMA 2/3使用GQA，效果很好

**2. FlashAttention是如何优化Attention计算的？**

优化策略：

- **分块计算**：将Q, K, V分成小块，每块在SRAM中计算
- **避免中间存储**：不存储n²的注意力矩阵，逐块计算Softmax
- **内存访问优化**：减少HBM读写次数

具体流程：

```
传统方法:
Q, K, V在HBM → Q@K^T写入HBM → Softmax写入HBM → @V写入HBM
内存访问: O(n²)

FlashAttention:
Q, K, V在HBM → 分块加载到SRAM → 块内计算 → 结果写回HBM
内存访问: O(n)
```

**3. 为什么MQA/GQA能减少KV Cache？这对推理有什么影响？**

减少KV Cache的原因：

```
MHA: KV Cache = 2 * n_heads * seq_len * d_head
MQA: KV Cache = 2 * 1 * seq_len * d_head
GQA: KV Cache = 2 * n_kv_groups * seq_len * d_head
```

对推理的影响：

- **内存占用减少**：可以支持更长的序列或更大的batch
- **带宽压力降低**：KV Cache读取更快
- **推理速度提升**：尤其是长序列场景
- **多用户并发**：相同显存可以服务更多请求

**4. LLaMA 2/3为什么选择GQA而不是MQA？**

原因：

- **质量考虑**：GQA的质量比MQA更接近MHA
- **实验验证**：LLaMA团队实验表明GQA效果更好
- **平衡设计**：GQA是MHA和MQA之间的折中
- **灵活性**：可以根据模型大小调整n\_kv\_groups

LLaMA 2配置：

- 7B/13B: 不使用GQA (MHA)
- 34B/70B: 使用GQA (n\_kv\_groups = 8)

**5. 请比较MHA、MQA、GQA的KV Cache大小和计算复杂度。**

| 指标             | MHA      | MQA    | GQA      |
| -------------- | -------- | ------ | -------- |
| **KV Cache大小** | 2*n*h\*d | 2*1*d  | 2*g*h\*d |
| **相对大小**       | 100%     | \~3%   | \~25%    |
| **计算复杂度**      | O(n²d)   | O(n²d) | O(n²d)   |
| **质量**         | 最好       | 略差     | 接近MHA    |

注：n=序列长度, h=head数, d=head维度, g=kv\_groups数

***

## Day 11: 推理中的模型结构

### 学习任务

| 时间      | 任务         | 目标          |
| ------- | ---------- | ----------- |
| 工作中     | 分析推理中的模型结构 | 了解推理特点      |
| 晚上 1.5h | 整理各模型推理特点  | 理解模型结构对推理影响 |

### 学习材料

#### 推荐阅读

- [vLLM源码](https://github.com/vllm-project/vllm)
- [Efficient Inference of Large Language Models](https://arxiv.org/abs/2304.00595)

### 核心知识点

#### 推理 vs 训练的模型结构差异

| 方面            | 训练      | 推理          |
| ------------- | ------- | ----------- |
| **Attention** | 全序列并行计算 | 自回归逐token生成 |
| **KV Cache**  | 不需要     | 需要缓存        |
| **Batch**     | 固定大小    | 动态变化        |
| **内存使用**      | 激活值占用大  | KV Cache占用大 |
| **计算特点**      | 计算密集    | 内存带宽密集      |

#### KV Cache对模型结构的要求

```
KV Cache大小 = 2 * n_layers * n_kv_heads * seq_len * d_head * dtype_size

例如 LLaMA 70B:
- n_layers = 80
- n_kv_heads = 8 (GQA)
- d_head = 128
- seq_len = 4096
- dtype_size = 2 (FP16)

KV Cache = 2 * 80 * 8 * 4096 * 128 * 2 = ~1.3GB per request
```

#### 模型结构对推理的影响

| 结构特点        | 对推理的影响          |
| ----------- | --------------- |
| **模型大小**    | 内存占用、加载时间       |
| **层数**      | 计算延迟、KV Cache大小 |
| **隐藏维度**    | 计算量、内存带宽        |
| **注意力头数**   | 并行度、KV Cache    |
| **GQA/MQA** | KV Cache大小、推理速度 |
| **上下文长度**   | KV Cache大小      |

#### 推理优化与模型结构

| 优化技术                     | 依赖的模型结构         |
| ------------------------ | --------------- |
| **PagedAttention**       | KV Cache管理      |
| **量化**                   | 权重和激活值          |
| **算子融合**                 | Attention + FFN |
| **Speculative Decoding** | 小模型辅助           |
| **KV Cache压缩**           | 注意力稀疏性          |

### 自测题

#### 问题

1. 推理和训练时模型结构的计算有什么不同？
2. KV Cache的大小由哪些因素决定？如何计算？
3. GQA如何影响推理性能？
4. 为什么推理是内存带宽密集型而不是计算密集型？
5. 模型结构如何影响推理优化策略？

#### 答案

**1. 推理和训练时模型结构的计算有什么不同？**

| 方面            | 训练       | 推理                      |
| ------------- | -------- | ----------------------- |
| **计算模式**      | 全序列并行    | 自回归串行                   |
| **Attention** | 一次计算所有位置 | 每步只计算新token             |
| **KV Cache**  | 不需要      | 必须缓存                    |
| **Batch**     | 固定       | 动态(Continuous Batching) |
| **内存使用**      | 激活值为主    | KV Cache为主              |

**2. KV Cache的大小由哪些因素决定？如何计算？**

决定因素：

- **层数 (n\_layers)**: 每层都需要KV Cache
- **KV头数 (n\_kv\_heads)**: GQA减少KV头数
- **序列长度 (seq\_len)**: 线性增长
- **头维度 (d\_head)**: 每个token的特征维度
- **数据类型 (dtype\_size)**: FP16=2, FP32=4

计算公式：

```
KV Cache = 2 * n_layers * n_kv_heads * seq_len * d_head * dtype_size
```

**3. GQA如何影响推理性能？**

影响：

- **KV Cache减少**: n\_kv\_heads < n\_heads，内存占用降低
- **带宽压力降低**: KV Cache读取更快
- **长序列支持**: 相同显存支持更长序列
- **多用户并发**: 相同显存服务更多请求
- **质量保持**: 相比MQA质量更好

**4. 为什么推理是内存带宽密集型而不是计算密集型？**

原因：

- **自回归生成**: 每次只生成一个token，计算量小
- **KV Cache读取**: 每次生成都要读取全部KV Cache
- **计算/访存比低**:
  ```
  计算: O(d²) (矩阵乘法)
  访存: O(n * d) (KV Cache读取)

  当n很大时，访存成为瓶颈
  ```
- **GPU利用率低**: 大量时间在等待内存传输

**5. 模型结构如何影响推理优化策略？**

| 模型结构        | 推理优化策略                    |
| ----------- | ------------------------- |
| **大模型**     | 量化、分布式推理                  |
| **GQA/MQA** | 减少KV Cache                |
| **长上下文**    | PagedAttention、KV Cache压缩 |
| **深层网络**    | 算子融合、流水并行                 |
| **大隐藏维度**   | 张量并行                      |

***

## Day 12: MoE和LongContext

### 学习任务

| 时间      | 任务           | 目标                |
| ------- | ------------ | ----------------- |
| 工作中     | 对比开源vs闭源模型结构 | 了解发展趋势            |
| 晚上 1.5h | 模型结构发展趋势     | 理解MoE/LongContext |

### 学习材料

#### 必读论文

- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) - MoE
- [LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)

#### 推荐阅读

- [Mixture of Experts详解](https://zhuanlan.zhihu.com/p/672025578)

### 核心知识点

#### MoE (Mixture of Experts)

**核心思想**：将FFN层替换为多个专家网络，每次只激活部分专家

```
传统FFN:
FFN(x) = ReLU(xW_1)W_2

MoE FFN:
MoE(x) = Σ_i G(x)_i * Expert_i(x)

其中:
- G(x): 门控网络，输出每个专家的权重
- Expert_i(x): 第i个专家网络
- 只激活top-k个专家 (通常k=2)
```

**优势**：

- 参数量大，但计算量小
- 专家专业化，性能更好
- 推理效率高

**代表模型**：

- Mixtral 8x7B
- GPT-4 (推测)
- DeepSeek-MoE

#### MoE架构图

```
输入 x
    ↓
┌─────────────────────────────┐
│        Gate Network         │
│   G(x) = Softmax(xW_g)      │
│   选择 top-k 个专家          │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│      Expert Network         │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐   │
│  │E_1│ │E_2│ │E_3│ │E_4│   │
│  └───┘ └───┘ └───┘ └───┘   │
│        ... (n个专家)         │
└─────────────────────────────┘
    ↓
加权求和 → 输出
```

#### LongContext技术

**挑战**：

- KV Cache随序列长度线性增长
- Attention复杂度O(n²)
- 位置编码外推能力

**解决方案**：

| 技术                 | 方法                       | 代表模型  |
| ------------------ | ------------------------ | ----- |
| **ALiBi**          | 线性偏置注意力                  | BLOOM |
| **RoPE扩展**         | 位置编码插值                   | LLaMA |
| **LongLoRA**       | Shifted Sparse Attention | -     |
| **Ring Attention** | 分布式计算长序列                 | -     |
| **KV Cache压缩**     | 稀疏注意力                    | H2O   |

#### LongContext推理优化

```
传统方法:
KV Cache = n_layers * n_heads * seq_len * d_head
seq_len=100K时，KV Cache可达数十GB

优化方法:
1. PagedAttention: 分页管理KV Cache
2. KV Cache压缩: 保留重要token
3. 滑动窗口: 只保留最近K个token
4. 分块计算: 减少内存占用
```

### 自测题

#### 问题

1. MoE的核心思想是什么？相比传统FFN有什么优势？
2. MoE推理时如何选择专家？为什么只激活部分专家？
3. LongContext面临的主要挑战是什么？
4. RoPE如何支持更长的上下文？
5. 请比较Dense模型和MoE模型的参数量和计算量。

#### 答案

**1. MoE的核心思想是什么？相比传统FFN有什么优势？**

核心思想：

- 将单个FFN替换为多个专家网络
- 通过门控网络选择激活哪些专家
- 每次只激活部分专家(top-k)

优势：

- **参数效率**: 参数量大但计算量小
- **专家专业化**: 不同专家处理不同类型的输入
- **推理效率**: 只计算激活的专家
- **性能提升**: 更好的泛化能力

**2. MoE推理时如何选择专家？为什么只激活部分专家？**

专家选择：

```
1. Gate Network计算每个专家的权重:
   G(x) = Softmax(x @ W_gate)

2. 选择top-k个权重最高的专家:
   experts = top_k(G(x), k)

3. 只计算选中的专家:
   output = Σ G(x)_i * Expert_i(x)
```

只激活部分专家的原因：

- **计算效率**: 减少计算量
- **专家专业化**: 每个专家处理特定类型输入
- **负载均衡**: 避免所有请求都选择相同专家

**3. LongContext面临的主要挑战是什么？**

主要挑战：

- **KV Cache内存**: 随序列长度线性增长
  ```
  KV Cache = 2 * n_layers * n_heads * seq_len * d_head
  seq_len=100K时，可达数十GB
  ```
- **Attention复杂度**: O(n²)计算和内存
- **位置编码外推**: 超出训练长度效果下降
- **内存带宽**: 大量KV Cache读取

**4. RoPE如何支持更长的上下文？**

方法：

- **位置插值**: 将长序列的位置映射到训练范围内
  ```
  原始位置: 0, 1, 2, ..., L (训练长度)
  扩展位置: 0, 1, 2, ..., L' (L' > L)
  插值后: 0, L/L', 2L/L', ..., L
  ```
- **NTK-aware插值**: 调整RoPE的频率
- **YaRN**: 结合多种插值方法

**5. 请比较Dense模型和MoE模型的参数量和计算量。**

以Mixtral 8x7B为例：

| 指标        | Dense 47B | Mixtral 8x7B   |
| --------- | --------- | -------------- |
| **总参数量**  | 47B       | 47B            |
| **激活参数量** | 47B       | \~13B (每token) |
| **专家数量**  | 1         | 8              |
| **每专家参数** | -         | \~7B           |
| **激活专家数** | 1         | 2              |
| **推理计算量** | 100%      | \~28%          |

MoE优势：相同参数量下，计算量大幅减少

***

## Day 13: 实战 - 对比两种模型的推理差异

### 学习任务

| 时间    | 任务          | 目标     |
| ----- | ----------- | ------ |
| 晚上 2h | 对比两种模型的推理差异 | 输出对比分析 |

### 实战要求

#### 任务描述

选择两个不同架构的LLM，对比其推理特点：

**推荐对比组合**：

1. LLaMA 7B vs LLaMA 70B (GQA)
2. LLaMA vs Mistral (不同架构)
3. Dense vs MoE (Mixtral)

#### 分析维度

1. **模型结构对比**
   - 层数、隐藏维度、注意力头数
   - 是否使用GQA/MQA
   - FFN结构差异
2. **推理性能对比**
   - KV Cache大小
   - 推理延迟
   - 内存占用
3. **优化策略对比**
   - 适用的优化技术
   - 瓶颈分析

### 验证标准

- [ ] 完成两个模型的结构对比
- [ ] 计算KV Cache大小差异
- [ ] 分析推理性能差异
- [ ] 提出优化建议

***

## Day 14: 本周复盘 + 自测

### 学习任务

| 时间    | 任务        | 目标     |
| ----- | --------- | ------ |
| 晚上 1h | 本周复盘 + 自测 | 检验学习效果 |

### 本周知识点回顾

#### 知识点清单

- [ ] GPT系列结构演进
- [ ] Decoder-only架构优势
- [ ] LLaMA系列结构改进
- [ ] RMSNorm和SwiGLU
- [ ] MQA/GQA/FlashAttention
- [ ] MoE架构
- [ ] LongContext技术

### 综合自测题

#### 问题

1. 为什么Decoder-only架构在LLM中成为主流？相比Encoder-Decoder有什么优势？
2. LLaMA相比原始Transformer做了哪些改进？这些改进对推理有什么影响？
3. GQA如何平衡MHA的质量和MQA的效率？
4. MoE架构的核心优势是什么？推理时如何选择专家？
5. 长上下文推理面临哪些挑战？有哪些解决方案？

#### 答案

**1. 为什么Decoder-only架构在LLM中成为主流？相比Encoder-Decoder有什么优势？**

Decoder-only成为主流的原因：

- **生成任务适配**: 自回归生成是LLM的核心任务
- **训练效率**: 单向注意力计算更简单
- **参数效率**: 所有参数都用于生成
- **扩展性**: Scaling Laws表明扩展效果最好
- **涌现能力**: 大规模模型涌现Few-shot能力

相比Encoder-Decoder的优势：

- 结构更简单，训练更快
- 参数更集中，效率更高
- 生成任务效果更好

**2. LLaMA相比原始Transformer做了哪些改进？这些改进对推理有什么影响？**

| 改进              | 对推理的影响          |
| --------------- | --------------- |
| RMSNorm         | 计算更快，推理延迟降低     |
| SwiGLU          | 质量更好，参数量增加      |
| RoPE            | 支持更长上下文         |
| Pre-LN          | 训练稳定，推理一致       |
| GQA (LLaMA 2/3) | KV Cache减少，推理更快 |

**3. GQA如何平衡MHA的质量和MQA的效率？**

平衡方式：

- **KV头数**: n\_kv\_groups介于1和n\_heads之间
- **质量**: 接近MHA，优于MQA
- **效率**: KV Cache减少，推理加速
- **灵活配置**: 可根据需求调整n\_kv\_groups

**4. MoE架构的核心优势是什么？推理时如何选择专家？**

核心优势：

- **参数效率**: 参数量大但计算量小
- **专家专业化**: 不同专家处理不同输入
- **推理效率**: 只激活部分专家

专家选择：

```
1. Gate Network计算权重: G(x) = Softmax(x @ W_gate)
2. 选择top-k专家
3. 加权求和: output = Σ G(x)_i * Expert_i(x)
```

**5. 长上下文推理面临哪些挑战？有哪些解决方案？**

挑战：

- KV Cache内存线性增长
- Attention复杂度O(n²)
- 位置编码外推能力

解决方案：

- **PagedAttention**: 分页管理KV Cache
- **KV Cache压缩**: 保留重要token
- **RoPE扩展**: 位置编码插值
- **Ring Attention**: 分布式计算

***

## 本周学习总结

### 完成情况

- [ ] Day 8: GPT系列结构演进
- [ ] Day 9: LLaMA系列结构
- [ ] Day 10: Attention实现变体
- [ ] Day 11: 推理中的模型结构
- [ ] Day 12: MoE和LongContext
- [ ] Day 13: 实战 - 对比模型推理差异
- [ ] Day 14: 本周复盘 + 自测

### 下周预告

Week 3: LLM推理核心原理

- Prefill vs Decode
- PagedAttention原理
- KV Cache管理
- 推理调度策略

