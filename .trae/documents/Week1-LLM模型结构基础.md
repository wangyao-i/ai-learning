# Week 1: LLM模型结构基础

## 本周目标
- 理解Transformer整体架构
- 掌握Self-Attention计算过程
- 理解FFN/MLP结构
- 掌握归一化层(LayerNorm/BatchNorm)
- 理解位置编码(RoPE/ALiBi/绝对位置编码)

---

## Day 1: Transformer整体架构

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 回顾vLLM代码结构 | 熟悉代码布局 |
| 晚上 1.5h | Transformer整体架构 | 能画出结构图 |

### 学习材料

#### 必读论文
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文

#### 中文教程
- [华为开发者联盟 - Transformer架构详解](https://developer.huawei.com/consumer/cn/blog/topic/03204914408829077)
- [Transformer模型架构百科](https://m.baike.com/wiki/Transformer%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84)

#### 英文教程
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 图解Transformer

### 核心知识点

#### Transformer架构图
```
输入序列 → Embedding → Positional Encoding
                              ↓
                    ┌─────────────────────┐
                    │   Encoder Stack     │
                    │   (N层, 通常N=6)    │
                    │                     │
                    │   Multi-Head        │
                    │   Self-Attention    │
                    │         ↓           │
                    │   Add & Norm        │
                    │         ↓           │
                    │   Feed Forward      │
                    │         ↓           │
                    │   Add & Norm        │
                    └─────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │   Decoder Stack     │
                    │   (N层, 通常N=6)    │
                    │                     │
                    │   Masked Multi-Head │
                    │   Self-Attention    │
                    │         ↓           │
                    │   Add & Norm        │
                    │         ↓           │
                    │   Encoder-Decoder   │
                    │   Attention         │
                    │         ↓           │
                    │   Add & Norm        │
                    │         ↓           │
                    │   Feed Forward      │
                    │         ↓           │
                    │   Add & Norm        │
                    └─────────────────────┘
                              ↓
                    Linear → Softmax
                              ↓
                         输出概率
```

#### 核心组件说明
| 组件 | 功能 | 关键点 |
|------|------|--------|
| **Embedding Layer** | 词嵌入，将token转为向量 | 维度通常为512/768/1024 |
| **Positional Encoding** | 注入位置信息 | Transformer本身无位置感知 |
| **Multi-Head Attention** | 多头注意力机制 | 并行捕捉不同依赖关系 |
| **Feed Forward Network** | 前馈神经网络 | 两层全连接，中间有激活函数 |
| **Add & Norm** | 残差连接+层归一化 | 稳定训练，缓解梯度消失 |

#### 三种Transformer变体
| 架构类型 | 代表模型 | 适用场景 |
|----------|----------|----------|
| **Encoder-only** | BERT, RoBERTa | 文本理解、分类、NER |
| **Decoder-only** | GPT, LLaMA | 文本生成、对话 |
| **Encoder-Decoder** | T5, BART | 翻译、摘要 |

### 自测题

#### 问题
1. Transformer由哪些主要组件构成？请画出架构图。
2. Encoder和Decoder的主要区别是什么？
3. 为什么Transformer需要位置编码(Positional Encoding)？
4. 残差连接(Residual Connection)的作用是什么？
5. Decoder中的Mask机制有什么作用？

#### 答案

**1. Transformer由哪些主要组件构成？请画出架构图。**

Transformer主要由以下组件构成：
- **输入部分**：Embedding Layer + Positional Encoding
- **Encoder Stack**：N个相同的Encoder层堆叠
  - Multi-Head Self-Attention
  - Add & LayerNorm
  - Feed-Forward Network
  - Add & LayerNorm
- **Decoder Stack**：N个相同的Decoder层堆叠
  - Masked Multi-Head Self-Attention
  - Add & LayerNorm
  - Encoder-Decoder Attention
  - Add & LayerNorm
  - Feed-Forward Network
  - Add & LayerNorm
- **输出部分**：Linear + Softmax

**2. Encoder和Decoder的主要区别是什么？**

| 方面 | Encoder | Decoder |
|------|---------|---------|
| 注意力类型 | Self-Attention (双向) | Masked Self-Attention (单向) + Encoder-Decoder Attention |
| 信息可见性 | 可以看到整个输入序列 | 只能看到当前位置之前的信息 |
| 主要功能 | 理解输入序列 | 生成输出序列 |
| 层数结构 | 2个子层 | 3个子层 |

**3. 为什么Transformer需要位置编码(Positional Encoding)？**

因为Transformer完全基于注意力机制，没有循环结构(RNN)或卷积结构(CNN)，无法感知序列中token的位置信息。位置编码通过给每个位置添加一个独特的向量，让模型能够区分不同位置的相同token，从而理解序列的顺序关系。

例如："我吃饭" 和 "饭吃我" 中，"我"和"饭"的位置不同，语义完全不同，位置编码帮助模型区分这种情况。

**4. 残差连接(Residual Connection)的作用是什么？**

残差连接的主要作用：
- **缓解梯度消失**：梯度可以直接通过残差连接传递，避免深层网络中的梯度消失问题
- **保留原始信息**：将输入直接加到输出中，保留原始特征信息
- **加速训练**：使网络更容易学习恒等映射，加速收敛
- **公式**：`Output = LayerNorm(x + Sublayer(x))`

**5. Decoder中的Mask机制有什么作用？**

Mask机制的作用：
- **防止信息泄露**：在生成第t个token时，确保模型只能看到前t-1个token的信息
- **保证自回归生成**：符合语言模型的自回归生成逻辑，即根据已生成的内容预测下一个token
- **实现方式**：将未来位置的attention score设为负无穷，经过softmax后变为0

---

## Day 2: Self-Attention机制详解

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 查阅Attention机制详解 | 了解基本概念 |
| 晚上 1.5h | 整理Self-Attention计算过程 | 理解Q/K/V原理 |

### 学习材料

#### 必读论文
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Section 3.2 Attention

#### 推荐教程
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) - 带代码注释的Transformer
- [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

### 核心知识点

#### Self-Attention计算过程

**Step 1: 线性变换得到Q, K, V**
```
输入 X (seq_len, d_model)
     ↓
Q = X @ W_q    (seq_len, d_k)
K = X @ W_k    (seq_len, d_k)
V = X @ W_v    (seq_len, d_v)
```

**Step 2: 计算注意力分数**
```
Attention Scores = Q @ K^T / sqrt(d_k)
                 (seq_len, seq_len)
```
- 除以sqrt(d_k)是为了防止点积过大导致softmax梯度消失

**Step 3: Softmax归一化**
```
Attention Weights = softmax(Attention Scores)
                   (seq_len, seq_len)
```
- 每一行的和为1，表示对每个位置，所有位置的权重和为1

**Step 4: 加权求和**
```
Output = Attention Weights @ V
        (seq_len, d_v)
```

#### 完整公式
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

#### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o

其中 head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)
```

**多头的作用**：
- 每个头可以关注不同的依赖关系
- 例如：一个头关注语法关系，另一个头关注语义关系
- 最后拼接所有头的输出，综合多种信息

#### 计算复杂度分析
- 时间复杂度：O(n² · d)，其中n是序列长度，d是维度
- 空间复杂度：O(n²)，需要存储注意力矩阵
- 这是Transformer处理长序列的主要瓶颈

### 自测题

#### 问题
1. Self-Attention中Q、K、V分别代表什么？它们是如何计算的？
2. 为什么在计算注意力分数时要除以sqrt(d_k)？
3. Multi-Head Attention相比单头有什么优势？
4. Self-Attention的时间复杂度是多少？为什么？
5. 请手写Self-Attention的计算过程（伪代码或公式）。

#### 答案

**1. Self-Attention中Q、K、V分别代表什么？它们是如何计算的？**

- **Q (Query)**：查询向量，表示当前token想要查询的信息
- **K (Key)**：键向量，表示其他token的特征，用于被查询
- **V (Value)**：值向量，表示token的实际内容信息

计算方式：
```
Q = X @ W_q    # X是输入，W_q是可学习的权重矩阵
K = X @ W_k
V = X @ W_v
```

类比理解：
- Q就像图书馆的检索关键词
- K就像每本书的标签/索引
- V就像书的内容
- 通过Q和K的匹配找到相关的V

**2. 为什么在计算注意力分数时要除以sqrt(d_k)？**

原因：
- 当d_k较大时，Q和K的点积结果也会很大
- 过大的值会导致softmax函数进入饱和区，梯度接近0
- 除以sqrt(d_k)可以将点积结果缩放到合理范围
- 假设Q和K的元素独立同分布，均值为0，方差为1，则点积的均值为0，方差为d_k
- 除以sqrt(d_k)后，方差变为1，保持数值稳定

**3. Multi-Head Attention相比单头有什么优势？**

优势：
- **捕捉多种依赖关系**：不同头可以学习关注不同类型的依赖
  - 头1：关注语法关系（主谓宾）
  - 头2：关注语义关系（同义词）
  - 头3：关注位置关系（相邻词）
- **增强表达能力**：多个子空间的特征拼接，信息更丰富
- **并行计算**：多个头可以并行计算，不增加计算时间

**4. Self-Attention的时间复杂度是多少？为什么？**

时间复杂度：**O(n² · d)**，其中n是序列长度，d是向量维度

分析：
- Q @ K^T：O(n · d) × O(d · n) = O(n² · d)
- Softmax：O(n²)
- Attention Weights @ V：O(n² · d)
- 总体：O(n² · d)

**问题**：当序列长度n很大时，n²会变得非常大，这是Transformer处理长序列的主要瓶颈。

**5. 请手写Self-Attention的计算过程（伪代码或公式）。**

```python
def self_attention(X, W_q, W_k, W_v):
    """
    X: 输入矩阵 (seq_len, d_model)
    W_q, W_k, W_v: 权重矩阵 (d_model, d_k)
    """
    d_k = W_q.shape[1]
    
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    
    scores = Q @ K.T / math.sqrt(d_k)
    attention_weights = softmax(scores, dim=-1)
    output = attention_weights @ V
    
    return output

def multi_head_attention(X, W_q, W_k, W_v, W_o, num_heads):
    """
    X: 输入矩阵 (seq_len, d_model)
    W_q, W_k, W_v: 权重矩阵 (d_model, d_model)
    W_o: 输出权重 (d_model, d_model)
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    
    Q = Q.view(seq_len, num_heads, d_k).transpose(0, 1)
    K = K.view(seq_len, num_heads, d_k).transpose(0, 1)
    V = V.view(seq_len, num_heads, d_v).transpose(0, 1)
    
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    attention = softmax(scores, dim=-1)
    heads = attention @ V
    
    concat = heads.transpose(0, 1).contiguous().view(seq_len, d_model)
    output = concat @ W_o
    
    return output
```

---

## Day 3: FFN/MLP结构

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 查看FFN/MLP结构 | 了解基本组成 |
| 晚上 1.5h | 整理FeedForward网络 | 理解激活函数/Gate |

### 学习材料

#### 必读论文
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Section 3.3 Position-wise Feed-Forward Networks
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - GLU激活函数

#### 推荐阅读
- [SwiGLU论文解析](https://kexue.fm/archives/9019)

### 核心知识点

#### 标准FFN结构
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

或写成：
FFN(x) = ReLU(xW_1)W_2
```

**参数说明**：
- W_1: (d_model, d_ff)，通常 d_ff = 4 * d_model
- W_2: (d_ff, d_model)
- 激活函数：ReLU / GELU / SwiGLU

#### FFN的作用
- **非线性变换**：Attention是线性操作，FFN引入非线性
- **特征提取**：对每个位置独立进行特征变换
- **升维再降维**：先升维到4倍，再降回原维度，增加表达能力

#### 不同激活函数对比

| 激活函数 | 公式 | 使用模型 |
|----------|------|----------|
| **ReLU** | max(0, x) | 原始Transformer |
| **GELU** | x * Φ(x) | BERT, GPT |
| **Swish** | x * sigmoid(x) | - |
| **SwiGLU** | Swish(xW) ⊗ (xV) | LLaMA, PaLM |

#### SwiGLU详解 (LLaMA使用)
```
SwiGLU(x, W, V, W2) = (Swish(xW) ⊗ xV) W2

其中：
- Swish(x) = x * sigmoid(x)
- ⊗ 表示逐元素乘法
- W, V: 两个独立的线性变换矩阵
```

**优势**：
- 门控机制(Gating)：通过逐元素乘法实现信息筛选
- 平滑激活：Swish比ReLU更平滑，梯度更稳定
- 性能更好：在LLM中表现优于ReLU/GELU

#### FFN vs MLP
- **FFN (Feed-Forward Network)**：Transformer中的标准叫法
- **MLP (Multi-Layer Perceptron)**：更通用的术语
- 在Transformer中，两者通常指同一个组件

### 自测题

#### 问题
1. FFN在Transformer中的作用是什么？为什么需要FFN？
2. 为什么FFN通常将维度先扩大4倍再缩小？
3. ReLU和GELU的区别是什么？为什么GELU在BERT/GPT中更常用？
4. SwiGLU相比传统FFN有什么改进？LLaMA为什么选择SwiGLU？
5. 请写出标准FFN的计算公式，并解释每个参数的含义。

#### 答案

**1. FFN在Transformer中的作用是什么？为什么需要FFN？**

作用：
- **引入非线性**：Self-Attention本质上是线性变换（加权求和），FFN通过激活函数引入非线性
- **特征变换**：对每个位置的向量进行独立的特征提取和变换
- **增加模型容量**：FFN包含大量参数（约占模型总参数的2/3），增加模型表达能力

为什么需要：
- 如果只有Attention，模型只能做线性组合
- FFN让模型能够学习更复杂的特征表示
- 类似于CNN中卷积层后的全连接层

**2. 为什么FFN通常将维度先扩大4倍再缩小？**

原因：
- **增加表达能力**：更高维度的空间可以学习更复杂的特征
- **参数量平衡**：FFN参数 = 2 * d_model * d_ff，当d_ff = 4*d_model时，FFN参数量与Attention参数量相近
- **经验最优**：实验表明4倍是一个较好的平衡点
- **计算效率**：不会过度增加计算量

参数量分析：
```
Attention参数：4 * d_model * d_model (Q,K,V,O四个矩阵)
FFN参数：d_model * d_ff + d_ff * d_model = 2 * d_model * d_ff

当 d_ff = 4 * d_model 时：
FFN参数 = 8 * d_model² ≈ 2 * Attention参数
```

**3. ReLU和GELU的区别是什么？为什么GELU在BERT/GPT中更常用？**

| 特性 | ReLU | GELU |
|------|------|------|
| 公式 | max(0, x) | x * Φ(x) (Φ是标准正态分布CDF) |
| 平滑性 | 在x=0处不可导 | 处处可导 |
| 负值处理 | 完全截断 | 软性衰减 |
| 梯度 | 0或1 | 平滑变化 |

GELU更常用的原因：
- **平滑性**：处处可导，训练更稳定
- **非单调性**：对负值不是完全截断，保留部分信息
- **概率解释**：可以理解为以x的大小为概率决定是否激活
- **实验效果**：在Transformer架构中表现更好

**4. SwiGLU相比传统FFN有什么改进？LLaMA为什么选择SwiGLU？**

SwiGLU的改进：
```
传统FFN: ReLU(xW_1)W_2
SwiGLU: (Swish(xW) ⊗ xV) W_2
```

改进点：
- **门控机制**：通过两个线性变换的逐元素乘法，实现信息筛选
- **Swish激活**：x * sigmoid(x)，比ReLU更平滑
- **更多参数**：两个独立的权重矩阵W和V，增加表达能力

LLaMA选择SwiGLU的原因：
- 实验表明在相同参数量下，SwiGLU性能优于ReLU/GELU
- 门控机制能够更好地学习特征表示
- 在大规模语言模型中效果显著

**5. 请写出标准FFN的计算公式，并解释每个参数的含义。**

```
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2

简化版：
FFN(x) = ReLU(xW_1)W_2
```

参数含义：
- **x**: 输入向量，维度 (seq_len, d_model)
- **W_1**: 第一个线性层权重，维度 (d_model, d_ff)
  - d_model: 模型隐藏维度，如512/768/4096
  - d_ff: FFN中间维度，通常 d_ff = 4 * d_model
- **b_1**: 第一个线性层偏置，维度 (d_ff,)
- **W_2**: 第二个线性层权重，维度 (d_ff, d_model)
- **b_2**: 第二个线性层偏置，维度 (d_model,)
- **ReLU**: 激活函数，max(0, x)

计算流程：
```
输入 x (seq_len, d_model)
    ↓
线性变换: x @ W_1 → (seq_len, d_ff)
    ↓
激活函数: ReLU → (seq_len, d_ff)
    ↓
线性变换: @ W_2 → (seq_len, d_model)
    ↓
输出 (seq_len, d_model)
```

---

## Day 4: 归一化层 (LayerNorm/BatchNorm)

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 分析LayerNorm/BatchNorm | 了解归一化原理 |
| 晚上 1.5h | 归一化层作用 | 理解Pre-LN vs Post-LN |

### 学习材料

#### 必读论文
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - Pre-LN vs Post-LN

#### 推荐阅读
- [Batch Normalization vs Layer Normalization](https://www.baeldung.com/cs/batch-normalization-vs-layer-normalization)

### 核心知识点

#### BatchNorm vs LayerNorm

```
BatchNorm: 对每个特征，在batch维度上归一化
LayerNorm: 对每个样本，在特征维度上归一化
```

**图示**：
```
输入张量: (batch_size, seq_len, hidden_dim)

BatchNorm: 对每个hidden_dim，在batch_size维度上计算均值和方差
           统计量形状: (hidden_dim,)
           
LayerNorm: 对每个样本，在hidden_dim维度上计算均值和方差
           统计量形状: (batch_size, seq_len)
```

#### LayerNorm公式
```
LN(x) = γ * (x - μ) / σ + β

其中：
- μ: 均值，在特征维度上计算
- σ: 标准差，在特征维度上计算
- γ, β: 可学习的缩放和偏移参数
```

#### 为什么Transformer用LayerNorm而不是BatchNorm？

| 原因 | 说明 |
|------|------|
| **序列长度可变** | 不同样本的序列长度可能不同，BatchNorm难以处理 |
| **小batch size** | NLP任务batch size通常较小，BatchNorm统计量不稳定 |
| **独立归一化** | LayerNorm对每个样本独立归一化，不受batch影响 |
| **推理一致性** | LayerNorm训练和推理行为一致，BatchNorm需要维护running stats |

#### Pre-LN vs Post-LN

```
Post-LN (原始Transformer):
x + Dropout(Attention(x))
    ↓
LayerNorm
    ↓
x + Dropout(FFN(x))
    ↓
LayerNorm

Pre-LN (LLaMA等现代模型):
LayerNorm
    ↓
x + Dropout(Attention(x))
    ↓
LayerNorm
    ↓
x + Dropout(FFN(x))
```

#### Pre-LN vs Post-LN 对比

| 特性 | Post-LN | Pre-LN |
|------|---------|--------|
| 归一化位置 | 残差连接之后 | 残差连接之前 |
| 训练稳定性 | 较差，需要warmup | 较好，不需要warmup |
| 梯度流动 | 梯度需要经过LN | 梯度可以直接通过残差连接 |
| 最终性能 | 略好 | 略差 |
| 使用模型 | 原始Transformer | LLaMA, GPT-2/3, BERT |

#### RMSNorm (LLaMA使用)
```
RMSNorm(x) = x / RMS(x) * γ

其中 RMS(x) = sqrt(mean(x²) + ε)

相比LayerNorm：
- 不计算均值，只计算均方根
- 更简单，计算更快
- 效果相当甚至更好
```

### 自测题

#### 问题
1. BatchNorm和LayerNorm的区别是什么？为什么Transformer使用LayerNorm？
2. LayerNorm的计算公式是什么？γ和β参数的作用是什么？
3. Pre-LN和Post-LN的区别是什么？各有什么优缺点？
4. 为什么Pre-LN训练更稳定？
5. RMSNorm相比LayerNorm有什么改进？LLaMA为什么选择RMSNorm？

#### 答案

**1. BatchNorm和LayerNorm的区别是什么？为什么Transformer使用LayerNorm？**

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | batch维度 | feature维度 |
| 统计量计算 | 跨样本计算 | 单样本内计算 |
| 对batch size依赖 | 依赖 | 不依赖 |
| 训练/推理差异 | 需要running stats | 一致 |
| 适用场景 | CNN, 图像 | NLP, Transformer |

Transformer使用LayerNorm的原因：
- **序列长度可变**：不同样本长度不同，BatchNorm难以处理
- **小batch size**：NLP任务batch通常较小，BatchNorm统计不稳定
- **独立归一化**：每个样本独立处理，不受其他样本影响
- **推理一致性**：训练和推理行为完全一致

**2. LayerNorm的计算公式是什么？γ和β参数的作用是什么？**

公式：
```
LN(x) = γ * (x - μ) / σ + β

其中：
μ = mean(x)      # 在特征维度上计算均值
σ = std(x)       # 在特征维度上计算标准差
```

γ和β的作用：
- **γ (缩放参数)**：控制归一化后的缩放，恢复特征的表达能力
- **β (偏移参数)**：控制归一化后的偏移，允许模型学习非零均值

为什么需要γ和β：
- 归一化后数据均值为0，方差为1
- 但某些特征可能需要非零均值或不同的方差
- γ和β让模型可以学习最优的特征分布
- 如果γ=σ, β=μ，可以恢复原始分布

**3. Pre-LN和Post-LN的区别是什么？各有什么优缺点？**

结构区别：
```
Post-LN:  Output = LN(x + Sublayer(x))
Pre-LN:   Output = x + Sublayer(LN(x))
```

| 特性 | Post-LN | Pre-LN |
|------|---------|--------|
| **优点** | 最终性能略好 | 训练稳定，不需要warmup |
| **缺点** | 训练不稳定，需要warmup | 性能略差 |
| **梯度流动** | 梯度需经过LN | 梯度可直接通过残差 |
| **使用模型** | 原始Transformer | LLaMA, GPT-2/3 |

**4. 为什么Pre-LN训练更稳定？**

原因分析：

**Post-LN的梯度问题**：
```
梯度需要经过: 残差连接 → LayerNorm → 下一层
LayerNorm会重新缩放梯度，可能导致梯度消失或爆炸
深层网络中，梯度需要经过多个LayerNorm，累积效应明显
```

**Pre-LN的梯度优势**：
```
梯度可以直接通过残差连接传递: x → x + Sublayer(LN(x))
残差连接提供了一条"梯度高速公路"
LayerNorm只影响Sublayer的输出，不影响残差连接
梯度可以无损地传递到浅层
```

数学解释：
```
Post-LN: ∂L/∂x = ∂L/∂LN * ∂LN/∂(x + sublayer) * (1 + ∂sublayer/∂x)
Pre-LN:  ∂L/∂x = ∂L/∂output * (1 + ∂sublayer/∂LN * ∂LN/∂x)

Pre-LN中，梯度可以直接通过"1"这一项传递，不受LN影响
```

**5. RMSNorm相比LayerNorm有什么改进？LLaMA为什么选择RMSNorm？**

RMSNorm公式：
```
RMSNorm(x) = x / RMS(x) * γ
RMS(x) = sqrt(mean(x²) + ε)
```

相比LayerNorm的改进：
| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 计算内容 | 均值 + 方差 | 均方根 |
| 偏移参数 | 有β | 无β |
| 计算量 | 较大 | 较小 |
| 效果 | 基准 | 相当或更好 |

LLaMA选择RMSNorm的原因：
- **计算效率**：不需要计算均值，计算更快
- **效果相当**：实验表明效果与LayerNorm相当甚至更好
- **简化设计**：去除偏移参数β，模型更简洁
- **大规模验证**：在大模型中表现稳定

---

## Day 5: 位置编码

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 分析位置编码实现 | 了解不同编码方式 |
| 晚上 1.5h | 位置编码原理 | 理解各编码方式优缺点 |

### 学习材料

#### 必读论文
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Section 3.5 Positional Encoding
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) - ALiBi

#### 推荐阅读
- [RoPE位置编码详解](https://kexue.fm/archives/8265)

### 核心知识点

#### 为什么需要位置编码？

Transformer的自注意力机制是**置换不变**的：
```
Attention([A, B, C]) = Attention([C, B, A])  # 输出相同
```

这意味着模型无法区分token的位置，因此需要位置编码来注入位置信息。

#### 三种主流位置编码

##### 1. 绝对位置编码 (Sinusoidal)

**原始Transformer使用**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

使用方式: x = x + PE
```

**特点**：
- 固定编码，不需要学习
- 可以处理任意长度序列
- 通过三角函数的性质，位置关系可以被模型学习

##### 2. RoPE (Rotary Position Embedding)

**LLaMA使用**
```
将位置信息通过旋转矩阵注入到Q和K中

对于位置m的token:
q_m = R_m * q
k_m = R_m * k

其中 R_m 是旋转矩阵:
R_m = [[cos(mθ), -sin(mθ)],
       [sin(mθ),  cos(mθ)]]
```

**核心思想**：
- 通过旋转矩阵，让位置m和位置n的token之间的点积包含位置差信息
- Attention(m, n) ∝ cos((m-n)θ)，只与相对位置有关

**优势**：
- **相对位置感知**：自动编码相对位置信息
- **长度外推**：可以处理比训练时更长的序列
- **理论优雅**：数学形式简洁

##### 3. ALiBi (Attention with Linear Biases)

```
在计算注意力分数时，加入线性偏置:
Attention(m, n) = q_m · k_n - m * |m - n|

其中m是斜率，每个注意力头不同
```

**特点**：
- 不需要位置编码，直接修改注意力计算
- 线性偏置惩罚远距离token
- 长度外推能力强

#### 三种位置编码对比

| 特性 | 绝对位置编码 | RoPE | ALiBi |
|------|-------------|------|-------|
| 编码方式 | 绝对位置 | 相对位置 | 相对位置 |
| 是否可学习 | 否(可改为可学习) | 否 | 否 |
| 长度外推 | 一般 | 好 | 很好 |
| 计算开销 | 小 | 中 | 小 |
| 使用模型 | 原始Transformer, BERT | LLaMA, GPT-NeoX | BLOOM, MPT |

### 自测题

#### 问题
1. 为什么Transformer需要位置编码？
2. 绝对位置编码(Sinusoidal)的公式是什么？为什么使用sin和cos？
3. RoPE的核心思想是什么？为什么它能编码相对位置？
4. ALiBi是如何实现长度外推的？
5. 对比三种位置编码的优缺点，LLaMA为什么选择RoPE？

#### 答案

**1. 为什么Transformer需要位置编码？**

原因：
- **置换不变性**：Transformer的自注意力机制对输入顺序不敏感
  ```
  Attention([A, B, C]) = Attention([C, B, A])
  ```
- **语义依赖位置**：自然语言中，词序非常重要
  - "我吃饭" vs "饭吃我" - 完全不同的语义
  - "我喜欢你" vs "你喜欢我" - 主宾关系颠倒
- **位置信息注入**：位置编码让模型能够区分不同位置的相同token

**2. 绝对位置编码(Sinusoidal)的公式是什么？为什么使用sin和cos？**

公式：
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

pos: token位置 (0, 1, 2, ...)
i: 维度索引 (0, 1, 2, ..., d_model/2-1)
```

使用sin和cos的原因：
- **唯一性**：每个位置有唯一的编码
- **周期性**：sin/cos是周期函数，可以捕捉位置模式
- **相对位置**：利用三角恒等式，可以表达相对位置关系
  ```
  sin(pos + k) = sin(pos)cos(k) + cos(pos)sin(k)
  ```
- **多尺度**：不同维度有不同的频率，捕捉不同粒度的位置信息
  - 低频维度：捕捉长距离位置关系
  - 高频维度：捕捉短距离位置关系
- **外推能力**：可以处理任意长度的序列

**3. RoPE的核心思想是什么？为什么它能编码相对位置？**

核心思想：
- 将位置信息通过**旋转矩阵**注入到Q和K中
- 不是加法(如绝对位置编码)，而是乘法(旋转)

数学原理：
```
对于位置m的token，其query向量:
q_m = R_m @ q

其中 R_m 是旋转矩阵:
R_m = [[cos(mθ), -sin(mθ)],
       [sin(mθ),  cos(mθ)]]

当计算位置m和位置n的注意力时:
q_m · k_n = (R_m @ q) · (R_n @ k)
          = q · (R_m^T @ R_n @ k)
          = q · R_{n-m} @ k
```

关键点：
- 点积结果只与**相对位置(n-m)**有关
- 自动实现了相对位置编码
- 不需要额外的位置编码向量

**4. ALiBi是如何实现长度外推的？**

ALiBi的机制：
```
Attention(m, n) = softmax(q_m · k_n - m * |m - n|)

其中 m 是每个注意力头的斜率，通常为:
m = 1 / 2^(head_index)
```

长度外推原理：
- **线性惩罚**：距离越远，惩罚越大
  - 训练时最大长度L，token距离最多L
  - 推理时长度>L，但惩罚机制仍然有效
- **不依赖绝对位置**：只关心相对距离|m-n|
- **平滑过渡**：线性惩罚使得模型对长距离token有合理的"忽略"行为
- **斜率设计**：不同头使用不同斜率，捕捉不同尺度的位置关系

**5. 对比三种位置编码的优缺点，LLaMA为什么选择RoPE？**

| 特性 | 绝对位置编码 | RoPE | ALiBi |
|------|-------------|------|-------|
| **优点** | 简单直接 | 相对位置、外推好 | 外推最强、计算简单 |
| **缺点** | 外推能力弱 | 计算稍复杂 | 可能损失精度 |
| **外推能力** | ★★☆ | ★★★★ | ★★★★★ |
| **训练效果** | ★★★★ | ★★★★★ | ★★★★ |

LLaMA选择RoPE的原因：
- **相对位置感知**：自动编码相对位置，更符合语言特性
- **长度外推**：可以处理比训练时更长的序列
- **性能优异**：在LLM任务中表现最好
- **理论优雅**：数学形式简洁，易于理解和实现
- **广泛验证**：在多个大模型中得到验证

---

## Day 6: 实战 - 手画Transformer结构图

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 晚上 2h | 手画Transformer结构图并标注 | 输出结构图 |

### 实战要求

#### 任务描述
手动绘制完整的Transformer架构图，要求包含：

1. **输入部分**
   - Input Embedding
   - Positional Encoding

2. **Encoder部分** (标注N层)
   - Multi-Head Attention
   - Add & Norm
   - Feed Forward
   - Add & Norm

3. **Decoder部分** (标注N层)
   - Masked Multi-Head Attention
   - Add & Norm
   - Encoder-Decoder Attention
   - Add & Norm
   - Feed Forward
   - Add & Norm

4. **输出部分**
   - Linear
   - Softmax

5. **标注**
   - 各组件名称
   - 数据流向箭头
   - 关键维度变化

### 验证标准
- [ ] 结构完整，无遗漏
- [ ] 标注清晰，易于理解
- [ ] 数据流向正确
- [ ] 维度变化正确

---

## 学习问答

### 核心概念问答

#### 1. Encoder与Decoder的核心差异
**问**：Encoder和Decoder的主要区别是什么？
**答**：
- **功能定位**：Encoder负责理解输入序列（双向上下文），Decoder负责生成输出序列（自回归+编码器信息）
- **注意力机制**：Encoder使用双向自注意力（可关注所有位置），Decoder使用掩码自注意力（仅关注历史位置）+交叉注意力（关注Encoder输出）
- **典型应用**：Encoder用于文本理解（如BERT），Decoder用于文本生成（如GPT）

#### 2. Dropout的作用
**问**：什么是Dropout？为什么需要它？
**答**：
- **定义**：训练时随机将部分神经元输出置为0的正则化技术
- **作用**：防止过拟合，减少神经元共适应，模拟集成学习效果
- **应用**：在注意力层、FFN层和嵌入层添加，推理时关闭

#### 3. 位置编码的必要性
**问**：Transformer为什么需要位置编码？
**答**：
- Transformer的自注意力机制是置换不变的，无法区分token顺序
- 位置编码通过注入位置信息，让模型理解序列顺序
- 主流方案：正弦余弦编码、RoPE（旋转位置编码）、ALiBi（线性偏置编码）

### 多头注意力细节

#### 4. W_o参数的作用
**问**：Multi-Head Attention公式中的W_o是什么？
**答**：
- **定义**：输出投影矩阵（Output Projection Matrix）
- **作用**：将多头拼接结果映射回d_model维度，实现多头信息融合
- **维度**：(h×d_v, d_model)，其中h是头数，d_v是每个头的维度

#### 5. 维度转置的逻辑
**问**：为什么需要`Q.view(seq_len, num_heads, d_k).transpose(0, 1)`操作？
**答**：
- **view操作**：将d_model拆分为(num_heads, d_k)，保证同一token的特征被分配到不同头
- **transpose操作**：将num_heads维度提前，实现多头并行计算
- **内存布局**：确保数据连续存储，便于高效计算

#### 6. contiguous()的作用
**问**：为什么transpose后需要contiguous()才能view？
**答**：
- transpose会导致Tensor内存非连续，view操作要求内存连续
- contiguous()重新排列内存，使数据连续存储
- 示例：`heads.transpose(0,1).contiguous().view(seq_len, d_model)`

## Day 7: 本周复盘 + 自测

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 晚上 1h | 本周复盘 + 自测 | 检验学习效果 |

### 本周知识点回顾

#### 知识点清单
- [ ] Transformer整体架构
- [ ] Self-Attention计算过程
- [ ] Multi-Head Attention原理
- [ ] FFN/MLP结构
- [ ] 激活函数(ReLU/GELU/SwiGLU)
- [ ] LayerNorm vs BatchNorm
- [ ] Pre-LN vs Post-LN
- [ ] 位置编码(绝对/RoPE/ALiBi)

### 综合自测题

#### 问题
1. 请完整描述Transformer的架构，包括Encoder和Decoder的组成。
2. Self-Attention的计算复杂度是多少？为什么？有什么优化方法？
3. 为什么LLaMA使用SwiGLU激活函数和RMSNorm？
4. Pre-LN相比Post-LN有什么优势？为什么现代LLM普遍使用Pre-LN？
5. RoPE位置编码相比绝对位置编码有什么优势？

#### 答案

**1. 请完整描述Transformer的架构，包括Encoder和Decoder的组成。**

Transformer架构：

**输入处理**：
- Input Embedding: 将token转换为向量 (vocab_size → d_model)
- Positional Encoding: 注入位置信息 (d_model → d_model)
- 两者相加得到最终输入

**Encoder Stack (N层)**：
每层包含：
1. Multi-Head Self-Attention
   - 让每个token关注所有token
   - 捕捉全局依赖关系
2. Add & LayerNorm
   - 残差连接 + 层归一化
3. Position-wise Feed-Forward Network
   - 两层全连接，中间有激活函数
   - 维度变化: d_model → d_ff → d_model
4. Add & LayerNorm

**Decoder Stack (N层)**：
每层包含：
1. Masked Multi-Head Self-Attention
   - 只能看到当前位置之前的token
   - 保证自回归生成
2. Add & LayerNorm
3. Encoder-Decoder Attention
   - Q来自Decoder，K,V来自Encoder
   - 让Decoder关注Encoder的输出
4. Add & LayerNorm
5. Position-wise Feed-Forward Network
6. Add & LayerNorm

**输出处理**：
- Linear: d_model → vocab_size
- Softmax: 得到每个token的概率

**2. Self-Attention的计算复杂度是多少？为什么？有什么优化方法？**

复杂度：**O(n² · d)**
- n: 序列长度
- d: 向量维度

原因：
```
Q @ K^T: O(n · d) × O(d · n) = O(n² · d)
Softmax: O(n²)
Attention @ V: O(n² · d)
总计: O(n² · d)
```

优化方法：
1. **Sparse Attention**: 只关注部分位置，减少计算
   - Longformer, BigBird
2. **Linear Attention**: 用核函数近似，复杂度O(n)
   - Linear Transformer, Performer
3. **Flash Attention**: 优化内存访问，不改变复杂度但大幅加速
4. **Multi-Query Attention (MQA)**: 多个head共享K,V，减少内存
5. **Sliding Window Attention**: 只关注局部窗口

**3. 为什么LLaMA使用SwiGLU激活函数和RMSNorm？**

**SwiGLU**：
- 门控机制，信息筛选能力更强
- Swish激活比ReLU更平滑，训练更稳定
- 实验表明在LLM中性能优于ReLU/GELU

**RMSNorm**：
- 不需要计算均值，计算更快
- 效果与LayerNorm相当甚至更好
- 去除偏移参数，模型更简洁
- 在大规模模型中表现稳定

**4. Pre-LN相比Post-LN有什么优势？为什么现代LLM普遍使用Pre-LN？**

优势：
- **训练稳定**：梯度可以直接通过残差连接传递
- **不需要warmup**：Post-LN需要warmup来稳定训练
- **更容易训练深层网络**：梯度消失问题更轻

原因：
```
Post-LN: 梯度需要经过 LayerNorm → 残差连接
Pre-LN:  梯度可以直接通过 残差连接 (不经过LayerNorm)
```

现代LLM普遍使用Pre-LN：
- 模型越来越深，训练稳定性更重要
- Pre-LN在大规模训练中更可靠
- 虽然最终性能可能略差，但训练效率更高

**5. RoPE位置编码相比绝对位置编码有什么优势？**

| 特性 | 绝对位置编码 | RoPE |
|------|-------------|------|
| 位置类型 | 绝对位置 | 相对位置 |
| 外推能力 | 弱 | 强 |
| 实现方式 | 加法 | 旋转乘法 |
| 相对位置 | 需要学习 | 自动编码 |

优势：
1. **相对位置感知**：自动编码相对位置信息
   - Attention(m,n) ∝ cos((m-n)θ)
   - 只与相对位置有关
2. **长度外推**：可以处理比训练时更长的序列
   - 绝对位置编码超出训练长度后效果下降
   - RoPE可以自然外推
3. **理论优雅**：数学形式简洁
4. **性能更好**：在LLM任务中表现更优

---

## 本周学习总结

### 完成情况
- [ ] Day 1: Transformer整体架构
- [ ] Day 2: Self-Attention机制详解
- [ ] Day 3: FFN/MLP结构
- [ ] Day 4: 归一化层
- [ ] Day 5: 位置编码
- [ ] Day 6: 实战 - 手画结构图
- [ ] Day 7: 本周复盘 + 自测

### 下周预告
Week 2: 主流LLM模型结构
- GPT系列结构
- LLaMA系列结构
- Attention实现变体
- MoE和LongContext
