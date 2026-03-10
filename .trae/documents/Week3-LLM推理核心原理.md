# Week 3: LLM推理核心原理

## 本周目标
- 理解LLM推理完整流程
- 掌握Prefill vs Decode的区别
- 深入理解PagedAttention原理
- 掌握KV Cache管理策略
- 理解推理调度算法

---

## Day 15: LLM推理流程全景

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 回顾vLLM推理流程 | 了解整体架构 |
| 晚上 1.5h | LLM推理流程全景 | 能描述完整推理流程 |

### 学习材料

#### 必读论文
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - vLLM论文

#### 推荐阅读
- [vLLM源码](https://github.com/vllm-project/vllm)
- [LLM推理优化综述](https://arxiv.org/abs/2304.00595)

### 核心知识点

#### LLM推理完整流程

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM推理流程                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 输入处理                                                 │
│     ┌─────────────────────────────────────────────────┐    │
│     │ Prompt → Tokenizer → Token IDs                  │    │
│     └─────────────────────────────────────────────────┘    │
│                         ↓                                   │
│  2. Prefill阶段 (处理整个Prompt)                            │
│     ┌─────────────────────────────────────────────────┐    │
│     │ Token IDs → Embedding → Transformer Layers      │    │
│     │           → 计算并缓存KV Cache → 输出第一个token │    │
│     └─────────────────────────────────────────────────┘    │
│                         ↓                                   │
│  3. Decode阶段 (自回归生成)                                  │
│     ┌─────────────────────────────────────────────────┐    │
│     │ 循环:                                            │    │
│     │   上一个token → Embedding → Transformer Layers  │    │
│     │   → 复用KV Cache → 计算新KV → 输出下一个token    │    │
│     │   → 直到生成EOS或达到最大长度                    │    │
│     └─────────────────────────────────────────────────┘    │
│                         ↓                                   │
│  4. 输出处理                                                 │
│     ┌─────────────────────────────────────────────────┐    │
│     │ Token IDs → Detokenizer → 输出文本              │    │
│     └─────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Prefill阶段详解

```
输入: "What is AI?" (整个Prompt)

步骤:
1. Tokenization: ["What", " is", " AI", "?"] → [1234, 567, 890, 123]
2. Embedding: Token IDs → 向量 (seq_len, d_model)
3. Transformer计算:
   - 并行处理所有token
   - 计算每层的KV Cache
   - 存储KV Cache供后续使用
4. 输出: 第一个生成的token + 完整的KV Cache

特点:
- 计算密集型 (一次处理多个token)
- 生成第一个token的时间 (TTFT)
```

#### Decode阶段详解

```
输入: 上一个生成的token

步骤:
1. Token Embedding: 单个token → 向量 (1, d_model)
2. Transformer计算:
   - 只计算新token的Q, K, V
   - 拼接到已有KV Cache
   - Attention时使用完整KV Cache
3. 输出: 下一个token

循环直到:
- 生成EOS token
- 达到最大长度
- 用户停止

特点:
- 内存带宽密集型 (每次读取完整KV Cache)
- 每次只生成一个token
- 生成速度 (TPS - Tokens Per Second)
```

#### 关键性能指标

| 指标 | 全称 | 含义 | 影响因素 |
|------|------|------|----------|
| **TTFT** | Time To First Token | 首token延迟 | Prefill计算速度 |
| **TPS** | Tokens Per Second | 生成速度 | Decode计算速度 |
| **Latency** | 端到端延迟 | TTFT + 生成时间 | 模型大小、序列长度 |
| **Throughput** | 吞吐量 | 单位时间生成的token数 | Batch size、并发数 |

### 自测题

#### 问题
1. LLM推理的完整流程是什么？请描述每个阶段的作用。
2. Prefill和Decode阶段有什么区别？为什么需要区分？
3. TTFT和TPS分别受什么因素影响？如何优化？
4. 为什么Decode阶段是内存带宽密集型？
5. 请画出LLM推理的完整流程图。

#### 答案

**1. LLM推理的完整流程是什么？请描述每个阶段的作用。**

完整流程：
1. **输入处理**: Prompt → Tokenizer → Token IDs
2. **Prefill阶段**: 处理整个Prompt，计算KV Cache，生成第一个token
3. **Decode阶段**: 自回归生成后续token，复用KV Cache
4. **输出处理**: Token IDs → Detokenizer → 输出文本

各阶段作用：
- **Prefill**: 一次性处理输入，建立上下文理解
- **Decode**: 逐token生成，利用上下文信息

**2. Prefill和Decode阶段有什么区别？为什么需要区分？**

| 特性 | Prefill | Decode |
|------|---------|--------|
| 输入长度 | 整个Prompt (多个token) | 单个token |
| 计算模式 | 并行计算 | 串行计算 |
| 计算特点 | 计算密集型 | 内存带宽密集型 |
| KV Cache | 创建并存储 | 读取并追加 |
| 性能指标 | TTFT | TPS |

区分原因：
- 优化策略不同
- Prefill可以并行优化
- Decode需要KV Cache优化

**3. TTFT和TPS分别受什么因素影响？如何优化？**

TTFT影响因素：
- Prompt长度
- 模型大小
- Prefill计算速度

TTFT优化：
- 算子融合
- Flash Attention
- 张量并行

TPS影响因素：
- KV Cache大小
- 内存带宽
- Decode计算速度

TPS优化：
- KV Cache管理 (PagedAttention)
- GQA/MQA
- 量化

**4. 为什么Decode阶段是内存带宽密集型？**

原因：
- 每次生成只计算1个token，计算量小
- 但需要读取完整KV Cache
- 计算/访存比低

分析：
```
计算量: O(d²) (单token的矩阵乘法)
访存量: O(n * d) (读取n个token的KV Cache)

当n很大时，访存成为瓶颈
GPU大部分时间在等待内存传输
```

**5. 请画出LLM推理的完整流程图。**

```
用户输入: "What is AI?"
         ↓
    ┌─────────────┐
    │  Tokenizer  │
    └─────────────┘
         ↓
    Token IDs: [1234, 567, 890, 123]
         ↓
    ┌─────────────────────────────────┐
    │         Prefill阶段              │
    │  ┌─────────────────────────────┐│
    │  │ Embedding (并行处理所有token)││
    │  │         ↓                   ││
    │  │ Transformer Layers          ││
    │  │ - 计算所有token的KV         ││
    │  │ - 存储KV Cache              ││
    │  │         ↓                   ││
    │  │ 输出第一个token: "AI"       ││
    │  └─────────────────────────────┘│
    └─────────────────────────────────┘
         ↓
    ┌─────────────────────────────────┐
    │         Decode阶段               │
    │  ┌─────────────────────────────┐│
    │  │ 循环:                        ││
    │  │ 1. 上一个token → Embedding  ││
    │  │ 2. 读取KV Cache             ││
    │  │ 3. 计算新token的KV          ││
    │  │ 4. 追加到KV Cache           ││
    │  │ 5. 输出下一个token          ││
    │  │ 重复直到EOS或最大长度        ││
    │  └─────────────────────────────┘│
    │  输出: " is" " a" " field" ...  │
    └─────────────────────────────────┘
         ↓
    ┌─────────────┐
    │ Detokenizer │
    └─────────────┘
         ↓
输出: "AI is a field of computer science..."
```

---

## Day 16: PagedAttention原理

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 查阅PagedAttention论文 | 了解核心思想 |
| 晚上 1.5h | 深入PagedAttention原理 | 理解KV Cache分页管理 |

### 学习材料

#### 必读论文
- [vLLM: Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180)

#### 推荐阅读
- [vLLM源码: attention.py](https://github.com/vllm-project/vllm/blob/main/vllm/attention/backends/)

### 核心知识点

#### 传统KV Cache管理的问题

**问题1: 静态预分配**
```
传统方法:
- 为每个请求预分配最大长度的KV Cache
- 例如: max_seq_len=2048, 实际使用=100
- 浪费: 2048 - 100 = 1948个token的空间
- 内存利用率低
```

**问题2: 内存碎片**
```
请求到达和结束时间不同:
- 请求A: 长度100, 占用内存块1
- 请求B: 长度200, 占用内存块2
- 请求A结束, 释放内存块1
- 请求C: 长度150, 无法使用内存块1 (太小)
- 产生外部碎片
```

**问题3: 无法共享**
```
多个请求可能有相同的前缀:
- 请求A: "请翻译: Hello"
- 请求B: "请翻译: World"
- "请翻译: " 的KV Cache无法共享
- 重复计算
```

#### PagedAttention核心思想

**借鉴操作系统的虚拟内存管理**:
- 将KV Cache划分为固定大小的块(Block)
- 按需分配，不预分配
- 支持块级共享

```
传统方法:
┌─────────────────────────────────────────┐
│ 请求A: [预分配2048个token空间]          │
│ 实际使用: [███░░░░░░░░░░░░░░░░░░░░░░░] │
│                    浪费                  │
└─────────────────────────────────────────┘

PagedAttention:
┌────┐┌────┐┌────┐
│Block││Block││Block│  按需分配
│  0  ││  1  ││  2  │  无浪费
└────┘└────┘└────┘
```

#### PagedAttention架构

```
┌─────────────────────────────────────────────────────────────┐
│                    PagedAttention架构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Logical Block Table (逻辑块表)                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Request A: [Block 0] → [Block 1] → [Block 2]        │   │
│  │ Request B: [Block 3] → [Block 4]                    │   │
│  │ Request C: [Block 0] → [Block 5] (共享Block 0)       │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  Physical Block Pool (物理块池)                             │
│  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐                    │
│  │ B0 ││ B1 ││ B2 ││ B3 ││ B4 ││ B5 │                    │
│  │████││████││████││████││████││████│                    │
│  └────┘└────┘└────┘└────┘└────┘└────┘                    │
│                                                             │
│  Block大小: block_size (通常=16 tokens)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Block结构

```
每个Block存储:
- block_size个token的KV Cache
- 维度: (block_size, n_kv_heads, d_head)
- 引用计数: 用于共享和释放

Block示例 (block_size=4):
┌─────────────────────────────────────┐
│ Token 0: K_0, V_0                   │
│ Token 1: K_1, V_1                   │
│ Token 2: K_2, V_2                   │
│ Token 3: K_3, V_3                   │
│ ref_count: 2 (被2个请求共享)         │
└─────────────────────────────────────┘
```

#### PagedAttention计算过程

```python
def paged_attention(query, key_cache, value_cache, block_tables, context_lens):
    """
    query: (num_tokens, num_heads, head_dim)
    key_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    value_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    block_tables: (num_seqs, max_num_blocks_per_seq)
    context_lens: (num_seqs,)
    """
    # 1. 根据block_tables获取每个序列的KV Cache
    # 2. 计算attention scores
    # 3. 应用mask (根据context_lens)
    # 4. 计算weighted sum
    # 5. 返回output
```

#### 内存效率对比

| 方法 | 内存利用率 | 碎片 | 共享 |
|------|-----------|------|------|
| 预分配 | ~20-40% | 严重 | 不支持 |
| PagedAttention | ~95%+ | 无 | 支持 |

### 自测题

#### 问题
1. 传统KV Cache管理有哪些问题？
2. PagedAttention的核心思想是什么？如何解决上述问题？
3. Block的结构是什么？block_size如何选择？
4. PagedAttention如何实现KV Cache共享？
5. 请画出PagedAttention的架构图，解释逻辑块和物理块的关系。

#### 答案

**1. 传统KV Cache管理有哪些问题？**

三大问题：
1. **静态预分配浪费**: 为每个请求预分配最大长度，实际使用少，浪费严重
2. **内存碎片**: 请求动态到达和结束，产生外部碎片
3. **无法共享**: 相同前缀的请求无法共享KV Cache，重复计算

**2. PagedAttention的核心思想是什么？如何解决上述问题？**

核心思想：
- 借鉴操作系统的虚拟内存管理
- 将KV Cache划分为固定大小的Block
- 按需分配，逻辑地址映射到物理地址

解决问题：
- **预分配浪费**: 按需分配，用多少分配多少
- **内存碎片**: Block固定大小，无碎片
- **无法共享**: Block可被多个请求引用 (Copy-on-Write)

**3. Block的结构是什么？block_size如何选择？**

Block结构：
```
每个Block包含:
- block_size个token的KV Cache
- 维度: (block_size, n_kv_heads, head_dim)
- 引用计数 (ref_count): 用于共享管理
```

block_size选择：
- **太小**: 管理开销大，Block Table大
- **太大**: 内存浪费，灵活性差
- **常用值**: 16 (vLLM默认)

权衡：
```
block_size = 16:
- 每个Block存储16个token的KV
- 内存浪费最多15个token
- 管理开销适中
```

**4. PagedAttention如何实现KV Cache共享？**

实现机制：
```
场景: 多个请求有相同前缀

请求A: "请翻译: Hello"
请求B: "请翻译: World"

步骤:
1. 请求A处理"请翻译: "，分配Block 0, 1
2. 请求B处理"请翻译: "，发现已有相同前缀
3. Block 0, 1的ref_count增加
4. 请求B直接引用Block 0, 1
5. 请求B只需计算"World"的KV

Copy-on-Write:
- 共享的Block只读
- 需要修改时才复制
```

**5. 请画出PagedAttention的架构图，解释逻辑块和物理块的关系。**

```
┌─────────────────────────────────────────────────────────────┐
│                    PagedAttention架构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  逻辑视图 (每个请求看到的是连续的KV Cache)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Request A: [Token 0-15] [Token 16-31] [Token 32-47] │   │
│  │            ↓           ↓           ↓                │   │
│  │         Block 0     Block 1     Block 2             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Block Table (逻辑块到物理块的映射)                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Request A: [0] → [1] → [2]                          │   │
│  │ Request B: [0] → [3] (共享Block 0)                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  物理块池 (实际存储)                                         │
│  ┌────┐┌────┐┌────┐┌────┐                                │
│  │ B0 ││ B1 ││ B2 ││ B3 │                                │
│  │ref:││ref:││ref:││ref:│                                │
│  │ 2  ││ 1  ││ 1  ││ 1  │                                │
│  └────┘└────┘└────┘└────┘                                │
│                                                             │
│  关系:                                                      │
│  - 逻辑块: 请求视角的连续地址空间                            │
│  - 物理块: 实际存储位置                                     │
│  - Block Table: 映射关系                                   │
│  - 支持共享: 多个逻辑块映射到同一物理块                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Day 17: Prefill vs Decode深入

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 分析Prefill vs Decode | 了解计算差异 |
| 晚上 1.5h | 整理两者计算差异 | 理解算子fusion策略 |

### 学习材料

#### 推荐阅读
- [vLLM源码: LLMEngine](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py)
- [Orca: A Distributed Serving System for Transformer-Based Generative Applications](https://arxiv.org/abs/2203.03839)

### 核心知识点

#### Prefill vs Decode计算对比

```
Prefill (处理Prompt):
输入: [T1, T2, T3, T4, T5] (多个token)
计算: 并行处理所有token
Attention: 每个token可以attend到所有之前的token
输出: 第一个生成的token + 完整KV Cache

Decode (自回归生成):
输入: [T6] (单个token)
计算: 只计算新token
Attention: T6可以attend到T1-T5和自己
输出: T7 + 更新的KV Cache
```

#### 计算量对比

```
假设:
- Prompt长度: P
- 生成长度: G
- 模型参数: N
- 隐藏维度: d

Prefill计算量:
- Attention: O(P² * d)
- FFN: O(P * d²)
- 总计: O(P * (P * d + d²))

Decode计算量 (每个token):
- Attention: O(P * d) (读取KV Cache)
- FFN: O(d²)
- 总计: O((P + d) * d)

总Decode计算量: O(G * (P + d) * d)
```

#### 为什么需要分离Prefill和Decode

| 原因 | 说明 |
|------|------|
| **计算模式不同** | Prefill计算密集，Decode访存密集 |
| **优化策略不同** | Prefill适合并行优化，Decode适合访存优化 |
| **调度策略不同** | Prefill可以batch，Decode需要逐token |
| **性能指标不同** | Prefill影响TTFT，Decode影响TPS |

#### Chunked Prefill

**问题**: 长Prompt的Prefill会阻塞其他请求

**解决方案**: 将长Prefill分块处理

```
传统Prefill:
[P1, P2, ..., P1000] → 一次性处理 → 阻塞其他请求

Chunked Prefill:
[P1, ..., P100] → 处理 → 让出资源
[P101, ..., P200] → 处理 → 让出资源
...
[P901, ..., P1000] → 处理 → 完成
```

**优势**:
- 减少长Prompt对短请求的阻塞
- 提高系统响应性
- 更好的公平性

### 自测题

#### 问题
1. Prefill和Decode的计算量有什么区别？
2. 为什么Prefill是计算密集型，Decode是访存密集型？
3. Chunked Prefill解决了什么问题？
4. 如何优化Prefill和Decode的性能？
5. 请分析一个推理请求的时间分布。

#### 答案

**1. Prefill和Decode的计算量有什么区别？**

计算量对比：
```
Prefill (Prompt长度P):
- Attention: O(P² * d) - 每对token都要计算
- FFN: O(P * d²) - 每个token独立
- 总计: O(P² * d + P * d²)

Decode (生成G个token):
- 每个token的Attention: O((P+i) * d) - 读取已有KV Cache
- 每个token的FFN: O(d²)
- 总计: O(G * P * d + G * d²)
```

关键差异：
- Prefill的Attention是O(P²)，随Prompt长度平方增长
- Decode的Attention是O(P)，线性增长

**2. 为什么Prefill是计算密集型，Decode是访存密集型？**

Prefill计算密集原因：
- 一次处理多个token，计算量大
- Attention计算O(P²)，计算密集
- GPU利用率高

Decode访存密集原因：
- 每次只处理1个token，计算量小
- 需要读取完整KV Cache，访存大
- 计算/访存比低
- GPU大部分时间等待内存

**3. Chunked Prefill解决了什么问题？**

解决的问题：
- 长Prompt阻塞短请求
- 系统响应性差
- 资源利用率不均

解决方案：
- 将长Prefill分成小块
- 每块处理后让出资源
- 与其他请求交替执行

效果：
- 短请求不被长请求阻塞
- 系统响应更及时
- 更公平的资源分配

**4. 如何优化Prefill和Decode的性能？**

Prefill优化：
- **Flash Attention**: 减少内存访问
- **算子融合**: 减少kernel启动开销
- **张量并行**: 加速计算
- **Chunked Prefill**: 提高响应性

Decode优化：
- **PagedAttention**: 高效KV Cache管理
- **GQA/MQA**: 减少KV Cache大小
- **量化**: 减少内存带宽需求
- **Speculative Decoding**: 加速生成

**5. 请分析一个推理请求的时间分布。**

```
请求: Prompt长度=1000, 生成长度=100

时间分布:
┌────────────────────────────────────────────────────────┐
│                    总延迟                               │
├──────────────────────┬─────────────────────────────────┤
│    Prefill阶段       │         Decode阶段              │
│    (TTFT)            │      (100 tokens生成)           │
│                      │                                 │
│  ┌────────────────┐  │  ┌──┐┌──┐┌──┐    ┌──┐         │
│  │ Flash Attention│  │  │T1││T2││T3│... │T100│       │
│  │ 计算密集       │  │  │  ││  ││  │    │    │       │
│  └────────────────┘  │  └──┘└──┘└──┘    └──┘         │
│                      │  每个token: 访存密集            │
│  时间: ~100ms        │  时间: ~10ms * 100 = 1000ms    │
└──────────────────────┴─────────────────────────────────┘

总延迟 ≈ TTFT + 生成时间 ≈ 100ms + 1000ms = 1100ms

优化重点:
- Prefill: Flash Attention, 算子融合
- Decode: KV Cache管理, 量化
```

---

## Day 18: KV Cache管理策略

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 查看Attention实现 | 了解KV Cache实现 |
| 晚上 1.5h | KV Cache压缩技术 | 理解Cache优化 |

### 学习材料

#### 必读论文
- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)
- [StreamingLLM: Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)

### 核心知识点

#### KV Cache生命周期

```
┌─────────────────────────────────────────────────────────────┐
│                    KV Cache生命周期                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 创建 (Prefill阶段)                                      │
│     ┌─────────────────────────────────────────────────┐    │
│     │ 处理Prompt，创建初始KV Cache                     │    │
│     │ 大小: 2 * n_layers * n_kv_heads * P * d_head    │    │
│     └─────────────────────────────────────────────────┘    │
│                         ↓                                   │
│  2. 扩展 (Decode阶段)                                       │
│     ┌─────────────────────────────────────────────────┐    │
│     │ 每生成一个token，追加新的KV                      │    │
│     │ 大小持续增长                                     │    │
│     └─────────────────────────────────────────────────┘    │
│                         ↓                                   │
│  3. 管理 (运行时)                                           │
│     ┌─────────────────────────────────────────────────┐    │
│     │ - 内存分配/释放                                  │    │
│     │ - 共享/复制                                      │    │
│     │ - 淘汰/压缩                                      │    │
│     └─────────────────────────────────────────────────┘    │
│                         ↓                                   │
│  4. 释放 (请求结束)                                         │
│     ┌─────────────────────────────────────────────────┐    │
│     │ 请求完成，释放KV Cache                           │    │
│     │ 减少引用计数，可能释放物理Block                  │    │
│     └─────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### KV Cache压缩技术

**1. H2O (Heavy-Hitter Oracle)**

核心思想：保留重要的token，淘汰不重要的token

```
策略:
- 保留最近的K个token
- 保留"重击"token (对输出影响大的token)
- 淘汰中间不重要的token

效果:
- KV Cache大小固定
- 性能损失小
```

**2. StreamingLLM**

核心思想：保留注意力汇(Attention Sinks)

```
发现:
- 开头的几个token对Attention很重要
- 即使它们的内容不重要
- 称为"Attention Sinks"

策略:
- 保留开头的K个token (Attention Sinks)
- 保留最近的L个token
- 淘汰中间的token

效果:
- 支持无限长度生成
- KV Cache大小固定
```

**3. 量化压缩**

```
FP16 → INT8:
- KV Cache大小减半
- 精度损失小
- 推理速度提升

FP16 → FP8:
- KV Cache大小减半
- 精度损失更小
- 硬件支持更好
```

#### KV Cache共享

```
场景1: 相同Prompt
请求A: "翻译: Hello"
请求B: "翻译: World"
→ "翻译: " 的KV Cache可共享

场景2: 多轮对话
用户: "你好"
AI: "你好，有什么可以帮助你的？"
用户: "今天天气怎么样？"
→ 历史对话的KV Cache可复用

场景3: Beam Search
→ 多个候选序列共享前缀的KV Cache
```

### 自测题

#### 问题
1. KV Cache的生命周期是什么？各阶段做什么？
2. H2O和StreamingLLM的压缩策略有什么区别？
3. 为什么开头的token(Attention Sinks)很重要？
4. KV Cache共享有哪些场景？如何实现？
5. 请设计一个KV Cache管理策略，支持长对话场景。

#### 答案

**1. KV Cache的生命周期是什么？各阶段做什么？**

生命周期：
1. **创建 (Prefill)**: 处理Prompt，创建初始KV Cache
2. **扩展 (Decode)**: 每生成一个token，追加新的KV
3. **管理 (运行时)**: 内存分配、共享、淘汰、压缩
4. **释放 (结束)**: 请求完成，释放KV Cache

各阶段操作：
- 创建: 分配内存，计算初始KV
- 扩展: 追加新KV，可能触发淘汰
- 管理: 引用计数、共享检测、压缩
- 释放: 减少引用计数，回收内存

**2. H2O和StreamingLLM的压缩策略有什么区别？**

| 策略 | H2O | StreamingLLM |
|------|-----|--------------|
| 保留策略 | 最近K个 + 重要token | 开头K个 + 最近L个 |
| 重要性判断 | 基于Attention分数 | 不判断，固定保留开头 |
| 目标 | 减少KV Cache | 支持无限长度 |
| 复杂度 | 较高 | 较低 |

**3. 为什么开头的token(Attention Sinks)很重要？**

原因：
- **Attention分布特性**: 开头token获得大量Attention分数
- **Softmax归一化**: 即使内容无关，也需要分配分数
- **数值稳定性**: 开头token作为"锚点"，稳定Attention计算

实验发现：
- 移除开头token，模型性能急剧下降
- 保留开头token，即使内容无关，性能保持
- 开头token充当"Attention Sink"

**4. KV Cache共享有哪些场景？如何实现？**

共享场景：
1. **相同Prompt**: 多个请求有相同前缀
2. **多轮对话**: 历史对话可复用
3. **Beam Search**: 候选序列共享前缀

实现方式：
- **引用计数**: Block被多个请求引用时，ref_count增加
- **Copy-on-Write**: 共享的Block只读，修改时复制
- **前缀匹配**: 新请求检查是否有可共享的前缀

**5. 请设计一个KV Cache管理策略，支持长对话场景。**

```
策略设计:

1. 分层存储
   - 热数据: 最近N轮对话，常驻内存
   - 温数据: N-M轮对话，可换出到CPU
   - 冷数据: M轮之前，压缩存储

2. 智能淘汰
   - 保留Attention Sinks (开头token)
   - 保留最近对话
   - 淘汰中间不重要的token

3. 按需加载
   - 用户追问历史话题时，加载对应KV Cache
   - 使用语义相似度判断相关性

4. 量化压缩
   - 热数据: FP16
   - 温数据: INT8
   - 冷数据: INT4

伪代码:
```
```python
class LongConversationKVCache:
    def __init__(self, hot_size=10, warm_size=50):
        self.hot_cache = {}  # 最近10轮
        self.warm_cache = {}  # 10-50轮
        self.cold_cache = {}  # 50轮之前
    
    def add_turn(self, turn_id, kv_cache):
        if len(self.hot_cache) >= self.hot_size:
            # 将最老的移到warm
            oldest = self.hot_cache.popitem(last=False)
            self.warm_cache[oldest[0]] = quantize(oldest[1], 'int8')
        
        self.hot_cache[turn_id] = kv_cache
    
    def get_turn(self, turn_id):
        if turn_id in self.hot_cache:
            return self.hot_cache[turn_id]
        elif turn_id in self.warm_cache:
            kv = dequantize(self.warm_cache[turn_id])
            self.hot_cache[turn_id] = kv
            return kv
        else:
            # 从冷存储加载
            return self.load_from_cold(turn_id)
```

---

## Day 19: 推理调度策略

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 分析vLLM调度器 | 了解调度逻辑 |
| 晚上 1.5h | 整理各类调度算法 | 理解调度流程 |

### 学习材料

#### 必读论文
- [Orca: A Distributed Serving System for Transformer-Based Generative Applications](https://arxiv.org/abs/2203.03839)
- [FastTransformer](https://github.com/NVIDIA/FasterTransformer)

#### 推荐阅读
- [vLLM源码: scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py)

### 核心知识点

#### 推理调度问题

```
挑战:
- 多个请求同时到达
- 请求长度不同
- Prefill和Decode计算模式不同
- GPU内存有限

目标:
- 最小化延迟
- 最大化吞吐量
- 公平性
```

#### 调度策略

**1. FCFS (First Come First Serve)**

```
简单策略:
- 按到达顺序处理
- 一个请求完成后再处理下一个
- 简单但效率低

问题:
- 长请求阻塞短请求
- GPU利用率低
```

**2. Continuous Batching**

```
核心思想:
- 不等待所有请求完成
- 动态添加新请求到batch
- 完成的请求立即移除

流程:
时刻T0: [请求A Prefill] [请求B Prefill]
时刻T1: [请求A Decode] [请求B Decode] [请求C Prefill]
时刻T2: [请求A 完成] [请求B Decode] [请求C Decode] [请求D Prefill]

优势:
- 提高GPU利用率
- 减少等待时间
- 提高吞吐量
```

**3. vLLM调度器**

```
调度步骤:
1. 调度Prefill请求
   - 按FCFS顺序
   - 受内存限制

2. 调度Decode请求
   - 所有正在运行的请求
   - 受内存限制

3. 抢占 (Preemption)
   - 内存不足时，暂停部分请求
   - 释放KV Cache
   - 等待资源后恢复

调度策略:
- FCFS: 先到先服务
- Priority: 按优先级
- Chris-Nice: 公平调度
```

#### 调度流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM调度器流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: 等待队列 + 运行队列                                   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Step 1: 检查内存                                    │   │
│  │  - 计算可用Block数量                                 │   │
│  │  - 判断是否需要抢占                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Step 2: 调度Prefill请求                             │   │
│  │  - 按FCFS从等待队列选择                              │   │
│  │  - 检查内存是否足够                                  │   │
│  │  - 加入运行队列                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Step 3: 调度Decode请求                              │   │
│  │  - 从运行队列选择所有Decode请求                      │   │
│  │  - 检查内存是否足够                                  │   │
│  │  - 内存不足时抢占低优先级请求                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  输出: 本次迭代的请求列表                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 抢占策略

```
当内存不足时:
1. 选择要抢占的请求
   - 优先抢占低优先级请求
   - 优先抢占后到达的请求

2. 处理被抢占的请求
   - 方式1: 释放KV Cache，重新计算
   - 方式2: 换出到CPU内存
   - 方式3: 压缩存储

3. 恢复
   - 资源充足时恢复被抢占的请求
```

### 自测题

#### 问题
1. 推理调度面临哪些挑战？目标是什么？
2. Continuous Batching相比传统批处理有什么优势？
3. vLLM调度器的调度流程是什么？
4. 什么时候需要抢占？抢占策略是什么？
5. 请设计一个调度策略，平衡延迟和吞吐量。

#### 答案

**1. 推理调度面临哪些挑战？目标是什么？**

挑战：
- 请求动态到达，长度不同
- Prefill和Decode计算模式不同
- GPU内存有限
- 长请求阻塞短请求

目标：
- **最小化延迟**: 减少用户等待时间
- **最大化吞吐量**: 单位时间处理更多请求
- **公平性**: 不让某些请求饿死

**2. Continuous Batching相比传统批处理有什么优势？**

传统批处理：
```
等待所有请求到达 → 批量处理 → 等待所有完成 → 下一批
问题: 短请求等待长请求，GPU空闲
```

Continuous Batching：
```
动态添加新请求，完成的请求立即移除
优势:
- 减少等待时间
- 提高GPU利用率
- 提高吞吐量
```

**3. vLLM调度器的调度流程是什么？**

流程：
1. **检查内存**: 计算可用Block，判断是否需要抢占
2. **调度Prefill**: 按FCFS从等待队列选择，检查内存
3. **调度Decode**: 选择运行队列中的Decode请求，内存不足时抢占
4. **输出**: 本次迭代的请求列表

**4. 什么时候需要抢占？抢占策略是什么？**

抢占时机：
- 新请求到达，内存不足
- Decode请求需要更多内存

抢占策略：
- 优先抢占低优先级请求
- 优先抢占后到达的请求
- 处理方式：释放KV Cache / 换出到CPU / 压缩存储

**5. 请设计一个调度策略，平衡延迟和吞吐量。**

```
策略设计:

1. 分级调度
   - 短请求 (Prompt<100): 高优先级，快速响应
   - 中等请求 (100-1000): 正常优先级
   - 长请求 (>1000): 低优先级，Chunked Prefill

2. 混合批处理
   - 同时调度Prefill和Decode
   - Prefill使用部分GPU资源
   - Decode使用剩余资源

3. 自适应Batch Size
   - 根据当前负载调整batch size
   - 负载高时减小batch，降低延迟
   - 负载低时增大batch，提高吞吐量

4. 预测性调度
   - 预测请求的生成长度
   - 提前预留内存
   - 避免中途抢占

伪代码:
```
```python
class AdaptiveScheduler:
    def schedule(self, waiting_queue, running_queue, available_memory):
        scheduled = []
        
        # 1. 调度高优先级短请求
        for req in waiting_queue:
            if req.prompt_len < 100:
                if self.can_allocate(req, available_memory):
                    scheduled.append(req)
                    available_memory -= self.estimate_memory(req)
        
        # 2. 调度Decode请求
        for req in running_queue:
            if self.can_allocate(req, available_memory):
                scheduled.append(req)
                available_memory -= self.estimate_memory(req)
        
        # 3. 调度其他Prefill请求
        for req in waiting_queue:
            if req not in scheduled:
                if self.can_allocate(req, available_memory):
                    scheduled.append(req)
                    available_memory -= self.estimate_memory(req)
        
        return scheduled
```

---

## Day 20: 实战 - 部署一个ML workload

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 晚上 2h | 实战: 部署一个ML workload | 完成实践作业 |

### 实战要求

#### 任务描述
使用vLLM部署一个LLM推理服务，并分析性能

#### 步骤

1. **安装vLLM**
```bash
pip install vllm
```

2. **启动推理服务**
```bash
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000
```

3. **发送请求**
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "What is AI?",
        "max_tokens": 100,
    }
)
print(response.json())
```

4. **性能分析**
- 测量TTFT
- 测量TPS
- 分析内存使用

### 验证标准
- [ ] 成功部署vLLM服务
- [ ] 完成推理请求
- [ ] 输出性能分析报告

---

## Day 21: 本周复盘 + 自测

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 晚上 1h | 本周复盘 + 自测 | 检验学习效果 |

### 本周知识点回顾

#### 知识点清单
- [ ] LLM推理完整流程
- [ ] Prefill vs Decode区别
- [ ] PagedAttention原理
- [ ] KV Cache管理
- [ ] 推理调度策略

### 综合自测题

#### 问题
1. 请描述LLM推理的完整流程，包括Prefill和Decode阶段。
2. PagedAttention如何解决传统KV Cache管理的问题？
3. 为什么Decode阶段是内存带宽密集型？如何优化？
4. Continuous Batching相比传统批处理有什么优势？
5. 请设计一个完整的推理系统架构，包括调度、内存管理、优化策略。

#### 答案

**1. 请描述LLM推理的完整流程，包括Prefill和Decode阶段。**

完整流程：
1. **输入处理**: Prompt → Tokenizer → Token IDs
2. **Prefill阶段**:
   - 并行处理整个Prompt
   - 计算并存储KV Cache
   - 输出第一个token
3. **Decode阶段**:
   - 自回归生成后续token
   - 读取并追加KV Cache
   - 直到EOS或最大长度
4. **输出处理**: Token IDs → Detokenizer → 文本

**2. PagedAttention如何解决传统KV Cache管理的问题？**

| 问题 | PagedAttention解决方案 |
|------|----------------------|
| 预分配浪费 | 按需分配，用多少分配多少 |
| 内存碎片 | Block固定大小，无碎片 |
| 无法共享 | Block可被多个请求引用 |

**3. 为什么Decode阶段是内存带宽密集型？如何优化？**

原因：
- 每次只处理1个token，计算量小
- 需要读取完整KV Cache，访存大
- 计算/访存比低

优化方法：
- PagedAttention: 高效KV Cache管理
- GQA/MQA: 减少KV Cache大小
- 量化: 减少内存带宽需求
- Speculative Decoding: 加速生成

**4. Continuous Batching相比传统批处理有什么优势？**

优势：
- 动态添加新请求，完成的立即移除
- 减少等待时间
- 提高GPU利用率
- 提高吞吐量

**5. 请设计一个完整的推理系统架构，包括调度、内存管理、优化策略。**

```
┌─────────────────────────────────────────────────────────────┐
│                    推理系统架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   API层                              │   │
│  │  - REST API / gRPC                                  │   │
│  │  - 请求验证 / 响应格式化                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   调度层                             │   │
│  │  - Continuous Batching                              │   │
│  │  - 优先级调度                                        │   │
│  │  - 抢占策略                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   内存管理层                         │   │
│  │  - PagedAttention                                   │   │
│  │  - KV Cache共享                                      │   │
│  │  - 量化压缩                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   计算层                             │   │
│  │  - Flash Attention                                  │   │
│  │  - 算子融合                                          │   │
│  │  - 张量并行                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   硬件层                             │   │
│  │  - GPU / NPU                                        │   │
│  │  - 内存管理                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 本周学习总结

### 完成情况
- [ ] Day 15: LLM推理流程全景
- [ ] Day 16: PagedAttention原理
- [ ] Day 17: Prefill vs Decode深入
- [ ] Day 18: KV Cache管理策略
- [ ] Day 19: 推理调度策略
- [ ] Day 20: 实战 - 部署ML workload
- [ ] Day 21: 本周复盘 + 自测

### 下周预告
Week 4: vLLM架构深入
- vLLM整体架构
- 调度器源码分析
- Worker通信机制
- Block Manager实现
