# LLM推理核心参数与使用场景详解

本文档整理了LLM推理中四个核心主题的关键参数变量及使用场景。

---

## 一、注意力计算 (Attention)

### 1.1 关键参数变量

| 参数名称 | 符号 | 含义 | 典型值 |
|----------|------|------|--------|
| `d_model` | d | 模型隐藏维度 | 768/4096/12288 |
| `d_head` | d_k | 每个注意力头的维度 | 64/128 |
| `n_heads` | h | 注意力头数量 | 12/32/96 |
| `n_kv_heads` | - | KV头数量(GQA/MQA) | 1/8/32 |
| `seq_len` | n | 序列长度 | 2048/4096/128K |
| `scale` | 1/√d_k | 缩放因子 | - |

### 1.2 Attention变体对比

| 变体 | Q头数 | K/V头数 | KV Cache大小 | 质量 | 速度 |
|------|-------|---------|--------------|------|------|
| **MHA** | n_heads | n_heads | 100% | 最好 | 最慢 |
| **MQA** | n_heads | 1 | ~3% | 略差 | 最快 |
| **GQA** | n_heads | n_kv_groups | ~25% | 接近MHA | 中等 |

### 1.3 Attention架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Attention变体选择                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  MHA (Multi-Head Attention)                                 │
│  ├── 场景: 小模型、质量优先、短序列                          │
│  ├── 代表: LLaMA 7B/13B, 原始Transformer                    │
│  └── 特点: 质量最好，但KV Cache大                            │
│                                                             │
│  MQA (Multi-Query Attention)                                │
│  ├── 场景: 推理速度优先、长序列、多并发                      │
│  ├── 代表: PaLM                                             │
│  └── 特点: KV Cache最小，质量略有损失                        │
│                                                             │
│  GQA (Grouped Query Attention)                              │
│  ├── 场景: 平衡质量和效率、大模型                            │
│  ├── 代表: LLaMA 2/3 (34B/70B), Mistral                     │
│  └── 特点: 质量接近MHA，效率接近MQA                          │
│                                                             │
│  FlashAttention                                             │
│  ├── 场景: 长序列训练、内存受限                              │
│  ├── 优化: 减少HBM访问，内存O(n²)→O(n)                      │
│  └── 特点: 加速2-4倍，内存大幅减少                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.4 计算复杂度

```
时间复杂度: O(n² · d)
- Q @ K^T: O(n² · d)
- Softmax: O(n²)
- @ V: O(n² · d)

KV Cache大小:
- MHA: 2 × n_layers × n_heads × seq_len × d_head
- GQA: 2 × n_layers × n_kv_heads × seq_len × d_head
- MQA: 2 × n_layers × 1 × seq_len × d_head
```

### 1.5 使用场景选择指南

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| 小模型推理 | MHA | 质量优先，KV Cache影响小 |
| 大模型推理 | GQA | 平衡质量和效率 |
| 超长序列 | MQA + FlashAttention | 减少KV Cache和内存访问 |
| 多用户并发 | GQA/MQA | 减少显存占用，支持更多并发 |

---

## 二、专家混合模型 (MoE)

### 2.1 关键参数变量

| 参数名称 | 符号 | 含义 | 典型值 |
|----------|------|------|--------|
| `n_experts` | E | 专家总数 | 8/64/128 |
| `top_k` | k | 激活的专家数量 | 2 |
| `expert_dim` | d_ff | 每个专家的中间维度 | 4×d_model |
| `gate_dim` | - | 门控网络维度 | d_model |
| `expert_capacity` | - | 每个专家的最大容量 | batch_size/k |

### 2.2 MoE架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    MoE架构                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入 x (batch, seq_len, d_model)                           │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Gate Network (门控网络)                 │   │
│  │   G(x) = Softmax(x @ W_gate)  → (batch, seq, E)     │   │
│  │   选择 top-k 个专家                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Expert Network (专家网络)               │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                   │   │
│  │  │ E_1 │ │ E_2 │ │ E_3 │ │...  │ │ E_n │           │   │
│  │  │FFN  │ │FFN  │ │FFN  │ │     │ │FFN  │           │   │
│  │  └─────┘ └─────┘ └─────┘ └─────┘                   │   │
│  │         只计算被选中的top-k个专家                     │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  加权求和: output = Σ G(x)_i × Expert_i(x)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 参数量与计算量对比

| 指标 | Dense模型 | MoE模型 (Mixtral 8x7B) |
|------|-----------|------------------------|
| **总参数量** | 47B | 47B |
| **激活参数量** | 47B | ~13B (每token) |
| **专家数量** | 1 | 8 |
| **每专家参数** | 47B | ~7B |
| **激活专家数** | 1 | 2 |
| **推理计算量** | 100% | ~28% |

### 2.4 使用场景

```
┌─────────────────────────────────────────────────────────────┐
│                    MoE使用场景                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  适用场景:                                                   │
│  ├── 大规模模型: 需要大参数量但计算资源有限                  │
│  ├── 多任务学习: 不同专家处理不同类型任务                    │
│  ├── 推理效率: 激活参数少，推理快                            │
│  └── 长文本处理: 专家专业化处理不同内容                      │
│                                                             │
│  代表模型:                                                   │
│  ├── Mixtral 8x7B: 8个专家，激活2个                          │
│  ├── GPT-4 (推测): MoE架构                                  │
│  ├── DeepSeek-MoE: 细粒度专家划分                            │
│  └── Switch Transformer: 单专家激活                          │
│                                                             │
│  不适用场景:                                                 │
│  ├── 小模型: 专家数量少，优势不明显                          │
│  ├── 单一任务: 专家专业化优势无法发挥                        │
│  └── 训练资源有限: MoE训练更复杂                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.5 推理时的专家选择代码示例

```python
def moe_forward(x, experts, gate, top_k=2):
    gate_scores = softmax(x @ gate.weight)  # (batch, seq, n_experts)
    topk_scores, topk_indices = topk(gate_scores, top_k)
    
    output = zeros_like(x)
    for i in range(top_k):
        expert_idx = topk_indices[:, :, i]
        expert_output = experts[expert_idx](x)
        output += topk_scores[:, :, i:i+1] * expert_output
    
    return output
```

---

## 三、批处理和序列 (Batch & Sequence)

### 3.1 关键参数变量

| 参数名称 | 符号 | 含义 | 典型值 |
|----------|------|------|--------|
| `batch_size` | B | 批次大小 | 1-256 |
| `max_seq_len` | L | 最大序列长度 | 2048/4096/128K |
| `prompt_len` | P | Prompt长度 | 变长 |
| `output_len` | G | 生成长度 | 变长 |
| `block_size` | - | PagedAttention块大小 | 16 |
| `max_num_seqs` | - | 最大并发序列数 | 256 |

### 3.2 Prefill vs Decode对比

```
┌─────────────────────────────────────────────────────────────┐
│                 Prefill vs Decode对比                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Prefill阶段:                                               │
│  ├── 输入: 整个Prompt (多个token)                           │
│  ├── 计算模式: 并行计算                                     │
│  ├── 计算特点: 计算密集型                                   │
│  ├── KV Cache: 创建并存储                                   │
│  ├── 性能指标: TTFT (Time To First Token)                   │
│  └── 计算量: O(P² · d)                                      │
│                                                             │
│  Decode阶段:                                                │
│  ├── 输入: 单个token                                        │
│  ├── 计算模式: 串行计算                                     │
│  ├── 计算特点: 内存带宽密集型                               │
│  ├── KV Cache: 读取并追加                                   │
│  ├── 性能指标: TPS (Tokens Per Second)                      │
│  └── 计算量: O((P+G) · d) 每token                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Batching策略对比

| 策略 | 描述 | 优势 | 劣势 |
|------|------|------|------|
| **Static Batching** | 固定batch，等待所有完成 | 实现简单 | GPU利用率低 |
| **Continuous Batching** | 动态添加/移除请求 | 高吞吐量 | 实现复杂 |
| **Chunked Prefill** | 长Prefill分块处理 | 减少阻塞 | 调度复杂 |

### 3.4 Continuous Batching流程

```
┌─────────────────────────────────────────────────────────────┐
│               Continuous Batching流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  时刻T0:                                                    │
│  [请求A Prefill] [请求B Prefill]                            │
│                                                             │
│  时刻T1:                                                    │
│  [请求A Decode] [请求B Decode] [请求C Prefill]              │
│                                                             │
│  时刻T2:                                                    │
│  [请求A 完成] [请求B Decode] [请求C Decode] [请求D Prefill] │
│                                                             │
│  特点:                                                      │
│  - 完成的请求立即移除                                       │
│  - 新请求动态加入                                           │
│  - 提高GPU利用率                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.5 调度器队列管理

```
Scheduler三个队列:

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Waiting Queue  │ ──→ │  Running Queue  │ ──→ │    Finished     │
│                 │     │                 │     │                 │
│  新到达的请求    │     │  正在执行的请求  │     │   完成的请求    │
│  (Prefill)      │     │  (Decode)       │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │
        │    内存不足时          │
        │    ↓                  │
        │               ┌─────────────────┐
        └──────────────→│  Swapped Queue  │
                        │  被换出的请求    │
                        │  (暂停执行)      │
                        └─────────────────┘
```

### 3.6 使用场景选择

| 场景 | 推荐策略 | 原因 |
|------|----------|------|
| 离线批量处理 | Static Batching | 请求长度相似，实现简单 |
| 在线服务 | Continuous Batching | 高吞吐量，低延迟 |
| 长Prompt混合短请求 | Chunked Prefill | 减少长请求阻塞 |

---

## 四、缓存管理 (Cache Management)

### 4.1 关键参数变量

| 参数名称 | 符号 | 含义 | 典型值 |
|----------|------|------|--------|
| `block_size` | - | 每个Block的token数 | 16 |
| `num_gpu_blocks` | - | GPU上的Block数量 | 根据显存计算 |
| `num_cpu_blocks` | - | CPU上的Block数量 | 用于swap |
| `dtype_size` | - | 数据类型字节数 | 2(FP16)/4(FP32) |
| `n_layers` | - | 模型层数 | 32/80 |
| `ref_count` | - | Block引用计数 | 用于共享管理 |

### 4.2 KV Cache大小计算

```
KV Cache大小 = 2 × n_layers × n_kv_heads × seq_len × d_head × dtype_size

示例计算 (LLaMA 70B):
- n_layers = 80
- n_kv_heads = 8 (GQA)
- d_head = 128
- seq_len = 4096
- dtype_size = 2 (FP16)

KV Cache = 2 × 80 × 8 × 4096 × 128 × 2 ≈ 1.3GB per request
```

### 4.3 PagedAttention架构

```
┌─────────────────────────────────────────────────────────────┐
│                 PagedAttention架构                           │
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
│  │ref:││ref:││ref:││ref:││ref:││ref:│                    │
│  │ 2  ││ 1  ││ 1  ││ 1  ││ 1  ││ 1  │                    │
│  └────┘└────┘└────┘└────┘└────┘└────┘                    │
│                                                             │
│  Block结构:                                                 │
│  ┌─────────────────────────────────────┐                   │
│  │ Token 0-15: K_0-15, V_0-15          │                   │
│  │ 维度: (block_size, n_kv_heads, d_head)│                 │
│  │ ref_count: 引用计数                  │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 KV Cache生命周期

```
┌─────────────────────────────────────────────────────────────┐
│                 KV Cache生命周期                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 创建 (Prefill阶段)                                      │
│     ├── 处理Prompt，创建初始KV Cache                        │
│     └── 分配Block，建立Block Table                          │
│                                                             │
│  2. 扩展 (Decode阶段)                                       │
│     ├── 每生成一个token，追加新的KV                         │
│     └── 按需分配新的Block                                   │
│                                                             │
│  3. 管理 (运行时)                                           │
│     ├── 内存分配/释放                                       │
│     ├── 共享/复制 (Copy-on-Write)                           │
│     └── 淘汰/压缩                                           │
│                                                             │
│  4. 释放 (请求结束)                                         │
│     ├── 请求完成，释放KV Cache                              │
│     └── 减少引用计数，回收Block                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.5 KV Cache压缩技术

| 技术 | 策略 | 优势 | 劣势 |
|------|------|------|------|
| **H2O** | 保留最近K个+重要token | 性能损失小 | 需要重要性判断 |
| **StreamingLLM** | 保留开头K个+最近L个 | 支持无限长度 | 可能丢失中间信息 |
| **量化** | FP16→INT8/FP8 | 大小减半 | 精度损失 |

### 4.6 Copy-on-Write机制

```
场景: 两个序列共享前缀，需要分叉

初始状态:
序列A: [Block 0] → [Block 1] → [Block 2]
序列B: [Block 0] → [Block 1] → [Block 2] (共享)
Block 0, 1的ref_count = 2

序列B需要修改Block 1:
Step 1: 分配新Block 3
Step 2: 复制Block 1的内容到Block 3
Step 3: 更新序列B的Block Table
Step 4: Block 1的ref_count减1

修改后:
序列A: [Block 0] → [Block 1] → [Block 2]
序列B: [Block 0] → [Block 3] → [Block 2]
Block 0的ref_count = 2
Block 1的ref_count = 1
Block 3的ref_count = 1
```

### 4.7 使用场景选择

| 场景 | 推荐策略 | 原因 |
|------|----------|------|
| 多请求并发 | PagedAttention | 内存利用率高，支持共享 |
| 相同Prompt前缀 | KV Cache共享 | 减少重复计算 |
| 超长上下文 | KV Cache压缩 | 支持更长序列 |
| 长对话历史 | 分层存储 | 平衡速度和容量 |

---

## 五、总结对比表

| 主题 | 核心参数 | 关键优化 | 主要场景 |
|------|----------|----------|----------|
| **注意力计算** | n_heads, n_kv_heads, d_head | GQA/MQA, FlashAttention | 平衡质量与效率 |
| **MoE** | n_experts, top_k, expert_dim | 稀疏激活 | 大模型高效推理 |
| **批处理** | batch_size, max_seq_len | Continuous Batching | 在线服务高吞吐 |
| **缓存管理** | block_size, n_layers | PagedAttention | 内存高效利用 |

---

## 六、快速参考

### 6.1 常用公式

```
1. KV Cache大小:
   KV = 2 × n_layers × n_kv_heads × seq_len × d_head × dtype_size

2. Attention计算量:
   Prefill: O(P² × d)
   Decode:  O((P+G) × d) per token

3. MoE激活参数:
   Active = base_params + top_k × expert_params

4. Block数量:
   num_blocks = ceil(seq_len / block_size)
```

### 6.2 配置示例

**LLaMA 70B配置:**
```python
config = {
    "n_layers": 80,
    "d_model": 8192,
    "n_heads": 64,
    "n_kv_heads": 8,  # GQA
    "d_head": 128,
    "max_seq_len": 4096,
    "block_size": 16,
}
```

**Mixtral 8x7B配置:**
```python
config = {
    "n_layers": 32,
    "d_model": 4096,
    "n_heads": 32,
    "n_kv_heads": 8,  # GQA
    "n_experts": 8,
    "top_k": 2,
    "expert_dim": 14336,
}
```

---

*文档创建时间: 2026-03-14*
