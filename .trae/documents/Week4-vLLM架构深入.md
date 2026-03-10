# Week 4: vLLM架构深入

## 本周目标
- 理解vLLM整体架构
- 掌握调度器源码实现
- 理解Worker通信机制
- 掌握Block Manager实现
- 理解Async Engine原理

---

## Day 22: vLLM整体架构

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 回顾vLLM整体架构 | 熟悉代码布局 |
| 晚上 1.5h | vLLM模块划分 | 能画出架构图 |

### 学习材料

#### 必读论文
- [vLLM: Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180)

#### 源码阅读
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM架构文档](https://vllm.readthedocs.io/)

### 核心知识点

#### vLLM整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM架构全景                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   API层                              │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐         │   │
│  │  │ OpenAI API│ │ REST API  │ │ gRPC API  │         │   │
│  │  └───────────┘ └───────────┘ └───────────┘         │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Engine层                           │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │              LLMEngine                       │    │   │
│  │  │  - Scheduler (调度器)                        │    │   │
│  │  │  - ModelExecutor (模型执行器)                │    │   │
│  │  │  - BlockManager (内存管理器)                 │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Worker层                           │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐         │   │
│  │  │ Worker 0  │ │ Worker 1  │ │ Worker N  │         │   │
│  │  │ (GPU 0)   │ │ (GPU 1)   │ │ (GPU N)   │         │   │
│  │  └───────────┘ └───────────┘ └───────────┘         │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Model层                            │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐         │   │
│  │  │ Attention │ │   FFN     │ │Embedding  │         │   │
│  │  │ Backends  │ │ Layers    │ │ Layers    │         │   │
│  │  └───────────┘ └───────────┘ └───────────┘         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 核心模块说明

| 模块 | 职责 | 关键文件 |
|------|------|----------|
| **LLMEngine** | 整体协调，管理请求生命周期 | `llm_engine.py` |
| **Scheduler** | 调度请求，管理资源分配 | `scheduler.py` |
| **BlockManager** | KV Cache内存管理 | `block_manager.py` |
| **ModelExecutor** | 模型执行，分布式协调 | `executor/` |
| **Worker** | 单GPU上的模型执行 | `worker.py` |
| **Model** | 模型定义，前向计算 | `models/` |

#### 请求处理流程

```
用户请求
    ↓
┌─────────────────────────────────────────────────────────┐
│ 1. API层接收请求                                         │
│    - 解析参数                                            │
│    - 创建SequenceGroup                                   │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Engine添加到等待队列                                  │
│    - add_request()                                       │
│    - 加入Scheduler的waiting队列                          │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Scheduler调度                                         │
│    - 选择要执行的请求                                    │
│    - 分配KV Cache内存                                    │
│    - 生成SchedulerOutput                                 │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Worker执行                                            │
│    - 执行模型前向计算                                    │
│    - 更新KV Cache                                        │
│    - 生成SamplerOutput                                   │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Engine处理输出                                        │
│    - 更新Sequence状态                                    │
│    - 检查是否完成                                        │
│    - 返回结果                                            │
└─────────────────────────────────────────────────────────┘
    ↓
返回用户
```

#### 关键数据结构

```python
class SequenceGroup:
    """一组相关的序列 (如Beam Search的多个候选)"""
    request_id: str
    sequences: List[Sequence]
    sampling_params: SamplingParams

class Sequence:
    """单个生成序列"""
    seq_id: int
    token_ids: List[int]
    status: SequenceStatus  # WAITING, RUNNING, FINISHED

class SchedulerOutput:
    """调度器的输出"""
    scheduled_seq_groups: List[ScheduledSequenceGroup]
    blocks_to_copy: List[Tuple[int, int]]  # Copy-on-Write
    blocks_to_swap_in: List[Tuple[int, int]]
    blocks_to_swap_out: List[Tuple[int, int]]
```

### 自测题

#### 问题
1. vLLM的整体架构分为哪几层？各层的职责是什么？
2. 请描述一个请求在vLLM中的完整处理流程。
3. SequenceGroup和Sequence的区别是什么？为什么需要SequenceGroup？
4. LLMEngine的核心职责是什么？
5. 请画出vLLM的架构图，标注关键模块。

#### 答案

**1. vLLM的整体架构分为哪几层？各层的职责是什么？**

| 层级 | 职责 |
|------|------|
| **API层** | 接收请求，解析参数，返回响应 |
| **Engine层** | 整体协调，管理请求生命周期 |
| **Worker层** | 单GPU上的模型执行 |
| **Model层** | 模型定义，前向计算 |

**2. 请描述一个请求在vLLM中的完整处理流程。**

流程：
1. API层接收请求，创建SequenceGroup
2. Engine添加到Scheduler的等待队列
3. Scheduler调度，选择请求，分配内存
4. Worker执行模型前向计算
5. Engine处理输出，更新状态，返回结果

**3. SequenceGroup和Sequence的区别是什么？为什么需要SequenceGroup？**

区别：
- **Sequence**: 单个生成序列，包含token_ids和状态
- **SequenceGroup**: 一组相关的Sequence，共享请求ID和参数

需要SequenceGroup的原因：
- 支持Beam Search (多个候选序列)
- 支持Parallel Sampling (多个独立采样)
- 统一管理同一请求的多个序列

**4. LLMEngine的核心职责是什么？**

核心职责：
- 管理请求生命周期
- 协调Scheduler和Worker
- 处理输入输出
- 管理资源分配

**5. 请画出vLLM的架构图，标注关键模块。**

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM架构                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  API层: OpenAI API / REST API / gRPC                       │
│                         ↓                                   │
│  Engine层:                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ LLMEngine                                           │   │
│  │  ├── Scheduler (调度请求)                           │   │
│  │  ├── ModelExecutor (执行模型)                       │   │
│  │  └── BlockManager (管理KV Cache)                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  Worker层:                                                 │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐               │
│  │ Worker 0  │ │ Worker 1  │ │ Worker N  │               │
│  └───────────┘ └───────────┘ └───────────┘               │
│                         ↓                                   │
│  Model层:                                                  │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐               │
│  │Attention  │ │   FFN     │ │Embedding  │               │
│  │Backends   │ │ Layers    │ │ Layers    │               │
│  └───────────┘ └───────────┘ └───────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Day 23: 调度器源码分析

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 分析调度器代码 | 了解调度逻辑 |
| 晚上 1.5h | 整理各类调度算法 | FCFS/Chris-Nice调度 |

### 学习材料

#### 源码阅读
- [vllm/core/scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py)

### 核心知识点

#### 调度器核心方法

```python
class Scheduler:
    def __init__(self, scheduler_config, cache_config):
        self.waiting: Deque[SequenceGroup] = deque()  # 等待队列
        self.running: List[SequenceGroup] = []        # 运行队列
        self.swapped: List[SequenceGroup] = []        # 换出队列
        self.block_manager = BlockManager()
    
    def schedule(self) -> SchedulerOutput:
        """核心调度方法"""
        # 1. 处理swapped队列
        # 2. 调度running队列 (Decode)
        # 3. 调度waiting队列 (Prefill)
        # 4. 处理抢占
        pass
```

#### 调度流程详解

```
schedule() 方法流程:

Step 1: 处理swapped队列
┌─────────────────────────────────────────────────────────┐
│ 尝试将swapped请求换回内存                                │
│ 如果内存足够，从swapped移到running                       │
└─────────────────────────────────────────────────────────┘
    ↓
Step 2: 调度running队列 (Decode)
┌─────────────────────────────────────────────────────────┐
│ 遍历running队列                                          │
│ 为每个请求分配新token的KV Cache                          │
│ 如果内存不足，触发抢占                                   │
└─────────────────────────────────────────────────────────┘
    ↓
Step 3: 调度waiting队列 (Prefill)
┌─────────────────────────────────────────────────────────┐
│ 按FCFS遍历waiting队列                                    │
│ 检查是否有足够内存                                       │
│ 有则加入running，否则停止                                │
└─────────────────────────────────────────────────────────┘
    ↓
Step 4: 生成SchedulerOutput
┌─────────────────────────────────────────────────────────┐
│ 返回本次要执行的请求列表                                 │
│ 包含内存操作指令 (copy, swap_in, swap_out)              │
└─────────────────────────────────────────────────────────┘
```

#### 抢占机制

```python
def _preempt(self, running_queue, target_memory):
    """抢占低优先级请求，释放内存"""
    while need_more_memory():
        # 选择要抢占的请求
        victim = self._select_victim(running_queue)
        
        # 处理方式
        if can_swap_out:
            # 换出到CPU内存
            self._swap_out(victim)
            self.swapped.append(victim)
        else:
            # 释放KV Cache，需要重新计算
            self._preempt_by_recompute(victim)
        
        self.running.remove(victim)
```

#### 调度策略对比

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| **FCFS** | 先到先服务 | 简单场景 |
| **Priority** | 按优先级调度 | 需要区分优先级 |
| **Chris-Nice** | 公平调度，防止饥饿 | 多租户场景 |

### 自测题

#### 问题
1. 调度器的三个队列(waiting, running, swapped)分别存储什么？
2. schedule()方法的核心流程是什么？
3. 什么时候会触发抢占？抢占有哪些处理方式？
4. FCFS调度有什么问题？如何改进？
5. 请分析调度器如何平衡延迟和吞吐量。

#### 答案

**1. 调度器的三个队列(waiting, running, swapped)分别存储什么？**

| 队列 | 存储内容 | 状态 |
|------|----------|------|
| **waiting** | 新到达的请求 | 等待调度 |
| **running** | 正在执行的请求 | 正在生成 |
| **swapped** | 被换出的请求 | 暂停执行 |

**2. schedule()方法的核心流程是什么？**

流程：
1. 处理swapped队列：尝试换回内存
2. 调度running队列：为Decode请求分配内存
3. 调度waiting队列：选择Prefill请求
4. 生成SchedulerOutput：返回调度结果

**3. 什么时候会触发抢占？抢占有哪些处理方式？**

触发时机：
- 新请求到达，内存不足
- Decode请求需要更多内存

处理方式：
- **Swap Out**: 换出到CPU内存，稍后恢复
- **Recompute**: 释放KV Cache，需要重新计算

**4. FCFS调度有什么问题？如何改进？**

问题：
- 长请求阻塞短请求
- 不考虑优先级
- 可能导致饥饿

改进：
- **Priority调度**: 按优先级排序
- **时间片轮转**: 限制单个请求的执行时间
- **公平调度**: 保证每个请求获得公平的资源

**5. 请分析调度器如何平衡延迟和吞吐量。**

平衡策略：
- **Continuous Batching**: 动态添加请求，提高吞吐量
- **抢占机制**: 保证高优先级请求的延迟
- **内存管理**: 合理分配资源，避免阻塞
- **Chunked Prefill**: 长请求分块，减少对短请求的阻塞

---

## Day 24: Worker通信机制

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 查看Worker通信机制 | 了解分布式推理 |
| 晚上 1.5h | 分布式推理通信 | 理解TP/PP/SP并行 |

### 学习材料

#### 源码阅读
- [vllm/worker/worker.py](https://github.com/vllm-project/vllm/blob/main/vllm/worker/worker.py)
- [vllm/executor/](https://github.com/vllm-project/vllm/tree/main/vllm/executor)

### 核心知识点

#### 分布式推理架构

```
┌─────────────────────────────────────────────────────────────┐
│                    分布式推理架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  单GPU推理:                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Worker 0 (GPU 0)                                    │   │
│  │ 完整模型                                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  张量并行 (TP):                                             │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐  │
│  │ Worker 0  │ │ Worker 1  │ │ Worker 2  │ │ Worker 3  │  │
│  │ GPU 0     │ │ GPU 1     │ │ GPU 2     │ │ GPU 3     │  │
│  │ 模型分片1  │ │ 模型分片2  │ │ 模型分片3  │ │ 模型分片4  │  │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘  │
│       ↑____________all-reduce____________↓                 │
│                                                             │
│  流水并行 (PP):                                             │
│  ┌─────────────────────┐ ┌─────────────────────┐          │
│  │ Worker 0 (Stage 0)  │ │ Worker 1 (Stage 1)  │          │
│  │ GPU 0: Layer 0-15   │→│ GPU 1: Layer 16-31  │          │
│  └─────────────────────┘ └─────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 张量并行 (Tensor Parallelism)

```
原理: 将模型层内的矩阵切分到多个GPU

Attention层TP:
┌─────────────────────────────────────────────────────────┐
│ GPU 0: Head 0-7                                         │
│ GPU 1: Head 8-15                                        │
│ GPU 2: Head 16-23                                       │
│ GPU 3: Head 24-31                                       │
│                                                         │
│ 每个GPU计算部分head的attention                           │
│ 通过all-reduce汇总结果                                   │
└─────────────────────────────────────────────────────────┘

FFN层TP:
┌─────────────────────────────────────────────────────────┐
│ GPU 0: FFN分片0 (d_model/4, d_ff)                       │
│ GPU 1: FFN分片1                                         │
│ GPU 2: FFN分片2                                         │
│ GPU 3: FFN分片3                                         │
│                                                         │
│ 每个GPU计算部分FFN                                       │
│ 通过all-reduce汇总结果                                   │
└─────────────────────────────────────────────────────────┘
```

#### 流水并行 (Pipeline Parallelism)

```
原理: 将模型层切分到多个GPU

Stage划分:
┌─────────────────────────────────────────────────────────┐
│ Stage 0 (GPU 0): Layer 0-15                             │
│ Stage 1 (GPU 1): Layer 16-31                            │
│ Stage 2 (GPU 2): Layer 32-47                            │
│ Stage 3 (GPU 3): Layer 48-63                            │
└─────────────────────────────────────────────────────────┘

数据流:
Input → Stage 0 → Stage 1 → Stage 2 → Stage 3 → Output

特点:
- 层间串行，层内并行
- 需要流水线调度
- 适合超深模型
```

#### Worker通信

```python
class Worker:
    def __init__(self, rank, local_rank):
        self.rank = rank  # 全局rank
        self.local_rank = local_rank  # 本地rank
        self.model_runner = ModelRunner()
    
    def execute_model(self, scheduler_output):
        """执行模型前向计算"""
        # 1. 准备输入
        input_ids, positions, kv_caches = self._prepare_inputs(scheduler_output)
        
        # 2. 执行模型
        hidden_states = self.model_runner.forward(
            input_ids, positions, kv_caches
        )
        
        # 3. 采样
        next_tokens = self.sampler.sample(hidden_states)
        
        return next_tokens
```

### 自测题

#### 问题
1. 张量并行(TP)和流水并行(PP)的区别是什么？
2. TP中如何切分Attention和FFN层？
3. Worker之间如何通信？使用什么通信原语？
4. 为什么TP需要all-reduce？PP需要什么通信？
5. 请分析TP和PP的优缺点及适用场景。

#### 答案

**1. 张量并行(TP)和流水并行(PP)的区别是什么？**

| 特性 | TP | PP |
|------|----|----|
| 切分方式 | 层内切分 | 层间切分 |
| 通信模式 | all-reduce | point-to-point |
| 通信频率 | 每层 | 每stage |
| 延迟 | 低 | 高 |
| 适用场景 | 大宽度模型 | 深层模型 |

**2. TP中如何切分Attention和FFN层？**

Attention切分：
- 按head切分，每个GPU负责部分head
- Q, K, V矩阵按列切分
- 输出通过all-reduce汇总

FFN切分：
- 第一层按列切分 (d_model, d_ff/tp_size)
- 第二层按行切分 (d_ff/tp_size, d_model)
- 输出通过all-reduce汇总

**3. Worker之间如何通信？使用什么通信原语？**

通信原语：
- **all-reduce**: 所有GPU汇总结果 (TP)
- **all-gather**: 收集所有GPU的数据
- **reduce-scatter**: 分散汇总结果
- **send/recv**: 点对点通信 (PP)

实现：
- NCCL (NVIDIA GPU)
- Gloo (CPU)
- MPI (通用)

**4. 为什么TP需要all-reduce？PP需要什么通信？**

TP需要all-reduce原因：
- 每个GPU计算部分结果
- 需要汇总得到完整输出
- 每层都需要通信

PP需要通信：
- send/recv: stage之间传递激活值
- 只在stage边界通信
- 通信频率低

**5. 请分析TP和PP的优缺点及适用场景。**

TP:
- 优点: 延迟低，适合推理
- 缺点: 通信开销大，扩展性受限
- 适用: 大宽度模型，低延迟场景

PP:
- 优点: 通信少，扩展性好
- 缺点: 延迟高，流水线气泡
- 适用: 深层模型，大batch场景

---

## Day 25: Block Manager实现

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 分析Block Manager代码 | 了解内存管理 |
| 晚上 1.5h | KV Cache内存管理 | 理解物理/逻辑块管理 |

### 学习材料

#### 源码阅读
- [vllm/core/block_manager.py](https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager.py)

### 核心知识点

#### Block Manager架构

```python
class BlockManager:
    def __init__(self, block_size, num_gpu_blocks, num_cpu_blocks):
        self.block_size = block_size
        self.gpu_allocator = GPUAllocator(num_gpu_blocks)
        self.cpu_allocator = CPUAllocator(num_cpu_blocks)
        self.block_tables: Dict[int, BlockTable] = {}
    
    def can_allocate(self, seq_group) -> bool:
        """检查是否有足够内存"""
        pass
    
    def allocate(self, seq_group) -> None:
        """为序列分配内存"""
        pass
    
    def free(self, seq) -> None:
        """释放序列的内存"""
        pass
```

#### Block Table结构

```
Block Table: 每个序列的逻辑块到物理块的映射

序列A的Block Table:
┌─────────────────────────────────────────────────────────┐
│ 逻辑块索引 │ 物理块索引 │ 已使用token数 │ 引用计数      │
├─────────────────────────────────────────────────────────┤
│     0      │     15     │      16       │      1        │
│     1      │     23     │      16       │      1        │
│     2      │     7      │      8        │      1        │
└─────────────────────────────────────────────────────────┘

物理块池:
┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐
│ B0 ││ B1 ││ B2 ││... ││B15 ││B16 │...
│空闲││空闲││空闲││    ││占用││占用│
└────┘└────┘└────┘└────┘└────┘└────┘
```

#### 内存分配流程

```
allocate(seq_group) 流程:

Step 1: 计算需要的block数量
┌─────────────────────────────────────────────────────────┐
│ num_blocks = ceil(prompt_length / block_size)          │
└─────────────────────────────────────────────────────────┘
    ↓
Step 2: 检查是否有足够的空闲block
┌─────────────────────────────────────────────────────────┐
│ if free_blocks < num_blocks:                            │
│     return False                                        │
└─────────────────────────────────────────────────────────┘
    ↓
Step 3: 分配物理block
┌─────────────────────────────────────────────────────────┐
│ for i in range(num_blocks):                             │
│     physical_block = gpu_allocator.allocate()           │
│     block_table.append(physical_block)                  │
└─────────────────────────────────────────────────────────┘
    ↓
Step 4: 创建Block Table
┌─────────────────────────────────────────────────────────┐
│ self.block_tables[seq_id] = block_table                 │
└─────────────────────────────────────────────────────────┘
```

#### Copy-on-Write机制

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

### 自测题

#### 问题
1. Block Manager的核心职责是什么？
2. Block Table存储什么信息？如何实现逻辑块到物理块的映射？
3. Copy-on-Write机制解决了什么问题？如何实现？
4. 如何判断是否有足够的内存分配给新请求？
5. 请设计一个内存分配策略，支持优先级和抢占。

#### 答案

**1. Block Manager的核心职责是什么？**

核心职责：
- 管理物理Block的分配和释放
- 维护逻辑块到物理块的映射
- 支持Block共享和Copy-on-Write
- 支持GPU和CPU内存交换

**2. Block Table存储什么信息？如何实现逻辑块到物理块的映射？**

存储信息：
- 逻辑块索引 → 物理块索引
- 每个block已使用的token数
- 物理块的引用计数

映射实现：
```python
class BlockTable:
    def __init__(self):
        self.physical_blocks: List[int] = []  # 物理块索引列表
        self.num_tokens_per_block: List[int] = []  # 每个block的token数
    
    def __getitem__(self, logical_idx):
        return self.physical_blocks[logical_idx]
```

**3. Copy-on-Write机制解决了什么问题？如何实现？**

解决问题：
- 多个序列共享前缀时，修改需要独立副本
- 避免不必要的复制，节省内存

实现：
1. 共享时增加引用计数
2. 修改时检查引用计数
3. 如果ref_count > 1，分配新block并复制内容
4. 更新Block Table，减少原block的引用计数

**4. 如何判断是否有足够的内存分配给新请求？**

判断逻辑：
```python
def can_allocate(self, seq_group):
    # 计算需要的block数量
    num_blocks = ceil(seq_group.get_len() / self.block_size)
    
    # 检查空闲block数量
    free_blocks = self.gpu_allocator.num_free_blocks()
    
    return free_blocks >= num_blocks
```

**5. 请设计一个内存分配策略，支持优先级和抢占。**

```python
class PriorityBlockManager:
    def allocate_with_priority(self, seq_group, priority):
        if self.can_allocate(seq_group):
            return self.allocate(seq_group)
        
        # 尝试抢占低优先级请求
        victims = self._find_victims(priority)
        for victim in victims:
            self._preempt(victim)
            if self.can_allocate(seq_group):
                return self.allocate(seq_group)
        
        return False
    
    def _find_victims(self, min_priority):
        # 找到优先级低于min_priority的请求
        victims = []
        for seq_id, block_table in self.block_tables.items():
            if seq_id.priority < min_priority:
                victims.append(seq_id)
        return sorted(victims, key=lambda x: x.priority)
```

---

## Day 26: Async Engine原理

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 查看Async Engine | 了解异步推理 |
| 晚上 1.5h | 异步推理机制 | 理解Async推理优势 |

### 学习材料

#### 源码阅读
- [vllm/engine/async_llm_engine.py](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py)

### 核心知识点

#### Sync vs Async Engine

```
Sync Engine:
┌─────────────────────────────────────────────────────────┐
│ 请求1: [Prefill] → [Decode] → [Decode] → [完成]        │
│                                                          │
│ 请求2:                    等待... → [Prefill] → ...     │
│                                                          │
│ 问题: 请求2必须等待请求1完全完成                         │
└─────────────────────────────────────────────────────────┘

Async Engine:
┌─────────────────────────────────────────────────────────┐
│ 请求1: [Prefill] → [Decode] → [Decode] → [完成]        │
│                     ↓                                    │
│ 请求2:          [Prefill] → [Decode] → [Decode] → ...   │
│                     ↓                                    │
│ 请求3:                    [Prefill] → [Decode] → ...    │
│                                                          │
│ 优势: 请求可以交错执行，提高吞吐量                       │
└─────────────────────────────────────────────────────────┘
```

#### Async Engine架构

```python
class AsyncLLMEngine:
    def __init__(self, engine_args):
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.request_tracker = RequestTracker()
        self.background_loop = None
    
    async def generate(self, prompt, sampling_params):
        """异步生成接口"""
        # 1. 创建请求
        request_id = str(uuid.uuid4())
        
        # 2. 添加到引擎
        self.engine.add_request(request_id, prompt, sampling_params)
        
        # 3. 异步等待结果
        async for output in self._stream_output(request_id):
            yield output
    
    async def _run_engine_loop(self):
        """后台引擎循环"""
        while True:
            # 调度
            scheduler_output = self.engine.scheduler.schedule()
            
            # 执行
            output = await self.engine.step()
            
            # 处理输出
            self._process_output(output)
```

#### Streaming输出

```python
async def generate_stream(self, prompt, sampling_params):
    """流式输出"""
    async for output in self.generate(prompt, sampling_params):
        # 每生成一个token就返回
        yield {
            "text": output.text,
            "token_ids": output.token_ids,
            "finished": output.finished,
        }
```

#### Async优势

| 特性 | Sync Engine | Async Engine |
|------|-------------|--------------|
| 请求处理 | 串行 | 并行 |
| 响应方式 | 等待完成 | 流式返回 |
| 吞吐量 | 低 | 高 |
| 适用场景 | 批量处理 | 在线服务 |

### 自测题

#### 问题
1. Async Engine相比Sync Engine有什么优势？
2. Async Engine如何实现流式输出？
3. 后台引擎循环的作用是什么？
4. 如何处理Async Engine中的并发请求？
5. 请设计一个支持多用户的异步推理服务架构。

#### 答案

**1. Async Engine相比Sync Engine有什么优势？**

优势：
- **并行处理**: 多个请求可以交错执行
- **流式输出**: 每生成一个token就返回
- **高吞吐量**: 更好的资源利用
- **低延迟**: 用户更快看到响应

**2. Async Engine如何实现流式输出？**

实现：
```python
async def generate_stream(self, prompt, sampling_params):
    request_id = self._add_request(prompt, sampling_params)
    
    while True:
        output = await self._get_next_output(request_id)
        yield output
        
        if output.finished:
            break
```

**3. 后台引擎循环的作用是什么？**

作用：
- 持续调度和执行请求
- 处理新到达的请求
- 管理请求状态
- 触发输出回调

**4. 如何处理Async Engine中的并发请求？**

处理方式：
- **Request Tracker**: 跟踪所有活跃请求
- **Continuous Batching**: 动态添加和移除请求
- **异步队列**: 使用asyncio.Queue管理请求

**5. 请设计一个支持多用户的异步推理服务架构。**

```
┌─────────────────────────────────────────────────────────────┐
│                    异步推理服务架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户1 ─┐                                                   │
│  用户2 ─┼─→ API Gateway → 请求队列                         │
│  用户3 ─┘         ↓                                         │
│                   ↓                                         │
│            ┌─────────────────────────────────────────┐     │
│            │         AsyncLLMEngine                  │     │
│            │  ┌─────────────────────────────────┐   │     │
│            │  │      Background Loop            │   │     │
│            │  │  - Scheduler                    │   │     │
│            │  │  - Model Execution              │   │     │
│            │  │  - Output Processing            │   │     │
│            │  └─────────────────────────────────┘   │     │
│            └─────────────────────────────────────────┘     │
│                         ↓                                   │
│            Streaming Response → 用户                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Day 27: 实战 - 对比vLLM不同版本差异

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 晚上 2h | 对比vLLM不同版本差异 | 输出对比分析 |

### 实战要求

#### 任务描述
对比vLLM不同版本的功能和性能差异

#### 分析维度

1. **功能对比**
   - 新增功能
   - API变化
   - 配置选项

2. **性能对比**
   - TTFT
   - TPS
   - 内存使用

3. **架构变化**
   - 模块重构
   - 新增组件

### 验证标准
- [ ] 完成版本功能对比
- [ ] 完成性能对比
- [ ] 输出对比分析报告

---

## Day 28: 本周复盘 + 自测

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 晚上 1h | 本周复盘 + 自测 | 检验学习效果 |

### 本周知识点回顾

#### 知识点清单
- [ ] vLLM整体架构
- [ ] 调度器源码实现
- [ ] Worker通信机制
- [ ] Block Manager实现
- [ ] Async Engine原理

### 综合自测题

#### 问题
1. 请画出vLLM的整体架构图，描述各模块的职责。
2. 调度器如何管理请求的生命周期？三个队列的作用是什么？
3. 张量并行和流水并行有什么区别？vLLM如何支持分布式推理？
4. Block Manager如何实现高效的KV Cache管理？
5. Async Engine相比Sync Engine有什么优势？如何实现流式输出？

#### 答案

**1. 请画出vLLM的整体架构图，描述各模块的职责。**

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM架构                                  │
├─────────────────────────────────────────────────────────────┤
│  API层: 接收请求，解析参数，返回响应                        │
│                         ↓                                   │
│  Engine层:                                                 │
│  - LLMEngine: 整体协调                                     │
│  - Scheduler: 调度请求                                     │
│  - BlockManager: 管理KV Cache                              │
│                         ↓                                   │
│  Worker层: 单GPU上的模型执行                               │
│                         ↓                                   │
│  Model层: 模型定义，前向计算                               │
└─────────────────────────────────────────────────────────────┘
```

**2. 调度器如何管理请求的生命周期？三个队列的作用是什么？**

生命周期：
- 新请求 → waiting队列
- 被调度 → running队列
- 内存不足 → swapped队列
- 完成 → 移除

队列作用：
- **waiting**: 存储新到达的请求
- **running**: 存储正在执行的请求
- **swapped**: 存储被换出的请求

**3. 张量并行和流水并行有什么区别？vLLM如何支持分布式推理？**

区别：
- TP: 层内切分，all-reduce通信
- PP: 层间切分，send/recv通信

vLLM支持：
- Ray作为分布式后端
- 支持TP和PP
- 自动处理通信

**4. Block Manager如何实现高效的KV Cache管理？**

实现：
- PagedAttention: 分页管理
- 按需分配: 避免预分配浪费
- Block Table: 逻辑到物理映射
- Copy-on-Write: 支持共享

**5. Async Engine相比Sync Engine有什么优势？如何实现流式输出？**

优势：
- 并行处理多个请求
- 流式输出，低延迟
- 高吞吐量

实现：
- 后台循环持续执行
- 异步生成器yield输出
- 每生成一个token就返回

---

## 本周学习总结

### 完成情况
- [ ] Day 22: vLLM整体架构
- [ ] Day 23: 调度器源码分析
- [ ] Day 24: Worker通信机制
- [ ] Day 25: Block Manager实现
- [ ] Day 26: Async Engine原理
- [ ] Day 27: 实战 - 对比vLLM版本差异
- [ ] Day 28: 本周复盘 + 自测

### 下周预告
Week 5: NPU/Ascend推理深入 + 性能优化
- CANN推理软件栈
- 推理模式开发
- 推理算子适配
- 性能Profiling
