# Week 7: vLLM Worker与分布式推理

## 本周目标

* 深入理解vLLM Worker的工作机制

* 掌握分布式推理的实现原理

* 理解TP/PP/SP等并行策略

* 能够阅读并理解分布式相关源码

***

## Day 1: Worker架构总览

### 学习任务

| 时间      | 任务           | 目标           |
| ------- | ------------ | ------------ |
| 工作中     | 阅读Worker相关代码 | 了解Worker入口   |
| 晚上 1.5h | Worker架构分析   | 能画出Worker架构图 |

### 学习材料

#### 源码位置

* `vllm/worker/worker.py` - Worker核心实现

* `vllm/worker/worker_base.py` - Worker基类

* `vllm/worker/model_runner.py` - 模型执行器

#### 核心类与数据结构

```python
class Worker:
    """Worker核心类 - 负责模型推理执行"""
    def __init__(self, worker_config):
        self.worker_config = worker_config
        self.model_runner: ModelRunner = None
        self.cache_engine: CacheEngine = None
        
    def init_model(self):
        """初始化模型"""
        # 1. 加载模型权重
        # 2. 初始化Cache Engine
        # 3. 设置分布式环境
        pass
        
    def execute_model(self, scheduler_output: SchedulerOutput):
        """执行模型推理"""
        # 1. 准备输入数据
        # 2. 执行模型前向传播
        # 3. 返回输出结果
        pass
```

### 核心知识点

#### Worker架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        Worker                                │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                    Model Runner                          │ │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │ │
│ │ │   Model     │ │  Attention  │ │    MLP      │         │ │
│ │ │   Layers    │ │   Layers    │ │   Layers    │         │ │
│ │ └─────────────┘ └─────────────┘ └─────────────┘         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                   Cache Engine                           │ │
│ │ ┌─────────────┐ ┌─────────────┐                         │ │
│ │ │  GPU Cache  │ │  CPU Cache  │                         │ │
│ │ └─────────────┘ └─────────────┘                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                Distributed Runtime                       │ │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │ │
│ │ │     Ray     │ │    NCCL     │ │  Gloo/NCCL  │         │ │
│ │ └─────────────┘ └─────────────┘ └─────────────┘         │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### Worker的核心职责

| 职责             | 说明          |
| -------------- | ----------- |
| **模型加载**       | 加载模型权重到GPU  |
| **推理执行**       | 执行模型前向传播    |
| **KV Cache管理** | 管理GPU/CPU缓存 |
| **分布式通信**      | 与其他Worker通信 |

### 自测题

#### 问题
1. Worker的核心职责有哪些？
2. Worker与Model Runner的关系是什么？
3. Cache Engine在Worker中的作用是什么？

#### 答案
1. **Worker核心职责**：
   - **模型加载**：加载模型权重到GPU
   - **推理执行**：执行模型前向传播
   - **KV Cache管理**：管理GPU/CPU缓存
   - **分布式通信**：与其他Worker通信
   - **资源管理**：管理GPU内存和计算资源

2. **Worker与Model Runner关系**：
   - **Worker是容器**：包含Model Runner、Cache Engine等组件
   - **Model Runner是执行器**：负责具体的模型推理执行
   - **协作关系**：Worker调用Model Runner执行推理任务
   - **生命周期**：Worker创建时初始化Model Runner

3. **Cache Engine作用**：
   - **KV Cache管理**：存储和管理键值缓存
   - **内存分配**：预分配GPU/CPU缓存空间
   - **数据交换**：处理GPU与CPU间的数据拷贝
   - **缓存优化**：实现分页、共享等优化策略

***

## Day 2: Model Runner实现

### 学习任务

| 时间      | 任务               | 目标       |
| ------- | ---------------- | -------- |
| 工作中     | 分析Model Runner代码 | 理解模型执行流程 |
| 晚上 1.5h | 模型执行优化           | 理解执行优化策略 |

### 学习材料

#### 源码位置

* `vllm/worker/model_runner.py` - Model Runner核心实现

* `vllm/model_executor/models/` - 各类模型实现

#### 核心类与数据结构

```python
class ModelRunner:
    """模型执行器"""
    def __init__(self, model_config, cache_config):
        self.model_config = model_config
        self.cache_config = cache_config
        self.model: nn.Module = None
        
    def load_model(self):
        """加载模型"""
        # 1. 根据model_config选择模型类
        # 2. 加载权重
        # 3. 移动到GPU
        pass
        
    def execute_model(self, seq_group_metadata_list, 
                      blocks_to_swap_in, 
                      blocks_to_swap_out,
                      blocks_to_copy):
        """执行模型前向传播"""
        # 1. 准备输入tensors
        input_ids, positions, input_metadata = self.prepare_input(
            seq_group_metadata_list
        )
        
        # 2. 执行swap操作
        self.cache_engine.swap_in(blocks_to_swap_in)
        self.cache_engine.swap_out(blocks_to_swap_out)
        
        # 3. 执行模型前向传播
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=self.cache_engine.gpu_cache,
            input_metadata=input_metadata
        )
        
        # 4. 返回输出
        return hidden_states
```

### 核心知识点

#### 模型执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Execution Flow                      │
│                                                              │
│  1. Prepare Input                                           │
│     ├── input_ids: [num_tokens]                             │
│     ├── positions: [num_tokens]                             │
│     └── input_metadata: attention metadata                  │
│                                                              │
│  2. Cache Operations                                        │
│     ├── swap_in: CPU → GPU                                  │
│     ├── swap_out: GPU → CPU                                 │
│     └── copy: Block间拷贝                                   │
│                                                              │
│  3. Model Forward                                           │
│     ├── Embedding Layer                                     │
│     ├── Transformer Layers (×N)                             │
│     │   ├── Attention                                       │
│     │   └── MLP                                             │
│     └── LM Head                                             │
│                                                              │
│  4. Return Output                                           │
│     └── hidden_states: [num_tokens, hidden_size]            │
└─────────────────────────────────────────────────────────────┘
```

#### 输入元数据（Input Metadata）

```python
@dataclass
class InputMetadata:
    """注意力计算的元数据"""
    # 位置信息
    positions: torch.Tensor
    
    # Block Table
    block_tables: torch.Tensor
    
    # 上下文长度
    context_lens: torch.Tensor
    
    # Padded序列长度
    max_context_len: int
    
    # 是否为prefill阶段
    is_prompt: bool
```

### 自测题

#### 问题
1. Model Runner的execute_model方法包含哪些关键步骤？
2. Input Metadata的作用是什么？包含哪些关键信息？
3. 为什么需要区分prefill和decode阶段？

#### 答案
1. **execute_model关键步骤**：
   - **准备输入**：构造input_ids、positions、input_metadata
   - **Cache操作**：执行swap_in、swap_out、copy操作
   - **模型前向**：执行Transformer模型，包含Attention和MLP层
   - **返回输出**：返回hidden_states作为推理结果

2. **Input Metadata作用**：
   - **位置信息**：记录每个token在序列中的位置
   - **Block Table**：存储KV Cache的Block索引
   - **上下文长度**：记录每个序列的有效长度
   - **元数据支持**：为Attention计算提供必要信息
   - **关键字段**：positions、block_tables、context_lens、is_prompt

3. **区分prefill和decode的原因**：
   - **计算模式不同**：prefill是并行计算，decode是串行生成
   - **KV Cache使用**：prefill需要计算并存储KV，decode只需读取
   - **性能优化**：prefill可并行处理，decode需要逐步生成
   - **内存访问**：prefill访问全部历史，decode只需最新token

***

## Day 3: Ray集成与分布式初始化

### 学习任务

| 时间      | 任务        | 目标           |
| ------- | --------- | ------------ |
| 工作中     | 分析Ray相关代码 | 理解Ray集成方式    |
| 晚上 1.5h | 分布式初始化流程  | 理解Worker启动过程 |

### 学习材料

#### 源码位置

* `vllm/engine/ray_utils.py` - Ray工具函数

* `vllm/worker/worker.py` - Worker分布式初始化

#### Ray集成核心代码

```python
class RayWorkerWrapper:
    """Ray Worker包装器"""
    def __init__(self, worker_config):
        self.worker = Worker(worker_config)
        
    def init_model(self):
        """初始化模型（Ray远程调用）"""
        return ray.get(self.worker.init_model.remote())
        
    def execute_model(self, scheduler_output):
        """执行模型（Ray远程调用）"""
        return ray.get(self.worker.execute_model.remote(scheduler_output))

def initialize_ray_cluster():
    """初始化Ray集群"""
    # 1. 连接Ray集群
    ray.init(address='auto')
    
    # 2. 获取集群资源
    cluster_resources = ray.cluster_resources()
    
    # 3. 创建Worker actors
    workers = []
    for i in range(num_workers):
        worker = ray.remote(RayWorkerWrapper).remote(worker_config)
        workers.append(worker)
        
    return workers
```

### 核心知识点

#### Ray分布式架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Ray Cluster                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                    Ray Driver                            │ │
│ │  (Scheduler, API Server, ...)                           │ │
│ └─────────────────────────────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                   Ray Object Store                       │ │
│ │  (共享内存，存储中间结果)                                │ │
│ └─────────────────────────────────────────────────────────┘ │
│                          │                                   │
│            ┌─────────────┼─────────────┐                    │
│            ▼             ▼             ▼                    │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │  Worker 0   │ │  Worker 1   │ │  Worker 2   │            │
│ │  (GPU 0)    │ │  (GPU 1)    │ │  (GPU 2)    │            │
│ └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

#### 分布式初始化流程

```python
def init_distributed_environment(self):
    """初始化分布式环境"""
    # 1. 获取分布式信息
    rank = self.worker_config.rank
    world_size = self.worker_config.world_size
    
    # 2. 初始化进程组
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
        init_method=self.worker_config.distributed_init_method
    )
    
    # 3. 设置设备
    torch.cuda.set_device(rank)
    
    # 4. 同步所有进程
    dist.barrier()
```

### 自测题

#### 问题
1. Ray在vLLM中的作用是什么？
2. 分布式初始化包含哪些关键步骤？
3. 为什么需要dist.barrier()？

#### 答案
1. **Ray的作用**：
   - **分布式计算框架**：管理多个Worker进程
   - **Actor模型**：每个Worker是一个Ray Actor
   - **资源管理**：自动分配GPU资源
   - **通信机制**：提供高效的进程间通信
   - **容错机制**：支持Worker故障恢复

2. **分布式初始化关键步骤**：
   - **获取分布式信息**：rank、world_size
   - **初始化进程组**：使用NCCL/Gloo后端
   - **设置设备**：为每个Worker分配GPU
   - **同步所有进程**：确保所有Worker初始化完成
   - **加载模型**：分布式加载模型权重

3. **dist.barrier()的作用**：
   - **同步点**：确保所有Worker都到达同一状态
   - **防止竞态**：避免Worker间状态不一致
   - **初始化同步**：确保分布式环境完全就绪
   - **训练/推理同步**：在关键操作前同步状态

***

## Day 4: Tensor Parallelism (TP)

### 学习任务

| 时间      | 任务       | 目标             |
| ------- | -------- | -------------- |
| 工作中     | 分析TP相关代码 | 理解张量并行实现       |
| 晚上 1.5h | TP通信机制   | 理解All-Reduce操作 |

### 学习材料

#### 源码位置

* `vllm/model_executor/parallel_utils/tensor_parallel.py` - TP实现

* `vllm/model_executor/layers/linear.py` - 并行线性层

#### Tensor Parallelism原理

```python
class ColumnParallelLinear(nn.Module):
    """列并行线性层 - 权重按列切分"""
    def __init__(self, input_size, output_size, world_size):
        self.input_size = input_size
        self.output_size_per_partition = output_size // world_size
        
        # 每个GPU只存储部分权重
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, input_size)
        )
        
    def forward(self, x):
        # 每个GPU计算部分输出
        output = F.linear(x, self.weight)
        
        # All-Reduce汇聚结果
        output = tensor_parallel.all_reduce(output)
        
        return output

class RowParallelLinear(nn.Module):
    """行并行线性层 - 权重按行切分"""
    def __init__(self, input_size, output_size, world_size):
        self.input_size_per_partition = input_size // world_size
        self.output_size = output_size
        
        # 每个GPU只存储部分权重
        self.weight = nn.Parameter(
            torch.empty(output_size, self.input_size_per_partition)
        )
        
    def forward(self, x):
        # 每个GPU计算部分结果
        output_parallel = F.linear(x, self.weight)
        
        # All-Reduce汇聚结果
        output = tensor_parallel.all_reduce(output_parallel)
        
        return output
```

### 核心知识点

#### TP并行策略图

```
┌─────────────────────────────────────────────────────────────┐
│              Tensor Parallelism (TP=2)                      │
│                                                              │
│  Input: [batch, hidden_size]                                │
│         ┌─────────────┬─────────────┐                       │
│         │   GPU 0     │   GPU 1     │                       │
│         ├─────────────┼─────────────┤                       │
│         │ QKV Linear  │ QKV Linear  │ (Column Parallel)     │
│         │ [h, h/2]    │ [h, h/2]    │                       │
│         └──────┬──────┴──────┬──────┘                       │
│                │             │                               │
│                ▼             ▼                               │
│         Attention      Attention                             │
│         (local)        (local)                               │
│                │             │                               │
│                └──────┬──────┘                               │
│                       ▼                                      │
│              All-Reduce                                      │
│                       │                                      │
│         ┌─────────────┴─────────────┐                       │
│         │   GPU 0     │   GPU 1     │                       │
│         ├─────────────┼─────────────┤                       │
│         │ MLP Linear  │ MLP Linear  │ (Row Parallel)        │
│         │ [h/2, h]    │ [h/2, h]    │                       │
│         └─────────────┴─────────────┘                       │
│                       │                                      │
│                       ▼                                      │
│              All-Reduce                                      │
│                       │                                      │
│                       ▼                                      │
│  Output: [batch, hidden_size]                               │
└─────────────────────────────────────────────────────────────┘
```

#### TP通信开销分析

* **通信操作**：All-Reduce

* **通信量**：`2 × batch_size × hidden_size × num_layers`

* **通信频率**：每层Transformer两次（Attention后 + MLP后）

* **优化策略**：通信与计算重叠

### 自测题

#### 问题
1. Column Parallel和Row Parallel的区别是什么？
2. TP的通信开销如何计算？如何优化？
3. 为什么Attention层使用Column Parallel，MLP层使用Row Parallel？

#### 答案
1. **Column vs Row Parallel区别**：
   - **Column Parallel**：
     - 权重按列切分：每个GPU存储部分输出维度
     - 计算：每个GPU计算部分输出，需要All-Reduce汇聚
     - 适用：输出维度较大的层（如Attention的QKV投影）
   - **Row Parallel**：
     - 权重按行切分：每个GPU存储部分输入维度
     - 计算：每个GPU计算完整输出，需要All-Reduce汇聚
     - 适用：输入维度较大的层（如MLP的第二个线性层）

2. **TP通信开销计算与优化**：
   - **通信量计算**：`2 × batch_size × hidden_size × num_layers`
   - **通信频率**：每层Transformer两次（Attention后 + MLP后）
   - **优化策略**：
     - 通信与计算重叠
     - 使用高效的All-Reduce实现
     - 合理选择TP大小，平衡通信和计算

3. **Attention层使用Column Parallel的原因**：
   - **QKV投影**：输出维度是输入维度的3倍，适合按列切分
   - **注意力计算**：QK^T计算后需要All-Reduce，自然使用Column Parallel
   - **输出投影**：Attention输出投影也适合Column Parallel
   
   **MLP层使用Row Parallel的原因**：
   - **第一个线性层**：通常先升维（如4×hidden_size），适合Row Parallel
   - **第二个线性层**：降维回hidden_size，使用Row Parallel可减少通信量
   - **计算模式**：MLP的两次矩阵乘法更适合Row + Column的组合

***

## Day 5: Pipeline Parallelism (PP)

### 学习任务

| 时间      | 任务       | 目标        |
| ------- | -------- | --------- |
| 工作中     | 分析PP相关代码 | 理解流水线并行实现 |
| 晚上 1.5h | PP调度策略   | 理解流水线调度   |

### 学习材料

#### 源码位置

* `vllm/model_executor/parallel_utils/pipeline_parallel.py` - PP实现

* `vllm/engine/pipeline_scheduler.py` - 流水线调度

#### Pipeline Parallelism原理

```python
class PipelineStage:
    """流水线阶段"""
    def __init__(self, stage_id, num_stages, layers):
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.layers = layers
        
    def forward(self, hidden_states):
        """执行当前阶段的层"""
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
        
    def send_to_next_stage(self, hidden_states):
        """发送到下一阶段"""
        if self.stage_id < self.num_stages - 1:
            dist.send(hidden_states, dst=self.stage_id + 1)
            
    def recv_from_prev_stage(self):
        """从上一阶段接收"""
        if self.stage_id > 0:
            hidden_states = torch.empty(...)
            dist.recv(hidden_states, src=self.stage_id - 1)
            return hidden_states
```

### 核心知识点

#### PP流水线调度

```
┌─────────────────────────────────────────────────────────────┐
│           Pipeline Parallelism (PP=4)                       │
│                                                              │
│  Time →                                                     │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐ │
│  │ GPU0 │ F0,0 │      │      │      │ F0,1 │      │      │ │
│  ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤ │
│  │ GPU1 │      │ F1,0 │      │      │      │ F1,1 │      │ │
│  ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤ │
│  │ GPU2 │      │      │ F2,0 │      │      │      │ F2,1 │ │
│  ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤ │
│  │ GPU3 │      │      │      │ F3,0 │      │      │      │ │
│  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘ │
│                                                              │
│  F{i,j} = 第j个micro-batch在第i个stage的前向传播           │
│                                                              │
│  问题：存在大量气泡（bubble）                                │
└─────────────────────────────────────────────────────────────┘
```

#### PP优化：GPipe vs 1F1B

```
GPipe调度（先全部前向，再全部反向）：
┌─────────────────────────────────────────────────────────────┐
│ F0,0 F0,1 F0,2 F0,3 │ B0,3 B0,2 B0,1 B0,0                   │
│      F1,0 F1,1 F1,2 F1,3 │ B1,3 B1,2 B1,1 B1,0              │
│           F2,0 F2,1 F2,2 F2,3 │ B2,3 B2,2 B2,1 B2,0         │
│                F3,0 F3,1 F3,2 F3,3 │ B3,3 B3,2 B3,1 B3,0    │
└─────────────────────────────────────────────────────────────┘

1F1B调度（交替前向反向，减少气泡）：
┌─────────────────────────────────────────────────────────────┐
│ F0,0 F0,1 F0,2 B0,0 F0,3 B0,1 B0,2 B0,3                     │
│      F1,0 F1,1 F1,2 B1,0 F1,3 B1,1 B1,2 B1,3                │
│           F2,0 F2,1 F2,2 B2,0 F2,3 B2,1 B2,2 B2,3           │
│                F3,0 F3,1 F3,2 B3,0 F3,3 B3,1 B3,2 B3,3      │
└─────────────────────────────────────────────────────────────┘
```

### 自测题

#### 问题

1. Pipeline Parallelism的基本原理是什么？
2. GPipe和1F1B调度策略的区别是什么？
3. PP的气泡（bubble）问题如何解决？

***

## Day 6: 实战 - 分布式推理源码走读

### 学习任务

| 时间    | 任务          | 目标        |
| ----- | ----------- | --------- |
| 晚上 2h | 完整走读分布式推理源码 | 能解释关键代码逻辑 |

### 实战要求

#### 任务描述

完整阅读vLLM分布式推理相关代码，重点理解：

1. **Worker初始化**：

```python
def init_distributed_environment(self):
    """初始化分布式环境 - 需要重点理解"""
    # TODO: 阅读并注释关键步骤
    pass
```

1. **TP通信实现**：

```python
def all_reduce(input_tensor):
    """All-Reduce实现 - 需要重点理解"""
    # TODO: 阅读并注释关键步骤
    pass
```

1. **PP调度实现**：

```python
def pipeline_schedule(self, micro_batches):
    """流水线调度 - 需要重点理解"""
    # TODO: 阅读并注释关键步骤
    pass
```

#### 验证标准

* [ ] 能画出TP的通信流程图

* [ ] 能解释PP的调度策略

* [ ] 能说明Worker间的协作机制

***

## Day 7: 本周复盘 + 自测

### 学习任务

| 时间    | 任务        | 目标     |
| ----- | --------- | ------ |
| 晚上 1h | 本周复盘 + 自测 | 检验学习效果 |

### 本周知识点回顾

#### 知识点清单

* [ ] Worker架构与核心职责

* [ ] Model Runner执行流程

* [ ] Ray集成与分布式初始化

* [ ] Tensor Parallelism原理与实现

* [ ] Pipeline Parallelism原理与调度

* [ ] 分布式通信机制（NCCL/All-Reduce）

### 综合自测题

#### 问题

1. 描述vLLM Worker的完整生命周期，从初始化到推理执行。
2. Tensor Parallelism和Pipeline Parallelism的区别是什么？各自的优缺点？
3. 为什么vLLM主要使用TP而不是PP？
4. All-Reduce操作的原理是什么？在TP中的作用？
5. 阅读以下代码，解释其作用：

```python
def column_parallel_linear_forward(self, x):
    # 输入x: [batch, hidden_size]
    
    # 每个GPU计算部分输出
    output_parallel = F.linear(x, self.weight)
    # output_parallel: [batch, hidden_size/tp_size]
    
    # All-Reduce汇聚
    output = all_reduce(output_parallel)
    # output: [batch, hidden_size]
    
    return output
```

#### 答案要点

**1. Worker完整生命周期**：

* 初始化：加载模型、初始化Cache Engine、设置分布式环境

* 推理执行：接收调度器输出、准备输入、执行模型、返回结果

* 清理：释放资源、关闭分布式连接

**2. TP vs PP对比**：

| 特性   | Tensor Parallelism | Pipeline Parallelism |
| ---- | ------------------ | -------------------- |
| 切分维度 | 层内切分               | 层间切分                 |
| 通信频率 | 高（每层多次）            | 低（层间一次）              |
| 通信量  | 大                  | 小                    |
| 气泡问题 | 无                  | 有                    |
| 适用场景 | 单机多卡               | 多机多卡                 |

**3. vLLM优先TP的原因**：

* LLM推理主要是计算密集型，TP能更好利用GPU算力

* TP无气泡问题，效率更高

* LLM模型宽度大，适合TP切分

* PP适合模型层数很多的场景

**4. All-Reduce原理**：

* 每个GPU持有部分结果

* 通过通信将所有部分结果求和

* 最终每个GPU都获得完整结果

* 在TP中用于汇聚并行计算的结果

**5. 代码解释**：

* 实现列并行线性层的前向传播

* 输入是完整hidden\_size，权重按列切分

* 每个GPU计算hidden\_size/tp\_size维度的输出

* 通过All-Reduce将所有GPU的结果求和

* 最终得到完整hidden\_size维度的输出

