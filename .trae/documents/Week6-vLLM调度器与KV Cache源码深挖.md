# Week 6: vLLM调度器与KV Cache源码深挖

## 本周目标
- 深入理解vLLM调度器的核心实现
- 掌握KV Cache的内存管理机制
- 理解Block Manager的工作原理
- 能够阅读并理解vLLM核心源码

---

## Day 1: vLLM调度器架构总览

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 阅读vLLM调度器相关代码 | 了解调度器入口 |
| 晚上 1.5h | 调度器架构分析 | 能画出调度流程图 |

### 学习材料

#### 源码位置
- `vllm/core/scheduler.py` - 调度器核心实现
- `vllm/core/schedulerOutput.py` - 调度器输出
- `vllm/sequence.py` - 序列管理

#### 核心类与数据结构
```python
class Scheduler:
    """调度器核心类"""
    def __init__(self, scheduler_config, cache_config):
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.block_manager = BlockManager(...)
        
        # 调度队列
        self.waiting: List[SequenceGroup] = []      # 等待队列
        self.running: List[SequenceGroup] = []      # 运行队列
        self.swapped: List[SequenceGroup] = []      # 换出队列
        
    def schedule(self) -> SchedulerOutput:
        """核心调度方法"""
        # 1. 处理swapped队列（优先恢复）
        # 2. 处理waiting队列（预emption）
        # 3. 处理running队列（继续执行）
        pass
```

### 核心知识点

#### 调度器架构图
```
┌─────────────────────────────────────────────────────────────┐
│                      Scheduler                               │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
│ │   Waiting   │ │   Running   │ │   Swapped   │             │
│ │   Queue     │ │   Queue     │ │   Queue     │             │
│ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘             │
│        │               │               │                     │
│        └───────────────┼───────────────┘                     │
│                        ▼                                     │
│              ┌─────────────────┐                             │
│              │  Block Manager  │                             │
│              │  (内存管理)      │                             │
│              └─────────────────┘                             │
│                        │                                     │
│                        ▼                                     │
│              ┌─────────────────┐                             │
│              │ SchedulerOutput │                             │
│              └─────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

#### 三大调度队列
| 队列 | 状态 | 说明 |
|------|------|------|
| **Waiting** | 等待调度 | 新到达的请求，等待资源分配 |
| **Running** | 正在执行 | 已分配资源，正在生成token |
| **Swapped** | 已换出 | 内存不足时换出到CPU内存 |

### 自测题

#### 问题
1. vLLM调度器的三大队列分别存储什么状态的序列？
2. 调度器的核心方法`schedule()`的执行流程是什么？
3. 为什么需要Swapped队列？什么情况下序列会被换出？

#### 答案
1. **三大队列**：
   - **Waiting队列**：存储新到达的请求，等待资源分配
   - **Running队列**：存储已分配资源、正在生成token的序列
   - **Swapped队列**：存储因内存不足被换出到CPU内存的序列

2. **schedule()执行流程**：
   - 首先处理Swapped队列，尝试恢复被抢占的序列
   - 然后处理Waiting队列，为新请求分配资源
   - 最后处理Running队列，继续执行正在运行的序列
   - 返回SchedulerOutput，包含本次调度的所有信息

3. **Swapped队列的作用**：
   - 当GPU内存不足时，需要暂停低优先级序列
   - 将其KV Cache换出到CPU内存，释放GPU资源
   - 等待资源充足时再恢复执行
   - 触发条件：GPU内存不足且无法为新请求分配Block

---

## Day 2: 调度策略深入

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 分析调度策略代码 | 理解FCFS策略 |
| 晚上 1.5h | 对比不同调度策略 | 理解各策略优劣 |

### 学习材料

#### 源码位置
- `vllm/core/policy.py` - 调度策略实现
- `vllm/core/scheduler.py` - 策略调用

#### 核心调度策略

##### 1. FCFS (First-Come-First-Served)
```python
class FCFSPolicy(Policy):
    """先来先服务策略"""
    def get_priority(self, seq_group: SequenceGroup) -> float:
        # 按到达时间排序
        return seq_group.arrival_time
```

##### 2. Priority Policy
```python
class PriorityPolicy(Policy):
    """优先级策略"""
    def get_priority(self, seq_group: SequenceGroup) -> float:
        # 按用户指定优先级排序
        return seq_group.priority
```

### 核心知识点

#### 调度决策流程
```
1. 检查Swapped队列
   ├── 有换出序列？
   │   ├── 能恢复？→ 恢复到Running
   │   └── 不能恢复？→ 继续等待
   
2. 处理Waiting队列
   ├── 有足够内存？
   │   ├── 是 → 分配Block，移到Running
   │   └── 否 → Preemption
   
3. 处理Running队列
   ├── 序列完成？→ 移除
   ├── 内存不足？→ Preemption
   └── 继续执行 → 生成下一个token
```

#### Preemption机制
- **原因**：内存不足时，需要暂停低优先级序列
- **方式**：
  1. **Swapping**：将KV Cache换出到CPU内存
  2. **Recomputation**：丢弃KV Cache，后续重新计算

```python
def _preempt(self, seq_group: SequenceGroup):
    """抢占机制"""
    if self.cache_config.swap_space > 0:
        # 有swap空间，执行swapping
        self._swap_out(seq_group)
    else:
        # 无swap空间，执行recomputation
        self._recompute(seq_group)
```

### 自测题

#### 问题
1. FCFS和Priority策略的区别是什么？
2. Preemption机制有哪两种方式？各自的优缺点？
3. 什么情况下会触发Preemption？

#### 答案
1. **FCFS vs Priority策略**：
   - **FCFS（先来先服务）**：按请求到达时间排序，先到达的请求优先处理
   - **Priority（优先级）**：按用户指定的优先级排序，高优先级请求优先处理
   - 区别：FCFS公平但无法区分请求重要性，Priority灵活但可能导致低优先级请求饥饿

2. **Preemption两种方式**：
   - **Swapping**：
     - 优点：恢复速度快（只需内存拷贝）
     - 缺点：需要额外的CPU内存空间
   - **Recomputation**：
     - 优点：无需额外内存
     - 缺点：恢复慢（需要重新计算Prefill）

3. **触发Preemption的条件**：
   - GPU内存不足，无法为新请求分配Block
   - Running队列中的序列需要更多内存
   - 有更高优先级的请求需要资源
   - 系统资源紧张，需要保证服务质量

---

## Day 3: Block Manager核心实现

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 阅读Block Manager代码 | 理解内存管理入口 |
| 晚上 1.5h | Block分配与回收机制 | 能描述完整流程 |

### 学习材料

#### 源码位置
- `vllm/core/block_manager.py` - Block Manager核心实现
- `vllm/core/block.py` - Block数据结构
- `vllm/core/block_table.py` - Block Table管理

#### 核心类与数据结构
```python
class BlockManager:
    """Block管理器"""
    def __init__(self, block_size, num_gpu_blocks, num_cpu_blocks):
        self.block_size = block_size
        
        # GPU和CPU Block池
        self.gpu_allocator = BlockAllocator(num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(num_cpu_blocks)
        
        # 每个序列的Block Table
        self.block_tables: Dict[int, BlockTable] = {}
        
    def allocate(self, seq_id: int) -> Block:
        """为序列分配新Block"""
        pass
        
    def free(self, seq_id: int):
        """释放序列的所有Block"""
        pass
```

### 核心知识点

#### Block Table结构
```
┌─────────────────────────────────────────────────────────┐
│                    Block Table                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Sequence 0: [Block 0, Block 2, Block 5, ...]       │ │
│ │               ↓        ↓        ↓                   │ │
│ │           Token 0-15  Token 16-31  Token 32-47      │ │
│ └─────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Sequence 1: [Block 1, Block 3, Block 4, ...]       │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

#### Block分配流程
```python
def allocate_block_for_sequence(self, seq_id: int) -> Block:
    """为序列分配Block的完整流程"""
    # 1. 从GPU Block池获取空闲Block
    block = self.gpu_allocator.allocate()
    
    # 2. 更新Block Table
    if seq_id not in self.block_tables:
        self.block_tables[seq_id] = BlockTable()
    self.block_tables[seq_id].append(block)
    
    # 3. 返回分配的Block
    return block
```

### 自测题

#### 问题
1. Block Manager如何管理GPU和CPU两级的Block池？
2. Block Table的作用是什么？如何实现序列到Block的映射？
3. Block分配和回收的具体流程是什么？

#### 答案
1. **Block Manager的内存管理**：
   - **GPU Block池**：管理GPU上的KV Cache Block，用于快速访问
   - **CPU Block池**：管理CPU上的swap空间，用于内存不足时的换出
   - **两级分配器**：分别维护GPU和CPU的空闲Block列表
   - **动态分配**：根据需求从相应池子分配Block

2. **Block Table的作用**：
   - **映射功能**：将逻辑上连续的序列映射到物理上离散的Block
   - **跟踪状态**：记录每个序列使用的所有Block索引
   - **支持变长**：通过Block列表支持任意长度的序列
   - **实现方式**：使用列表存储Block索引，按顺序映射token位置

3. **Block分配和回收流程**：
   - **分配流程**：
     1. 检查目标池子（GPU/CPU）是否有空闲Block
     2. 从空闲列表取出Block
     3. 更新Block Table，添加新Block索引
     4. 返回分配的Block
   - **回收流程**：
     1. 从Block Table中移除Block索引
     2. 将Block归还到相应池子的空闲列表
     3. 清空Block内容（可选）

---

## Day 4: KV Cache内存管理

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 分析KV Cache相关代码 | 理解Cache存储方式 |
| 晚上 1.5h | Cache优化技术 | 理解内存优化策略 |

### 学习材料

#### 源码位置
- `vllm/worker/cache_engine.py` - Cache Engine实现
- `vllm/attention.py` - Attention中的Cache使用

#### KV Cache数据结构
```python
class CacheEngine:
    """KV Cache引擎"""
    def __init__(self, cache_config, model_config):
        # GPU KV Cache
        self.gpu_cache: List[torch.Tensor] = []
        # shape: [num_layers, 2, num_blocks, block_size, num_heads, head_dim]
        # 2表示Key和Value
        
        # CPU KV Cache (用于swap)
        self.cpu_cache: List[torch.Tensor] = []
        
    def allocate_cache(self):
        """预分配KV Cache内存"""
        for _ in range(self.num_layers):
            gpu_cache = torch.zeros(
                self.num_gpu_blocks,
                self.block_size,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                device='cuda'
            )
            self.gpu_cache.append(gpu_cache)
```

### 核心知识点

#### KV Cache内存布局
```
┌─────────────────────────────────────────────────────────┐
│                   KV Cache 内存布局                      │
│                                                          │
│  Layer 0: ┌──────────────────────────────────────────┐  │
│           │ Key Cache: [num_blocks, block_size, ...] │  │
│           │ Value Cache: [num_blocks, block_size, ...]│  │
│           └──────────────────────────────────────────┘  │
│                                                          │
│  Layer 1: ┌──────────────────────────────────────────┐  │
│           │ Key Cache: [num_blocks, block_size, ...] │  │
│           │ Value Cache: [num_blocks, block_size, ...]│  │
│           └──────────────────────────────────────────┘  │
│                         ...                              │
│                                                          │
│  Layer N-1: ┌────────────────────────────────────────┐  │
│             │ Key Cache: [num_blocks, block_size, ...]│  │
│             │ Value Cache: [num_blocks, block_size, ...]│  │
│             └────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### Cache优化技术
1. **Paged Attention**：分页管理，减少内存碎片
2. **Continuous Batching**：动态批处理，提高吞吐
3. **Prefix Caching**：前缀缓存复用，减少计算

### 自测题

#### 问题
1. KV Cache的内存布局是怎样的？为什么要分层存储？
2. Paged Attention如何减少内存碎片？
3. Continuous Batching的原理是什么？

#### 答案
1. **KV Cache内存布局**：
   - **分层结构**：`[num_layers, 2, num_blocks, block_size, num_heads, head_dim]`
   - **每层独立**：每个Transformer层有自己的KV Cache
   - **Key-Value分离**：每层包含Key和Value两个张量
   - **原因**：便于并行计算、支持流水线、减少跨层依赖

2. **Paged Attention减少内存碎片**：
   - **固定大小Block**：使用固定大小的Block（如16/32）存储KV
   - **按需分配**：只在需要时分配Block，避免预分配浪费
   - **Block复用**：序列完成后Block可回收复用
   - **非连续存储**：逻辑连续的序列可映射到物理离散的Block

3. **Continuous Batching原理**：
   - **动态批处理**：新请求可随时加入当前批次
   - **序列完成即释放**：完成的序列立即释放资源
   - **高效利用GPU**：保持GPU持续忙碌，提高吞吐量
   - **减少等待**：新请求无需等待整个批次完成

---

## Day 5: Swapping与Recomputation

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 分析swap相关代码 | 理解换入换出机制 |
| 晚上 1.5h | Recomputation实现 | 理解重计算策略 |

### 学习材料

#### 源码位置
- `vllm/core/scheduler.py` - Swapping逻辑
- `vllm/worker/cache_engine.py` - Cache swap实现

#### Swapping实现
```python
def swap_out(self, seq_group: SequenceGroup) -> None:
    """将序列的KV Cache换出到CPU"""
    for seq in seq_group.get_seqs():
        block_table = self.block_manager.get_block_table(seq.seq_id)
        
        # 将GPU Block内容拷贝到CPU Block
        for gpu_block, cpu_block in zip(block_table.gpu_blocks, 
                                          block_table.cpu_blocks):
            self.cache_engine.swap(gpu_block, cpu_block, 
                                   src_device='cuda', 
                                   dst_device='cpu')
        
        # 释放GPU Block
        self.block_manager.free_gpu_blocks(seq.seq_id)

def swap_in(self, seq_group: SequenceGroup) -> None:
    """将序列的KV Cache换入到GPU"""
    for seq in seq_group.get_seqs():
        # 分配GPU Block
        self.block_manager.allocate_gpu_blocks(seq.seq_id)
        
        # 将CPU Block内容拷贝到GPU Block
        block_table = self.block_manager.get_block_table(seq.seq_id)
        for gpu_block, cpu_block in zip(block_table.gpu_blocks,
                                          block_table.cpu_blocks):
            self.cache_engine.swap(cpu_block, gpu_block,
                                   src_device='cpu',
                                   dst_device='cuda')
```

### 核心知识点

#### Swapping vs Recomputation对比
| 特性 | Swapping | Recomputation |
|------|----------|---------------|
| **内存开销** | 需要CPU内存 | 无额外内存 |
| **恢复速度** | 快（内存拷贝） | 慢（重新计算） |
| **计算开销** | 低 | 高 |
| **适用场景** | 有足够CPU内存 | CPU内存受限 |

#### Recomputation流程
```
1. 记录序列的原始输入tokens
2. 当需要preempt时，直接丢弃KV Cache
3. 当序列恢复时，重新执行Prefill计算
4. 重新生成KV Cache
```

### 自测题

#### 问题
1. Swapping和Recomputation的区别是什么？各自的适用场景？
2. Swapping的实现流程是怎样的？
3. 为什么vLLM默认使用Recomputation而不是Swapping？

#### 答案
1. **Swapping vs Recomputation区别**：
   - **Swapping**：
     - 原理：将KV Cache换出到CPU内存
     - 优点：恢复速度快（只需内存拷贝）
     - 缺点：需要额外的CPU内存空间
     - 适用场景：CPU内存充足时
   - **Recomputation**：
     - 原理：丢弃KV Cache，后续重新计算
     - 优点：无需额外内存
     - 缺点：恢复慢（需要重新计算Prefill）
     - 适用场景：CPU内存受限时

2. **Swapping实现流程**：
   - 为序列分配CPU Block
   - 将GPU Block内容拷贝到CPU Block
   - 释放GPU Block
   - 更新Block Table
   - 序列状态标记为Swapped

3. **默认使用Recomputation的原因**：
   - 大多数部署环境CPU内存有限
   - Recomputation无需额外硬件资源
   - 实现更简单，维护成本低
   - 虽然恢复慢，但Preemption发生频率相对较低
   - 可以通过调优减少Preemption触发

---

## Day 6: 实战 - 调度器源码走读

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 晚上 2h | 完整走读调度器源码 | 能解释关键代码逻辑 |

### 实战要求

#### 任务描述
完整阅读`vllm/core/scheduler.py`，重点理解：

1. **schedule()方法**：核心调度逻辑
```python
def schedule(self) -> SchedulerOutput:
    """核心调度方法 - 需要重点理解"""
    # TODO: 阅读并注释关键步骤
    pass
```

2. **_schedule_running()方法**：处理运行队列
3. **_schedule_swapped()方法**：处理换出队列
4. **_schedule_waiting()方法**：处理等待队列

#### 验证标准
- [ ] 能画出完整的调度流程图
- [ ] 能解释每个队列的处理逻辑
- [ ] 能说明Preemption的触发条件

---

## Day 7: 本周复盘 + 自测

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 晚上 1h | 本周复盘 + 自测 | 检验学习效果 |

### 本周知识点回顾

#### 知识点清单
- [ ] vLLM调度器架构与三大队列
- [ ] FCFS/Priority调度策略
- [ ] Preemption机制（Swapping vs Recomputation）
- [ ] Block Manager内存管理
- [ ] KV Cache内存布局与优化
- [ ] Swapping与Recomputation实现

### 综合自测题

#### 问题
1. 描述vLLM调度器的完整工作流程，从请求到达到token生成。
2. Block Manager如何实现高效的内存管理？与传统的连续内存分配相比有什么优势？
3. 当系统内存不足时，vLLM如何处理？请详细说明两种Preemption策略。
4. KV Cache为什么采用分层存储？这种设计有什么优势？
5. 阅读以下代码，解释其作用：
```python
def _schedule_swapped(self):
    # 被抢占的序列优先恢复
    ret: SchedulerOutput = SchedulerOutput(...)
    
    while self.swapped:
        seq_group = self.swapped[0]
        
        # 检查是否能恢复
        num_new_tokens = self._get_num_new_tokens(seq_group)
        can_allocate = self.block_manager.can_allocate(seq_group)
        
        if not can_allocate:
            break
            
        # 恢复序列
        self._swap_in(seq_group)
        self.swapped.pop(0)
        self.running.append(seq_group)
        
    return ret
```

#### 答案要点

**1. 调度器完整工作流程**：
- 请求到达 → 进入Waiting队列
- 调度器检查资源 → 分配Block → 移到Running队列
- Running序列生成token → 更新KV Cache
- 内存不足时 → Preemption → 移到Swapped队列
- 序列完成 → 移除并释放资源

**2. Block Manager优势**：
- 分页管理，减少内存碎片
- 按需分配，提高内存利用率
- 支持变长序列，无需预分配固定大小
- Block可跨序列复用

**3. Preemption策略**：
- **Swapping**：换出到CPU内存，恢复快但需要额外内存
- **Recomputation**：丢弃KV Cache，重新计算，无需额外内存但计算开销大
- 选择依据：CPU内存是否充足

**4. KV Cache分层存储优势**：
- 每层独立管理，便于并行计算
- 支持流水线并行
- 便于实现prefix caching等优化

**5. 代码解释**：
- 该方法处理Swapped队列中的被抢占序列
- 优先恢复被抢占的序列（先入先出）
- 检查是否有足够资源恢复
- 能恢复则执行swap_in，移到Running队列
- 资源不足则停止恢复，等待后续调度