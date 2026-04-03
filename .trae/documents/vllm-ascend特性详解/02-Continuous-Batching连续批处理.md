# Continuous Batching (连续批处理) 详解

## 1. 特性概述

### 1.1 什么是Continuous Batching？

Continuous Batching（连续批处理）是vLLM的核心调度策略，它允许在迭代级别动态调整batch，实现请求的动态插入和移除，从而最大化GPU/NPU利用率。

### 1.2 传统批处理的问题

**静态批处理**：
```
时间轴:
T0: [Request 1 (prefill)] [Request 2 (prefill)] [Request 3 (prefill)]
T1: [Request 1 (decode)]  [Request 2 (decode)]  [Request 3 (decode)]
T2: [Request 1 (decode)]  [Request 2 (decode)]  [Request 3 (decode)]
T3: [Request 1 (decode)]  [Request 2 (decode)]  [Request 3 (decode)]
T4: [Request 1 (decode)]  [Request 2 (decode)]  [Request 3 (decode)]
T5: [IDLE]                [IDLE]                [Request 3 (decode)]
T6: [IDLE]                [IDLE]                [Request 3 (decode)]
```

问题：
- Request 1和2完成后，资源闲置
- 无法动态插入新请求
- 吞吐量低

### 1.3 Continuous Batching解决方案

**连续批处理**：
```
时间轴:
T0: [Request 1 (prefill)] [Request 2 (prefill)] [Request 3 (prefill)]
T1: [Request 1 (decode)]  [Request 2 (decode)]  [Request 3 (decode)]
T2: [Request 1 (decode)]  [Request 2 (decode)]  [Request 3 (decode)]
T3: [Request 1 (decode)]  [Request 2 (decode)]  [Request 3 (decode)]
T4: [Request 4 (prefill)] [Request 5 (prefill)] [Request 3 (decode)]  # Request 1,2完成，插入新请求
T5: [Request 4 (decode)]  [Request 5 (decode)]  [Request 3 (decode)]
T6: [Request 4 (decode)]  [Request 5 (decode)]  [Request 6 (prefill)]  # Request 3完成，插入新请求
```

优势：
- 动态插入新请求
- 资源利用率高
- 吞吐量提升

## 2. 设计方案

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│              Continuous Batching架构                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Request Queue                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Waiting Queue (等待队列)                      │   │
│  │  [Req1] [Req2] [Req3] [Req4] [Req5] ...       │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  Scheduler                                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  1. 检查资源 (显存、计算能力)                  │   │
│  │  2. 选择请求 (优先级、资源需求)                │   │
│  │  3. 构建Batch (prefill + decode混合)          │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  Running Batch                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Running Queue (运行队列)                      │   │
│  │  [Req1 (decode)] [Req2 (decode)] [Req3 (prefill)]│  │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  Output Processing                                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  1. 检查完成条件 (EOS, max_tokens)            │   │
│  │  2. 移除完成的请求                            │   │
│  │  3. 返回结果                                  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 调度策略

#### 2.2.1 迭代级调度
```python
class IterationLevelScheduler:
    """
    迭代级调度器
    
    每次迭代（生成一个token）后重新调度
    """
    
    def schedule_iteration(self):
        """
        单次迭代的调度流程
        """
        # 1. 移除已完成的请求
        self.running = [req for req in self.running if not req.is_finished()]
        
        # 2. 计算可用资源
        available_blocks = self.block_manager.get_num_free_blocks()
        
        # 3. 从waiting队列选择新请求
        while available_blocks > 0 and self.waiting:
            # 选择优先级最高的请求
            req = self.waiting.pop(0)
            
            # 检查资源是否足够
            required_blocks = self._estimate_blocks(req)
            if available_blocks >= required_blocks:
                self.running.append(req)
                available_blocks -= required_blocks
            else:
                # 资源不足，放回队列
                self.waiting.insert(0, req)
                break
        
        # 4. 返回当前batch
        return self.running
```

#### 2.2.2 Prefill-Decode混合
```python
class PrefillDecodeMixer:
    """
    Prefill和Decode混合调度
    
    关键：平衡prefill和decode的资源占用
    """
    
    def __init__(self, prefill_ratio=0.3):
        self.prefill_ratio = prefill_ratio  # prefill请求占比
    
    def build_mixed_batch(self, running: List[Request]):
        """
        构建prefill和decode混合的batch
        
        策略：
        1. 优先调度decode请求（延迟敏感）
        2. 在资源允许时插入prefill请求
        3. 控制prefill请求数量，避免阻塞decode
        """
        decode_requests = [r for r in running if r.is_decode_phase()]
        prefill_requests = [r for r in running if r.is_prefill_phase()]
        
        # 限制prefill数量
        max_prefill = int(len(decode_requests) * self.prefill_ratio)
        selected_prefill = prefill_requests[:max_prefill]
        
        # 组合batch
        batch = decode_requests + selected_prefill
        
        return batch
```

### 2.3 Chunked Prefill

```python
class ChunkedPrefillScheduler:
    """
    Chunked Prefill调度器
    
    将长prefill切分为多个chunk，与decode交替执行
    """
    
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size  # 每个chunk的token数
    
    def schedule_chunked_prefill(self, prefill_req: Request):
        """
        将prefill请求切分为chunks
        
        优势：
        1. 避免长prefill阻塞decode
        2. 提高响应速度
        3. 更好的资源利用
        """
        total_tokens = len(prefill_req.prompt_tokens)
        chunks = []
        
        for start in range(0, total_tokens, self.chunk_size):
            end = min(start + self.chunk_size, total_tokens)
            chunk = prefill_req.prompt_tokens[start:end]
            chunks.append(chunk)
        
        return chunks
    
    def execute_chunked_prefill(self, chunks: List[List[int]]):
        """
        执行chunked prefill
        
        流程：
        1. 执行一个chunk的prefill
        2. 执行一轮decode
        3. 重复直到所有chunk完成
        """
        for i, chunk in enumerate(chunks):
            # 执行prefill chunk
            prefill_output = self.model.prefill(chunk)
            
            # 如果不是最后一个chunk，执行decode
            if i < len(chunks) - 1:
                decode_output = self.model.decode(self.running_decode_requests)
```

## 3. 关键代码解读

### 3.1 Scheduler核心实现

```python
class Scheduler:
    """
    vLLM Scheduler核心实现
    """
    
    def __init__(self, 
                 block_manager: BlockManager,
                 max_num_seqs: int = 256,
                 max_num_batched_tokens: int = 8192):
        self.block_manager = block_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        
        # 请求队列
        self.waiting: Deque[SequenceGroup] = deque()
        self.running: List[SequenceGroup] = []
        self.swapped: List[SequenceGroup] = []
        
        # 调度策略
        self.policy = Policy.FIFO  # FIFO or PRIORITY
    
    def add_request(self, request: SequenceGroup):
        """添加新请求到waiting队列"""
        self.waiting.append(request)
    
    def schedule(self) -> SchedulerOutputs:
        """
        核心调度函数
        
        Returns:
            SchedulerOutputs: 包含要执行的序列和操作
        """
        # 1. 检查swapped队列（优先恢复）
        if self.swapped:
            self._try_swap_in()
        
        # 2. 移除已完成的序列
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]
        
        # 3. 从waiting队列调度新请求
        self._schedule_waiting()
        
        # 4. 构建输出
        return self._build_scheduler_outputs()
    
    def _schedule_waiting(self):
        """从waiting队列调度请求"""
        # 计算当前batch的token数
        curr_num_batched_tokens = sum(
            seq_group.get_num_seqs()
            for seq_group in self.running
        )
        
        while self.waiting:
            # 检查序列数限制
            if len(self.running) >= self.max_num_seqs:
                break
            
            # 获取下一个请求
            seq_group = self.waiting[0]
            
            # 检查token数限制
            num_new_tokens = seq_group.get_num_tokens()
            if curr_num_batched_tokens + num_new_tokens > self.max_num_batched_tokens:
                break
            
            # 检查Block资源
            num_required_blocks = self._get_num_required_blocks(seq_group)
            if self.block_manager.can_allocate(num_required_blocks):
                # 分配Block
                self.block_manager.allocate(seq_group)
                
                # 移动到running队列
                self.waiting.popleft()
                self.running.append(seq_group)
                curr_num_batched_tokens += num_new_tokens
            else:
                # 资源不足，尝试preempt
                if self._preempt(seq_group):
                    continue
                else:
                    # 无法preempt，停止调度
                    break
    
    def _preempt(self, seq_group: SequenceGroup) -> bool:
        """
        抢占策略
        
        策略：
        1. Swap: 将低优先级序列swap到CPU
        2. Recomputation: 释放序列，后续重新计算
        """
        # 找到最低优先级的running序列
        victim = self._find_preemption_victim()
        
        if victim is None:
            return False
        
        # 尝试swap
        if self.block_manager.can_swap_in():
            self.block_manager.swap_out(victim)
            self.running.remove(victim)
            self.swapped.append(victim)
            return True
        else:
            # Recomputation
            self.block_manager.free(victim)
            self.running.remove(victim)
            self.waiting.appendleft(victim)
            return True
    
    def _build_scheduler_outputs(self) -> SchedulerOutputs:
        """构建调度输出"""
        # 分离prefill和decode
        prefill_seq_groups = []
        decode_seq_groups = []
        
        for seq_group in self.running:
            if seq_group.is_prefill():
                prefill_seq_groups.append(seq_group)
            else:
                decode_seq_groups.append(seq_group)
        
        return SchedulerOutputs(
            prefill_seq_groups=prefill_seq_groups,
            decode_seq_groups=decode_seq_groups,
            blocks_to_swap_in=self.blocks_to_swap_in,
            blocks_to_swap_out=self.blocks_to_swap_out,
        )
```

### 3.2 NPU优化的Continuous Batching

```python
class NPUContinuousBatching:
    """
    针对Ascend NPU优化的Continuous Batching实现
    """
    
    def __init__(self, model, block_manager):
        self.model = model
        self.block_manager = block_manager
        
        # NPU特有优化
        self.npu_stream = torch.npu.Stream()
        self.async_executor = AsyncExecutor()
    
    def execute_batch(self, scheduler_outputs: SchedulerOutputs):
        """
        执行batch推理
        
        NPU优化点：
        1. 使用NPU Stream异步执行
        2. 批量处理prefill和decode
        3. 算子融合
        """
        # 1. 准备输入
        prefill_inputs = self._prepare_prefill_inputs(
            scheduler_outputs.prefill_seq_groups
        )
        decode_inputs = self._prepare_decode_inputs(
            scheduler_outputs.decode_seq_groups
        )
        
        # 2. 使用NPU Stream异步执行
        with torch.npu.stream(self.npu_stream):
            # 执行prefill
            if prefill_inputs:
                prefill_outputs = self._execute_prefill(prefill_inputs)
            
            # 执行decode
            if decode_inputs:
                decode_outputs = self._execute_decode(decode_inputs)
        
        # 3. 同步结果
        torch.npu.synchronize()
        
        return prefill_outputs, decode_outputs
    
    def _execute_prefill(self, inputs):
        """
        执行prefill
        
        NPU优化：
        1. 使用npu_fusion_attention
        2. 批量处理多个prefill请求
        """
        # 合并多个prefill请求
        # [num_prefill_seqs, max_seq_len, hidden_dim]
        combined_input = torch.cat([inp['input_ids'] for inp in inputs], dim=0)
        
        # 使用NPU融合算子
        hidden_states = self.model.embed(combined_input)
        
        # 使用npu_fusion_attention
        attention_output = torch_npu.npu_fusion_attention(
            hidden_states, hidden_states, hidden_states,
            head_num=self.model.config.num_attention_heads,
            input_layout="BSND",
            scale=1.0 / math.sqrt(self.model.config.head_dim)
        )[0]
        
        # FFN层
        output = self.model.ffn(attention_output)
        
        return output
    
    def _execute_decode(self, inputs):
        """
        执行decode
        
        NPU优化：
        1. 使用PagedAttention
        2. 批量decode
        """
        # 收集所有decode请求的最后一个token
        # [num_decode_seqs, 1, hidden_dim]
        last_tokens = torch.cat([inp['last_token'] for inp in inputs], dim=0)
        
        # Embedding
        hidden_states = self.model.embed(last_tokens)
        
        # 使用PagedAttention
        for layer in self.model.layers:
            # Attention with KV Cache
            hidden_states = layer.attention(
                hidden_states,
                kv_cache=inputs[0]['kv_cache'],  # 使用共享的KV Cache
                block_tables=inputs[0]['block_tables']
            )
            
            # FFN
            hidden_states = layer.ffn(hidden_states)
        
        # Logits
        logits = self.model.lm_head(hidden_states)
        
        return logits
```

### 3.3 Chunked Prefill实现

```python
class ChunkedPrefillExecutor:
    """
    Chunked Prefill执行器
    
    将长prefill切分为chunks，与decode交替执行
    """
    
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size
    
    def execute(self, prefill_requests, decode_requests):
        """
        执行chunked prefill + decode
        
        流程：
        1. 执行一个prefill chunk
        2. 执行一轮decode
        3. 重复
        """
        # 切分prefill为chunks
        prefill_chunks = []
        for req in prefill_requests:
            chunks = self._split_into_chunks(req)
            prefill_chunks.extend(chunks)
        
        # 交替执行
        results = []
        chunk_idx = 0
        
        while chunk_idx < len(prefill_chunks) or decode_requests:
            # 执行prefill chunk
            if chunk_idx < len(prefill_chunks):
                chunk = prefill_chunks[chunk_idx]
                prefill_output = self._execute_prefill_chunk(chunk)
                results.append(prefill_output)
                chunk_idx += 1
            
            # 执行decode
            if decode_requests:
                decode_output = self._execute_decode(decode_requests)
                results.append(decode_output)
        
        return results
    
    def _split_into_chunks(self, request):
        """将请求切分为chunks"""
        tokens = request.prompt_tokens
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i+self.chunk_size]
            chunks.append({
                'tokens': chunk_tokens,
                'request_id': request.request_id,
                'is_last_chunk': (i + self.chunk_size >= len(tokens))
            })
        
        return chunks
```

## 4. 性能优化建议

### 4.1 Batch Size调优

```python
def get_optimal_batch_size(model_config, hardware_config):
    """
    Batch Size优化指南
    
    考虑因素：
    1. 显存容量
    2. 序列长度分布
    3. NPU计算能力
    """
    # 最大序列数
    max_num_seqs = min(
        256,  # 经验值
        hardware_config.memory_gb * 1024 // (model_config.hidden_size * 4)
    )
    
    # 最大batched tokens
    max_num_batched_tokens = min(
        8192,  # 经验值
        hardware_config.memory_gb * 1024 * 1024 // model_config.hidden_size
    )
    
    return max_num_seqs, max_num_batched_tokens
```

### 4.2 Prefill-Decode比例

```python
def get_optimal_prefill_ratio(request_distribution):
    """
    Prefill-Decode比例优化
    
    场景：
    1. 长序列为主：降低prefill_ratio (0.1-0.2)
    2. 短序列为主：提高prefill_ratio (0.3-0.4)
    3. 混合场景：中等prefill_ratio (0.2-0.3)
    """
    avg_seq_len = request_distribution.avg_seq_len
    
    if avg_seq_len > 2048:
        return 0.1
    elif avg_seq_len < 512:
        return 0.4
    else:
        return 0.2
```

### 4.3 Chunk Size选择

```python
def get_optimal_chunk_size(model_config):
    """
    Chunk Size优化
    
    原则：
    1. 平衡prefill和decode延迟
    2. 避免过小的chunk（开销大）
    3. 避免过大的chunk（阻塞decode）
    """
    # 根据模型大小选择
    if model_config.hidden_size <= 4096:
        return 512
    elif model_config.hidden_size <= 8192:
        return 256
    else:
        return 128
```

## 5. 实际应用案例

### 5.1 高并发场景

```python
# 高并发场景配置
config = SchedulerConfig(
    max_num_seqs=256,              # 最大并发序列数
    max_num_batched_tokens=16384,  # 最大batched tokens
    prefill_ratio=0.2,             # prefill占比
    chunk_size=512,                # chunk大小
)

# 启动服务
server = vLLMServer(
    model="Qwen/Qwen-72B",
    scheduler_config=config,
    tensor_parallel_size=8,  # 8卡TP
)

# 性能指标
# - 吞吐量: 2000+ tokens/s
# - 平均延迟: < 100ms
# - P99延迟: < 500ms
```

### 5.2 长序列场景

```python
# 长序列场景配置
config = SchedulerConfig(
    max_num_seqs=64,               # 降低并发数
    max_num_batched_tokens=32768,  # 提高token数
    prefill_ratio=0.1,             # 降低prefill占比
    chunk_size=256,                # 减小chunk
)

# 启动服务
server = vLLMServer(
    model="Qwen/Qwen-72B",
    scheduler_config=config,
    max_model_len=32768,  # 支持32K上下文
)

# 性能指标
# - 支持32K上下文
# - 吞吐量: 500+ tokens/s
# - 首token延迟: < 2s
```

## 6. 性能监控

### 6.1 关键指标

```python
class ContinuousBatchingMetrics:
    """性能监控指标"""
    
    # 吞吐量
    throughput = 0.0  # tokens/s
    
    # 延迟
    avg_latency = 0.0  # 平均延迟
    p50_latency = 0.0
    p99_latency = 0.0
    
    # 资源利用率
    gpu_utilization = 0.0
    memory_utilization = 0.0
    
    # 队列状态
    waiting_queue_size = 0
    running_queue_size = 0
    swapped_queue_size = 0
    
    # Batch统计
    avg_batch_size = 0.0
    avg_num_batched_tokens = 0.0
    
    def report(self):
        print(f"吞吐量: {self.throughput:.2f} tokens/s")
        print(f"平均延迟: {self.avg_latency:.2f} ms")
        print(f"P99延迟: {self.p99_latency:.2f} ms")
        print(f"GPU利用率: {self.gpu_utilization:.2%}")
        print(f"内存利用率: {self.memory_utilization:.2%}")
        print(f"平均Batch Size: {self.avg_batch_size:.2f}")
```

## 7. 总结

Continuous Batching通过迭代级调度和动态batch管理，实现了：

1. **高吞吐量**：动态插入新请求，最大化资源利用
2. **低延迟**：优先调度decode请求，减少等待时间
3. **灵活性**：支持不同长度和优先级的请求
4. **可扩展性**：支持大规模并发请求

在Ascend NPU上，通过使用NPU Stream、融合算子等技术，进一步提升了Continuous Batching的性能。

---

## 8. Continuous Batching vs Chunked Prefill 深度对比

### 8.1 核心差异分析

#### 8.1.1 调度策略差异

```
┌─────────────────────────────────────────────────────────────────┐
│                    调度策略核心差异                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【Continuous Batching】                                         │
│  ┌──────────────────────────────────────────────────┐           │
│  │ 调度粒度: 迭代级 (每次生成一个token后重新调度)    │           │
│  │ 核心机制: 动态插入/移除请求                       │           │
│  │ 优先级: decode > prefill                         │           │
│  │ 适用场景: 高并发、请求长度差异大                  │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  【Chunked Prefill】                                             │
│  ┌──────────────────────────────────────────────────┐           │
│  │ 调度粒度: 分块级 (将长序列切分为多个chunk)        │           │
│  │ 核心机制: prefill分块 + decode交替执行            │           │
│  │ 优先级: decode > prefill_chunk                   │           │
│  │ 适用场景: 长序列、避免长prefill阻塞               │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 8.1.2 内存管理差异

```
┌─────────────────────────────────────────────────────────────────┐
│                    内存管理策略对比                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【Continuous Batching】                                         │
│  ┌──────────────────────────────────────────────────┐           │
│  │ KV Cache分配: 请求开始时一次性分配                │           │
│  │ 内存释放: 请求完成后立即释放                      │           │
│  │ 内存峰值: 取决于最大并发请求数                    │           │
│  │ 内存利用率: 高 (动态管理)                         │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  【Chunked Prefill】                                             │
│  ┌──────────────────────────────────────────────────┐           │
│  │ KV Cache分配: 每个chunk逐步分配                   │           │
│  │ 内存释放: 所有chunk完成后释放                     │           │
│  │ 内存峰值: 较低 (分块处理)                         │           │
│  │ 内存利用率: 中高 (需要chunk管理开销)              │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 性能对比分析

#### 8.2.1 吞吐量对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    吞吐量对比 (tokens/s)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  场景1: 高并发短序列 (100并发, 平均长度512)                      │
│  Static Batching:     ████████░░░░░░░░  800                    │
│  Continuous Batching: ████████████████  1600  ← 提升2倍         │
│  Chunked Prefill:     ████████████░░░░  1200                   │
│                                                                 │
│  场景2: 中等并发长序列 (50并发, 平均长度2048)                    │
│  Static Batching:     ████░░░░░░░░░░░░  400                    │
│  Continuous Batching: ████████░░░░░░░░  800                    │
│  Chunked Prefill:     ████████████████  1200  ← 最优           │
│                                                                 │
│  场景3: 低并发超长序列 (20并发, 平均长度8192)                    │
│  Static Batching:     ██░░░░░░░░░░░░░░  200                    │
│  Continuous Batching: ████░░░░░░░░░░░░  400                    │
│  Chunked Prefill:     ████████████░░░░  1000  ← 显著优势       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 8.2.2 延迟对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    延迟对比 (ms)                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【首Token延迟 (TTFT)】                                          │
│  Static Batching:     ████████████████  2000                   │
│  Continuous Batching: ████████░░░░░░░░  1000                   │
│  Chunked Prefill:     ████░░░░░░░░░░░░  500   ← 最优           │
│                                                                 │
│  【平均延迟】                                                    │
│  Static Batching:     ████████████████  200                    │
│  Continuous Batching: ████████░░░░░░░░  100                    │
│  Chunked Prefill:     ██████░░░░░░░░░░  80                    │
│                                                                 │
│  【P99延迟】                                                     │
│  Static Batching:     ████████████████  500                    │
│  Continuous Batching: ████████████░░░░  300                    │
│  Chunked Prefill:     ████████████░░░░  350                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 适用场景分析

```
┌─────────────────────────────────────────────────────────────────┐
│                    适用场景对比                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【Continuous Batching 最佳场景】                                │
│  ┌──────────────────────────────────────────────────┐           │
│  │ ✓ 高并发在线服务 (100+ QPS)                      │           │
│  │ ✓ 请求长度差异大 (128-2048 tokens)               │           │
│  │ ✓ 对吞吐量要求高                                 │           │
│  │ ✓ 对延迟敏感 (需要低P99延迟)                     │           │
│  │ ✗ 长序列场景 (>4096 tokens)                      │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  【Chunked Prefill 最佳场景】                                    │
│  ┌──────────────────────────────────────────────────┐           │
│  │ ✓ 长序列推理 (>2048 tokens)                      │           │
│  │ ✓ 对首Token延迟敏感 (需要低TTFT)                 │           │
│  │ ✓ 混合场景 (长短序列并存)                        │           │
│  │ ✓ 内存受限场景                                   │           │
│  │ ✗ 纯短序列场景 (<512 tokens)                     │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.4 组合使用策略

```python
class HybridScheduler:
    """
    混合调度器：结合Continuous Batching和Chunked Prefill
    
    策略：
    1. 短序列请求：使用Continuous Batching
    2. 长序列请求：使用Chunked Prefill
    3. 动态切换：根据请求长度自动选择策略
    """
    
    def __init__(self, 
                 long_seq_threshold=2048,
                 chunk_size=512):
        self.long_seq_threshold = long_seq_threshold
        self.chunk_size = chunk_size
        
        # 两个调度器
        self.continuous_batching_scheduler = ContinuousBatchingScheduler()
        self.chunked_prefill_scheduler = ChunkedPrefillScheduler(chunk_size)
    
    def schedule(self, requests: List[Request]):
        """
        混合调度
        
        自动识别请求长度，选择最优调度策略
        """
        # 分类请求
        short_requests = [r for r in requests if len(r.tokens) < self.long_seq_threshold]
        long_requests = [r for r in requests if len(r.tokens) >= self.long_seq_threshold]
        
        # 分别调度
        short_batch = self.continuous_batching_scheduler.schedule(short_requests)
        long_batch = self.chunked_prefill_scheduler.schedule(long_requests)
        
        # 合并batch
        combined_batch = self._merge_batches(short_batch, long_batch)
        
        return combined_batch
    
    def _merge_batches(self, short_batch, long_batch):
        """
        合并两个batch
        
        策略：
        1. 优先执行decode请求
        2. 穿插执行short prefill和long prefill chunks
        """
        merged = []
        
        # 1. 所有decode请求
        merged.extend(short_batch.decode_requests)
        merged.extend(long_batch.decode_requests)
        
        # 2. 交替执行prefill
        short_prefill_idx = 0
        long_prefill_idx = 0
        
        while (short_prefill_idx < len(short_batch.prefill_requests) or
               long_prefill_idx < len(long_batch.prefill_chunks)):
            
            # 添加一个short prefill
            if short_prefill_idx < len(short_batch.prefill_requests):
                merged.append(short_batch.prefill_requests[short_prefill_idx])
                short_prefill_idx += 1
            
            # 添加一个long prefill chunk
            if long_prefill_idx < len(long_batch.prefill_chunks):
                merged.append(long_batch.prefill_chunks[long_prefill_idx])
                long_prefill_idx += 1
        
        return merged
```

### 8.5 性能调优建议

#### 8.5.1 Continuous Batching调优

```python
def tune_continuous_batching(workload_characteristics):
    """
    Continuous Batching调优指南
    
    参数：
    - max_num_seqs: 最大并发序列数
    - max_num_batched_tokens: 最大batched tokens
    - prefill_ratio: prefill请求占比
    """
    
    # 场景1: 高并发短序列
    if workload_characteristics['concurrency'] > 100 and \
       workload_characteristics['avg_seq_len'] < 512:
        return {
            'max_num_seqs': 256,
            'max_num_batched_tokens': 16384,
            'prefill_ratio': 0.3,
        }
    
    # 场景2: 中等并发中等序列
    elif workload_characteristics['concurrency'] > 50 and \
         workload_characteristics['avg_seq_len'] < 2048:
        return {
            'max_num_seqs': 128,
            'max_num_batched_tokens': 8192,
            'prefill_ratio': 0.2,
        }
    
    # 场景3: 低并发长序列
    else:
        return {
            'max_num_seqs': 64,
            'max_num_batched_tokens': 4096,
            'prefill_ratio': 0.1,
        }
```

#### 8.5.2 Chunked Prefill调优

```python
def tune_chunked_prefill(workload_characteristics):
    """
    Chunked Prefill调优指南
    
    参数：
    - chunk_size: 每个chunk的token数
    - max_chunks_per_iteration: 每次迭代最多执行的chunk数
    - decode_priority: decode请求优先级权重
    """
    
    # 场景1: 超长序列 (>8K)
    if workload_characteristics['avg_seq_len'] > 8192:
        return {
            'chunk_size': 256,
            'max_chunks_per_iteration': 2,
            'decode_priority': 0.8,
        }
    
    # 场景2: 长序列 (2K-8K)
    elif workload_characteristics['avg_seq_len'] > 2048:
        return {
            'chunk_size': 512,
            'max_chunks_per_iteration': 3,
            'decode_priority': 0.7,
        }
    
    # 场景3: 中等序列 (512-2K)
    else:
        return {
            'chunk_size': 1024,
            'max_chunks_per_iteration': 4,
            'decode_priority': 0.6,
        }
```

### 8.6 监控指标

```python
class SchedulingMetrics:
    """调度性能监控指标"""
    
    # Continuous Batching指标
    continuous_batching = {
        'throughput': 0.0,           # tokens/s
        'avg_latency': 0.0,          # ms
        'p99_latency': 0.0,          # ms
        'queue_size': 0,             # 等待队列大小
        'running_size': 0,           # 运行队列大小
        'gpu_utilization': 0.0,      # GPU利用率
        'memory_utilization': 0.0,   # 内存利用率
    }
    
    # Chunked Prefill指标
    chunked_prefill = {
        'throughput': 0.0,           # tokens/s
        'ttft': 0.0,                 # 首Token延迟
        'avg_chunk_time': 0.0,       # 平均chunk执行时间
        'num_chunks': 0,             # chunk数量
        'chunk_efficiency': 0.0,     # chunk执行效率
    }
    
    def compare_strategies(self):
        """对比两种策略的性能"""
        print("=== Continuous Batching vs Chunked Prefill ===")
        print(f"吞吐量: {self.continuous_batching['throughput']:.2f} vs {self.chunked_prefill['throughput']:.2f} tokens/s")
        print(f"平均延迟: {self.continuous_batching['avg_latency']:.2f} vs {self.chunked_prefill['ttft']:.2f} ms")
        print(f"资源利用率: {self.continuous_batching['gpu_utilization']:.2%} vs {self.chunked_prefill['chunk_efficiency']:.2%}")
```

---

## 9. 最佳实践

### 9.1 选择合适的调度策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    调度策略选择决策树                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      请求平均长度？                              │
│                           │                                     │
│                ┌──────────┴──────────┐                          │
│                │                     │                          │
│            < 2048 tokens      >= 2048 tokens                    │
│                │                     │                          │
│                ▼                     ▼                          │
│        并发请求数？           并发请求数？                       │
│            │                     │                              │
│     ┌──────┴──────┐       ┌──────┴──────┐                      │
│     │             │       │             │                      │
│   < 50         >= 50    < 50         >= 50                      │
│     │             │       │             │                      │
│     ▼             ▼       ▼             ▼                      │
│  Static      Continuous  Chunked    Hybrid                     │
│  Batching    Batching    Prefill    (混合策略)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 实际部署建议

1. **测试环境验证**：在生产环境前，使用真实负载测试不同策略
2. **监控关键指标**：持续监控吞吐量、延迟、资源利用率
3. **动态调整参数**：根据负载变化动态调整调度参数
4. **混合策略**：对于复杂场景，考虑使用混合调度策略
