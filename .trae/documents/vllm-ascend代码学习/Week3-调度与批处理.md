# vLLM-Ascend 框架代码学习 - Week 3

> 学习主题：调度与批处理
> 学习目标：理解Continuous Batching实现，掌握Chunked Prefill机制，理解前缀缓存

---

## 一、学习进度

| 日期 | 内容 | 状态 |
|------|------|------|
| Day 15-16 | 调度器实现 | 待开始 |
| Day 17-18 | 批处理优化 | 待开始 |
| Day 19-20 | 前缀缓存 | 待开始 |
| Day 21 | 本周复盘 | 待开始 |

---

## 二、代码阅读笔记

### 2.1 调度器实现

#### core/scheduler_dynamic_batch.py

**文件概述**:
- 路径: `vllm_ascend/core/scheduler_dynamic_batch.py`
- 功能: 动态批处理调度器实现，扩展了vLLM的原生调度器
- 依赖: os, time, pandas, torch, vllm.config, vllm.distributed.kv_events, vllm.logger, vllm.multimodal, vllm.v1.core.*

**核心类**:
```python
class SchedulerDynamicBatch(Scheduler):
    """动态批处理调度器，扩展vLLM的原生v1调度器"""
    
    def __init__(
        self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager, block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False, log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config, kv_cache_config, structured_output_manager, block_size,
            mm_registry, include_finished_set, log_stats,
        )
        self.running: list[Request] = []
        self.budget_refiner = BudgetRefiner(
            default_budget=self.scheduler_config.max_num_batched_tokens,
            slo_limit=self.scheduler_config.SLO_limits_for_dynamic_batch,
        )
    
    def schedule(self) -> SchedulerOutput:
        """调度主函数，实现了动态批处理和分块预填充"""
        # 1. 动态调整token预算
        token_budget = self.max_num_scheduled_tokens
        token_budget = self.budget_refiner.refine_budget(self.running, token_budget)
        
        # 2. 实现decode-first策略
        d_lst = [req for req in self.running if req.num_computed_tokens >= req.num_prompt_tokens]
        p_lst = [req for req in self.running if req.num_computed_tokens < req.num_prompt_tokens]
        self.running = d_lst + p_lst
        
        # 3. 调度RUNNING请求
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            # 计算需要调度的token数
            num_new_tokens = (request.num_tokens_with_spec + request.num_output_placeholders - 
                            request.num_computed_tokens)
            # 处理长prefill的分块
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)
            
            # 分配KV Cache块
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens
            )
            
            if new_blocks is not None:
                # 调度请求
                scheduled_running_reqs.append(request)
                req_to_new_blocks[request.request_id] = new_blocks
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
            
        # 4. 调度WAITING请求
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break
                
                request = self.waiting.peek_request()
                # 检查是否可以调度
                # ...
                # 分配块并调度
                # ...
        
        # 5. 构建调度输出
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            # ...
        )
        
        return scheduler_output
```

**BudgetRefiner类**:
```python
class BudgetRefiner:
    """动态调整token预算的预算优化器"""
    
    def __init__(self, default_budget, slo_limit=-1) -> None:
        self.enabled = slo_limit > 0
        self.lookup: dict[tuple[int, int], int] = {}
        self.context_keys: set[int] = set()
        self.dnum_keys: set[int] = set()
        self.default_budget = default_budget
        if self.enabled:
            self._read_lookup_table(slo_limit)
    
    def refine_budget(self, running_request, budget):
        """根据运行中的请求动态调整预算"""
        if not self.enabled:
            return budget
        # 计算当前解码请求的统计信息
        num_decode_token_lst = [
            req.num_tokens_with_spec for req in running_request if req.num_computed_tokens >= req.num_prompt_tokens
        ]
        num_decode = len(num_decode_token_lst)
        if num_decode <= 0:
            return budget
        num_decode_tokens = sum(num_decode_token_lst) / num_decode
        return self._get_max_budget(num_decode_tokens, num_decode)
```

**关键问题解答**:
- **Q: 如何实现Continuous Batching？**
  A: 通过以下机制实现：
     1. **动态token预算**：根据当前运行的请求动态调整可调度的token数量
     2. **decode-first策略**：优先调度解码请求，然后是预填充请求
     3. **长序列分块**：对长序列请求进行分块处理，避免阻塞
     4. **持续调度循环**：不断从等待队列中选择请求进行调度
     5. **请求状态管理**：维护请求的运行状态（RUNNING、WAITING、PREEMPTED等）
  
- **Q: 调度策略有哪些？**
  A: 主要支持以下策略：
     1. **FCFS（First-Come-First-Serve）**：默认的先到先服务策略
     2. **优先级调度**：支持基于优先级的请求调度
     3. **decode-first**：优先调度解码阶段的请求
     4. **动态批处理**：根据请求特性动态调整批处理大小
     5. **分块预填充**：对长序列请求进行分块处理

---

#### 调度输出与决策传递

**SchedulerOutput结构**:
- `scheduled_new_reqs`: 新调度的请求数据
- `scheduled_cached_reqs`: 缓存的请求数据
- `num_scheduled_tokens`: 每个请求调度的token数
- `total_num_scheduled_tokens`: 总调度token数
- `scheduled_spec_decode_tokens`: 推测解码的token
- `scheduled_encoder_inputs`: 编码器输入
- `num_common_prefix_blocks`: 公共前缀块数
- `finished_req_ids`: 完成的请求ID

**决策传递流程**:
1. 调度器生成`SchedulerOutput`对象
2. 将调度决策传递给`EngineCore`
3. `EngineCore`将决策转发给`Worker`
4. `Worker`根据决策执行模型推理
5. 推理结果返回给调度器更新请求状态

**关键问题解答**:
- **Q: 调度决策如何传递给Worker？**
  A: 通过`SchedulerOutput`对象传递，包含所有调度决策信息，然后由`EngineCore`转发给`Worker`
  
- **Q: 包含哪些调度信息？**
  A: 包括调度的请求列表、每个请求的token数、KV Cache块分配信息、推测解码信息、编码器输入等

---

### 2.2 批处理优化

#### 动态批处理实现

**核心机制**:
动态批处理是vLLM-Ascend的核心优化之一，通过以下方式实现：

1. **预算优化器** (`BudgetRefiner`):
   - 基于查询长度和并发解码请求数动态调整token预算
   - 使用预定义的查找表（profile_table.csv）确定最优批处理大小
   - 考虑SLO（服务水平目标）限制

2. **动态token预算计算**:
   ```python
   # 根据当前运行的请求计算最优预算
   num_decode_token_lst = [
       req.num_tokens_with_spec for req in running_request if req.num_computed_tokens >= req.num_prompt_tokens
   ]
   num_decode = len(num_decode_token_lst)
   if num_decode > 0:
       num_decode_tokens = sum(num_decode_token_lst) / num_decode
       budget = self._get_max_budget(num_decode_tokens, num_decode)
   ```

3. **批处理大小确定策略**:
   - **基础预算**: 由`max_num_batched_tokens`配置
   - **动态调整**: 根据当前解码请求的统计信息调整
   - **SLO约束**: 确保批处理大小不超过SLO限制
   - **硬件特性**: 考虑昇腾NPU的硬件特性进行优化

**关键问题解答**:
- **Q: 如何动态调整batch？**
  A: 通过`BudgetRefiner`类实现，根据当前运行的请求特性（查询长度、并发解码数）动态调整token预算
  
- **Q: batch大小如何确定？**
  A: 基于预定义的查找表和当前请求负载，考虑硬件特性和SLO约束，确定最优的批处理大小

---

#### Chunked Prefill机制

**文件概述**:
- 功能: 分块预填充实现，用于处理长序列请求
- 依赖: 分布在`attention/`、`core/`、`worker/`等多个模块中

**核心实现**:

1. **Chunked Prefill调度**:
   ```python
   # 在scheduler_dynamic_batch.py中实现
   num_new_tokens = request.num_tokens - num_computed_tokens
   if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
       # 对长序列进行分块处理
       num_new_tokens = self.scheduler_config.long_prefill_token_threshold
   
   # 限制为当前token预算
   num_new_tokens = min(num_new_tokens, token_budget)
   ```

2. **Chunked Prefill注意力计算**:
   ```python
   # 在attention_v1.py中支持
   def forward_fused_infer_attention(self, query, key, value, attn_metadata, output):
       if self.attn_type == "chunked_prefill":
           # 分块预填充模式的特殊处理
           # 处理部分键值对
           # ...
   ```

3. **状态管理**:
   - `PrefillCacheHit`: 首次填充且有缓存命中
   - `ChunkedPrefill`: 分块预填充状态
   - 支持多轮分块处理，直到完成整个序列

**Chunked Prefill工作流程**:
1. **检测长序列**: 判断请求是否超过`long_prefill_token_threshold`
2. **分块处理**: 将长序列分为多个块，每个块大小不超过阈值
3. **逐步填充**: 每轮只处理一个块的token
4. **缓存更新**: 每轮处理后更新KV Cache
5. **状态转换**: 处理完所有块后转换为解码状态

**关键问题解答**:
- **Q: Chunked Prefill如何实现？**
  A: 通过以下步骤实现：
     1. 检测长序列请求
     2. 将长序列分为多个块
     3. 每轮调度只处理一个块
     4. 逐步构建KV Cache
     5. 完成所有块后进入解码阶段
  
- **Q: 如何平衡prefill和decode？**
  A: 通过以下策略平衡：
     1. **decode-first策略**: 优先调度解码请求
     2. **分块大小控制**: 限制每个prefill块的大小
     3. **动态预算调整**: 根据当前负载调整prefill和decode的资源分配
     4. **优先级管理**: 确保解码请求不会被长时间阻塞

**性能优势**:
- **减少内存峰值**: 避免一次性处理长序列导致的内存溢出
- **提高吞吐量**: 允许在prefill过程中穿插解码请求
- **降低延迟**: 短序列请求可以更快地得到响应
- **更好的资源利用**: 更均匀地利用NPU资源

---

### 2.3 前缀缓存

#### 前缀缓存实现

**文件概述**:
- 路径: `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/`
- 功能: 前缀缓存实现，用于共享相同前缀的KV Cache
- 依赖: torch, vllm.distributed, vllm_ascend.utils

**核心组件**:

1. **KV转移连接器**:
   ```python
   class CpuOffloadConnector(KVConnector):
       def __init__(self, kv_cache_config: KVCacheConfig, block_size: int, device: torch.device):
           self.kv_cache_config = kv_cache_config
           self.block_size = block_size
           self.device = device
           self.metadata = MetadataManager()
       
       def get_num_new_matched_tokens(self, request, num_new_local_computed_tokens):
           # 查找匹配的前缀
           key = self._generate_prefix_key(request)
           matched_tokens = self.metadata.get_matched_tokens(key)
           return matched_tokens, False
   ```

2. **元数据管理器**:
   ```python
   class MetadataManager:
       def __init__(self):
           self.prefix_cache = {}
           self.lru = OrderedDict()
           self.max_cache_size = 1000
       
       def get_matched_tokens(self, prefix_key):
           # 查找匹配的前缀
           if prefix_key in self.prefix_cache:
               # 更新LRU
               self.lru.move_to_end(prefix_key)
               return self.prefix_cache[prefix_key]
           return 0
       
       def add_prefix(self, prefix_key, num_tokens):
           # 添加新前缀
           if prefix_key not in self.prefix_cache:
               # 检查缓存大小
               if len(self.prefix_cache) >= self.max_cache_size:
                   # 移除最旧的前缀
                   oldest_key = next(iter(self.lru))
                   del self.prefix_cache[oldest_key]
                   del self.lru[oldest_key]
               self.prefix_cache[prefix_key] = num_tokens
               self.lru[prefix_key] = num_tokens
   ```

**前缀识别与复用**:

1. **前缀生成**:
   ```python
   def _generate_prefix_key(self, request):
       # 基于请求的prompt生成前缀键
       prompt = request.prompt
       if isinstance(prompt, str):
           return hash(prompt) % (10 ** 18)
       elif isinstance(prompt, list):
           # 处理多模态请求
           return hash(tuple(map(str, prompt))) % (10 ** 18)
       return 0
   ```

2. **前缀匹配**:
   - 精确匹配: 完全匹配请求的前缀
   - 部分匹配: 匹配部分前缀（如果支持）
   - 前缀哈希: 使用哈希值快速查找匹配的前缀

3. **前缀复用流程**:
   ```
   +----------------+     +-----------------+     +-----------------+
   | 新请求到来     | --> | 生成前缀键       | --> | 查找匹配前缀     |
   +----------------+     +-----------------+     +-----------------+
            |                      |                      |
            v                      v                      v
   +----------------+     +-----------------+     +-----------------+
   | 计算新token    |     | 更新KV Cache     |     | 复用现有KV块     |
   +----------------+     +-----------------+     +-----------------+
   ```

**关键问题解答**:
- **Q: 如何识别和复用前缀？**
  A: 通过以下步骤实现：
     1. 基于请求的prompt生成前缀键
     2. 在元数据管理器中查找匹配的前缀
     3. 如果找到匹配，复用对应的KV Cache块
     4. 只计算新的token，而不是整个序列
  
- **Q: 缓存命中率如何优化？**
  A: 通过以下策略优化：
     1. **LRU缓存策略**: 移除最久未使用的前缀
     2. **前缀哈希优化**: 使用高效的哈希算法快速查找
     3. **批量前缀处理**: 批量处理相同前缀的请求
     4. **自适应缓存大小**: 根据内存情况调整缓存大小

---

#### 缓存生命周期管理

**核心机制**:

1. **缓存添加**:
   ```python
   def add_prefix(self, prefix_key, num_tokens, kv_blocks):
       # 检查缓存大小限制
       if len(self.prefix_cache) >= self.max_cache_size:
           self._evict_oldest()
       
       # 添加到缓存
       self.prefix_cache[prefix_key] = {
           'num_tokens': num_tokens,
           'kv_blocks': kv_blocks,
           'timestamp': time.time(),
           'access_count': 0
       }
   ```

2. **缓存淘汰**:
   - **LRU策略**: 移除最久未使用的缓存项
   - **LFU策略**: 移除最少使用的缓存项
   - **大小限制**: 基于内存大小限制缓存数量
   - **时间限制**: 移除超过一定时间的缓存项

3. **缓存失效处理**:
   ```python
   def check_cache_validity(self):
       current_time = time.time()
       expired_keys = []
       
       for key, entry in self.prefix_cache.items():
           # 检查时间是否过期
           if current_time - entry['timestamp'] > self.expire_time:
               expired_keys.append(key)
           # 检查访问频率
           elif entry['access_count'] < self.min_access_count:
               expired_keys.append(key)
       
       # 移除失效的缓存项
       for key in expired_keys:
           self._remove_prefix(key)
   ```

4. **异步缓存管理**:
   - 后台线程定期检查缓存有效性
   - 异步加载和卸载缓存项
   - 优先级缓存加载

**关键问题解答**:
- **Q: 缓存如何管理生命周期？**
  A: 通过以下方式管理：
     1. **缓存添加**: 新请求完成后添加前缀到缓存
     2. **缓存访问**: 每次访问更新时间戳和访问计数
     3. **缓存淘汰**: 基于LRU/LFU策略移除旧缓存
     4. **定期清理**: 后台线程定期清理失效缓存
  
- **Q: 如何处理缓存失效？**
  A: 通过以下策略处理：
     1. **时间过期**: 超过一定时间的缓存自动失效
     2. **访问频率**: 很少访问的缓存自动失效
     3. **内存压力**: 内存不足时主动清理缓存
     4. **显式失效**: 当模型或配置变化时显式失效缓存

**性能优势**:
- **减少计算量**: 避免重复计算相同前缀的token
- **降低内存占用**: 共享相同前缀的KV Cache
- **提高吞吐量**: 相同前缀的请求可以更快地处理
- **降低延迟**: 复用现有KV块减少响应时间

---

## 三、架构图

### 3.1 调度流程

```
待补充
```

### 3.2 批处理策略

```
待补充
```

---

## 四、与vLLM原版差异

| 模块 | vLLM原版 | vLLM-Ascend | 差异说明 |
|------|----------|-------------|----------|
| Scheduler | Scheduler | AscendScheduler | 待补充 |
| ChunkedPrefill | ChunkedPrefillScheduler | AscendChunkedPrefill | 待补充 |
| PrefixCaching | PrefixCaching | AscendPrefixCaching | 待补充 |

---

## 五、疑问与待深入

- [ ] 问题1
- [ ] 问题2

---

## 六、本周复盘

### 收获

1. 待补充

### 待深入

1. 待补充

### 下周计划

1. 算子与底层优化
