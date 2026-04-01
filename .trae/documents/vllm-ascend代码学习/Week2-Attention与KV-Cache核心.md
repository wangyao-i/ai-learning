# vLLM-Ascend 框架代码学习 - Week 2

> 学习主题：Attention与KV Cache核心
> 学习目标：理解PagedAttention实现，掌握MLA/SFA差异，理解Block分配算法

***

## 一、学习进度

| 日期        | 内容                | 状态  |
| --------- | ----------------- | --- |
| Day 8-9   | Attention Backend | 待开始 |
| Day 10-11 | MLA与稀疏Attention   | 待开始 |
| Day 12-13 | KV Cache管理        | 待开始 |
| Day 14    | 本周复盘              | 待开始 |

***

## 二、代码阅读笔记

### 2.1 Attention Backend

#### attention/attention\_v1.py

**文件概述**:

- 路径: `vllm_ascend/attention/attention_v1.py`
- 功能: 基础Attention实现，包含PagedAttention和FusedInferAttention
- 依赖: torch, torch\_npu, vllm.core.attention.backend

**核心类**:

```python
@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    
    @staticmethod
    def get_name() -> str:
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"
    
    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPImpl
            return AscendAttentionCPImpl
        return AscendAttentionBackendImpl

class AscendAttentionBackendImpl(AttentionImpl):
    def __init__(self, num_heads, head_size, scale, num_kv_heads, alibi_slopes, 
                 sliding_window, kv_cache_dtype, logits_soft_cap, attn_type, 
                 kv_sharing_target_layer_name, sinks=None, **kwargs):
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.is_kv_producer = False
        self.sinks = sinks
    
    def forward_paged_attention(self, query, attn_metadata, output=None):
        if _EXTRA_CTX.capturing:
            return self.full_graph_pa(query, attn_metadata, output)
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output
    
    def forward_fused_infer_attention(self, query, key, value, attn_metadata, output):
        # 实现融合推理注意力
        pass
```

**关键问题解答**:

- **Q: 如何实现PagedAttention？**
  A: 通过`torch_npu._npu_paged_attention`内核实现，该内核接收查询张量、键值缓存、块表等参数，实现高效的注意力计算。
- **Q: 与Flash Attention的关系？**
  A: vLLM-Ascend提供了兼容Flash Attention的接口，但底层使用了昇腾NPU优化的注意力实现，包括PagedAttention和FusedInferAttention。

***

**核心功能**:

1. **PagedAttention实现**: 通过`forward_paged_attention`方法调用NPU优化的PagedAttention内核
2. **融合推理注意力**: 通过`forward_fused_infer_attention`方法实现高效的融合注意力计算
3. **元数据管理**: 通过`AscendMetadataBuilder`构建注意力计算所需的元数据
4. **图模式支持**: 支持ACL Graph模式以提高性能

**Attention状态管理**:

- PrefillNoCache: 首次填充无缓存
- PrefillCacheHit: 首次填充有缓存命中
- DecodeOnly: 仅解码阶段
- ChunkedPrefill: 分块预填充
- SpecDecoding: 推测解码

***

### 2.2 MLA与稀疏Attention

#### attention/mla\_v1.py

**文件概述**:

- 路径: `vllm_ascend/attention/mla_v1.py`
- 功能: Multi-Latent Attention实现，通过潜向量压缩KV Cache
- 依赖: torch, torch\_npu, vllm.model\_executor.layers.attention.mla\_attention

**核心类**:

```python
@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND_MLA")
class AscendMLABackend(AttentionBackend):
    accept_output_buffer: bool = True
    
    @staticmethod
    def get_name() -> str:
        return "ASCEND_MLA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"
    
    @staticmethod
    def get_builder_cls():
        if enable_cp():
            from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaCPMetadataBuilder
            return AscendMlaCPMetadataBuilder
        return AscendMLAMetadataBuilder

class AscendMLAImpl(MLAAttentionImpl):
    def __init__(self, num_heads, head_size, scale, num_kv_heads, alibi_slopes, 
                 sliding_window, kv_cache_dtype, logits_soft_cap, attn_type, 
                 kv_sharing_target_layer_name, **kwargs):
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        
        # MLA特有参数
        self.q_lora_rank = kwargs["q_lora_rank"]
        self.kv_lora_rank = kwargs["kv_lora_rank"]
        self.qk_nope_head_dim = kwargs["qk_nope_head_dim"]
        self.qk_rope_head_dim = kwargs["qk_rope_head_dim"]
        self.qk_head_dim = kwargs["qk_head_dim"]
        self.v_head_dim = kwargs["v_head_dim"]
        self.rotary_emb = kwargs["rotary_emb"]
        self.q_proj = kwargs["q_proj"] if self.q_lora_rank is None else kwargs["q_b_proj"]
        self.kv_b_proj = kwargs["kv_b_proj"]
        self.o_proj = kwargs["o_proj"]
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
    
    def _q_proj_and_k_up_proj(self, x):
        # 查询投影和键上投影
        q_nope, q_pe = (
            self.q_proj(x)[0]
            .view(-1, self.num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )
        
        # 转换为(N, B, P)格式
        q_nope = q_nope.transpose(0, 1)
        # 计算(ql_nope = q_nope @ W_UK_T)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # 转换回(B, N, L)格式
        return ql_nope.transpose(0, 1), q_pe
    
    def _v_up_proj(self, x):
        # 值上投影
        x = x.view(self.num_heads, -1, self.kv_lora_rank)
        x = torch_npu.npu_transpose_batchmatmul(x, self.W_UV, perm_y=(1, 0, 2))
        x = x.reshape(-1, self.num_heads * self.v_head_dim)
        return x
```

**关键问题解答**:

- **Q: MLA如何压缩KV Cache？**
  A: MLA通过低秩分解（LoRA）技术将高维的键值向量压缩为低维的潜向量。使用`q_lora_rank`和`kv_lora_rank`控制压缩比率，减少KV Cache的内存占用。
- **Q: 潜向量如何计算？**
  A: 通过`_q_proj_and_k_up_proj`和`_v_up_proj`方法实现：
  1. 查询向量通过`q_proj`投影并分为nope和pe两部分
  2. 键向量通过`W_UK_T`矩阵转换为潜向量
  3. 值向量通过`W_UV`矩阵转换为潜向量

***

#### attention/sfa\_v1.py

**文件概述**:

- 路径: `vllm_ascend/attention/sfa_v1.py`
- 功能: Sparse Flash Attention实现，通过稀疏计算优化注意力性能
- 依赖: torch, torch\_npu, scipy, vllm.model\_executor.layers.attention.mla\_attention

**核心类**:

```python
@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND_SFA")
class AscendSFABackend(AttentionBackend):
    accept_output_buffer: bool = True
    
    @staticmethod
    def get_name() -> str:
        return "ASCEND_SFA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"
    
    @staticmethod
    def get_impl_cls() -> type["AscendSFAImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFACPImpl
            return AscendSFACPImpl
        return AscendSFAImpl

class AscendSFAImpl(MLAAttentionImpl):
    def __init__(self, num_heads, head_size, scale, num_kv_heads, alibi_slopes, 
                 sliding_window, kv_cache_dtype, logits_soft_cap, attn_type, 
                 kv_sharing_target_layer_name, **kwargs):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        
        # MLA参数（SFA继承自MLA）
        self.q_lora_rank = kwargs["q_lora_rank"]
        self.kv_lora_rank = kwargs["kv_lora_rank"]
        self.qk_nope_head_dim = kwargs["qk_nope_head_dim"]
        self.qk_rope_head_dim = kwargs["qk_rope_head_dim"]
        self.qk_head_dim = kwargs["qk_head_dim"]
        self.v_head_dim = kwargs["v_head_dim"]
        self.rotary_emb = kwargs["rotary_emb"]
        self.q_proj = kwargs["q_proj"] if self.q_lora_rank is None else kwargs["q_b_proj"]
        self.kv_b_proj = kwargs["kv_b_proj"]
        self.o_proj = kwargs["o_proj"]
        self.indexer = kwargs["indexer"]
        
        # SFA特有参数
        self.n_head: int = self.indexer.n_head  # 64
        self.head_dim: int = self.indexer.head_dim  # 128
        self.wq_b = self.indexer.wq_b
        self.wk = self.indexer.wk
        self.weights_proj = self.indexer.weights_proj
        self.k_norm = self.indexer.k_norm
        
        # 稀疏C8支持
        self.use_sparse_c8_indexer = get_ascend_config().enable_sparse_c8
        if self.use_sparse_c8_indexer:
            self.c8_k_cache_dtype = torch.int8
            self.c8_k_scale_cache_dtype = torch.float16
    
    def indexer_select_pre_process(self, x, cos, sin):
        # 索引器预处理，生成稀疏键
        k_li, _ = self.wk(x)  # [b,s,7168] @ [7168,128] = [b,s,128]
        k_li = self.k_norm(k_li).unsqueeze(1)
        k_li = k_li.view(-1, 1, self.head_dim)
        
        # 应用RoPE
        if HAS_TRITON:
            cos = cos.view(-1, self.qk_rope_head_dim)
            sin = sin.view(-1, self.qk_rope_head_dim)
            k_li = rope_forward_triton_siso(
                k_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            # 标准RoPE实现
            pass
        
        # 稀疏C8量化
        if self.use_sparse_c8_indexer:
            k_li = k_li @ AscendSFAImpl.k_hadamard
            k_li, k_li_scale = torch_npu.npu_dynamic_quant(k_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype)
            k_li_scale = k_li_scale.to(self.c8_k_scale_cache_dtype)
            k_li_scale = k_li_scale.unsqueeze(-1)
        else:
            k_li_scale = None
        
        return k_li, k_li_scale
```

**关键问题解答**:

- **Q: 稀疏Attention如何优化？**
  A: SFA通过以下方式优化：
  1. **索引器机制**：使用索引器（indexer）选择重要的键值对，减少计算量
  2. **稀疏量化**：支持INT8稀疏量化（sparse\_c8），进一步减少内存占用
  3. **Hadamard变换**：通过Hadamard变换提高稀疏表示效率
  4. **混合精度**：根据计算需求使用不同精度
- **Q: 稀疏模式有哪些？**
  A: 主要支持：
  1. **动态稀疏**：通过索引器动态选择重要的键值对
  2. **稀疏量化**：INT8稀疏量化（sparse\_c8）
  3. **固定稀疏模式**：预定义的稀疏注意力模式

***

### 2.3 KV Cache管理

#### worker/block\_table.py

**文件概述**:

- 路径: `vllm_ascend/worker/block_table.py`
- 功能: Block Table实现，管理KV Cache的物理块与逻辑块映射
- 依赖: numpy, torch, vllm.distributed, vllm.utils.math\_utils, vllm.v1.utils.CpuGpuBuffer

**核心实现**:

```python
class BlockTable:
    def __init__(
        self, block_size: int, max_num_reqs: int, max_num_blocks_per_req: int,
        max_num_batched_tokens: int, pin_memory: bool, device: torch.device,
        kernel_sizes: list[int] | None = None, cp_kv_cache_interleave_size: int = 1,
        num_speculative_tokens: int = 0,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device
        self.physical_block_size = block_size
        
        # 初始化block_table和slot_mapping
        logical_table_size = max_num_blocks_per_req * (self.blocks_per_phys_block if self.use_hybrid_blocks else 1)
        self.block_table = self._make_buffer(max_num_reqs * duplicate_size, logical_table_size, dtype=torch.int32)
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens + 2 * self.pcp_world_size * self.max_num_reqs, dtype=torch.int32
        )
    
    def append_row(self, block_ids, row_idx: int) -> None:
        # 添加块ID到指定请求的行
        block_ids = np.array(block_ids)
        if self.use_hybrid_blocks:
            block_ids = self._convert_physical_to_logical_blocks(block_ids)
        
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids
        self.num_blocks_per_row[row_idx] += num_blocks
    
    def compute_slot_mapping(self, req_indices: np.ndarray, positions: np.ndarray) -> None:
        # 计算token位置到KV Cache槽的映射
        # 支持分布式场景下的虚拟块计算
        # 处理混合块模式下的逻辑块索引
```

**关键问题解答**:

- **Q: Block如何分配和回收？**
  A: Block分配通过`append_row`方法添加到BlockTable中，回收通过`kv_cache_manager.free(request)`方法实现。分配时支持：
  1. 混合块大小（hybrid blocks）
  2. 分布式环境下的块分配
  3. 块的预分配和延迟分配
  4. 基于Copy-on-Write的块共享
- **Q: 如何处理Copy-on-Write？**
  A: 通过以下机制实现：
  1. 检测请求前缀是否与现有请求匹配
  2. 共享匹配的KV Cache块
  3. 当生成的token与预期不同时触发复制
  4. 为当前请求创建独立的块副本
  5. 后续推理使用独立副本

***

#### 分布式KV转移

**文件概述**:

- 路径: `vllm_ascend/distributed/kv_transfer/`
- 功能: 分布式环境下的KV Cache转移与共享
- 依赖: torch, vllm.distributed, vllm\_ascend.utils

**核心实现**:

```python
# KV转移池实现
class AscendStoreConnector(KVConnector):
    def __init__(self, kv_cache_config: KVCacheConfig, block_size: int, device: torch.device):
        self.kv_cache_config = kv_cache_config
        self.block_size = block_size
        self.device = device
        self.pool_scheduler = PoolScheduler(worker_count=1)
        self.pool_scheduler.start()
    
    def get_num_new_matched_tokens(self, request, num_new_local_computed_tokens):
        # 查找可共享的KV Cache块
        key = self._generate_kv_cache_key(request)
        matched_tokens = self.kv_cache_store.get_matched_tokens(key)
        return matched_tokens, load_kv_async
    
    def update_state_after_alloc(self, request, blocks, num_external_computed_tokens):
        # 更新KV Cache状态
        # 处理异步加载的KV块
```

**关键功能**:

1. **KV池管理**: 管理分布式环境下的KV Cache池
2. **异步转移**: 支持KV Cache的异步转移
3. **前缀匹配**: 基于请求前缀匹配可共享的KV块
4. **内存优化**: 减少分布式环境下的内存占用

***

#### 块分配与碎片管理

**Block分配算法**:

1. **首次适应算法**: 寻找第一个足够大的连续块 
2. **最佳适应算法**: 寻找最小的足够大的块
3. **块预分配**: 为长序列请求预分配足够的块
4. **延迟分配**: 仅在需要时分配块

**碎片处理策略**:

1. **块合并**: 合并连续的空闲块
2. **碎片整理**: 定期整理碎片化的块
3. **动态调整**: 根据请求模式动态调整块大小
4. **优先级回收**: 回收低优先级请求的块

**关键问题解答**:

- **Q: Block分配算法是什么？**
  A: 主要使用首次适应算法和预分配策略：
  1. 对于短序列请求使用首次适应
  2. 对于长序列请求使用预分配
  3. 支持延迟分配以提高内存利用率
- **Q: 如何处理Block碎片？**
  A: 通过以下策略处理：
  1. 块合并：合并连续的空闲块
  2. 碎片整理：定期整理碎片化的块
  3. 动态块大小：支持混合块大小以减少碎片
  4. 优先级回收：优先回收低优先级请求的块

***

## 三、架构图

### 3.1 Attention实现对比

```
待补充
```

### 3.2 KV Cache管理流程

```
待补充
```

***

## 四、与vLLM原版差异

| 模块               | vLLM原版                | vLLM-Ascend            | 差异说明 |
| ---------------- | --------------------- | ---------------------- | ---- |
| AttentionBackend | FlashAttentionBackend | AscendAttentionBackend | 待补充  |
| BlockManager     | BlockManager          | AscendBlockManager     | 待补充  |
| KV Cache         | GPU KV Cache          | NPU KV Cache           | 待补充  |

***

## 五、疑问与待深入

- [ ] 问题1
- [ ] 问题2

***

## 六、本周复盘

### 收获

1. 待补充

### 待深入

1. 待补充

### 下周计划

1. 调度与批处理

