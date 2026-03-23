# Page-Attention (分页注意力) 详解

## 1. 特性概述

### 1.1 什么是Page-Attention？

Page-Attention是vLLM的核心创新之一，它将操作系统的虚拟内存管理思想应用到KV Cache管理中，实现了高效的内存利用和动态内存分配。

### 1.2 核心问题

传统KV Cache管理存在的问题：
- **内存碎片化**：预分配连续内存导致碎片
- **静态分配**：无法动态调整内存大小
- **内存浪费**：预分配最大长度导致浪费
- **共享困难**：不同请求难以共享公共前缀

### 1.3 解决方案

Page-Attention通过以下机制解决上述问题：
- **分块管理**：将KV Cache划分为固定大小的Block
- **按需分配**：动态分配Block，避免预分配浪费
- **虚拟地址**：逻辑Block到物理Block的映射
- **引用计数**：支持多个序列共享Block

## 2. 设计方案

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Page-Attention架构                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Scheduler Layer                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  BlockManager                                   │   │
│  │  ├── BlockAllocator (物理Block分配器)          │   │
│  │  ├── BlockTable (逻辑到物理映射表)             │   │
│  │  └── RefCounter (引用计数管理)                 │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  Memory Layer                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Physical Blocks (物理Block池)                 │   │
│  │  ┌───┬───┬───┬───┬───┬───┬───┬───┐           │   │
│  │  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │           │   │
│  │  └───┴───┴───┴───┴───┴───┴───┴───┘           │   │
│  │  每个Block: 16 tokens * 2 (K+V) * hidden_dim  │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  Sequence Layer                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Logical Blocks (逻辑Block序列)                │   │
│  │  Seq1: [Block 0] -> [Block 3] -> [Block 5]    │   │
│  │  Seq2: [Block 0] -> [Block 2] -> [Block 4]    │   │
│  │  (共享Block 0)                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Block设计

#### Block大小选择
```python
# Block大小设计考虑因素
block_size = 16  # tokens per block

# 为什么选择16？
# 1. 内存对齐：16 * hidden_dim * 2 (K+V) * 2 bytes = 64KB (典型值)
# 2. 碎片控制：过小导致映射表过大，过大导致内存浪费
# 3. NPU优化：适配NPU的内存访问粒度
```

#### Block结构
```python
class Block:
    """
    物理Block结构
    
    内存布局:
    [K cache] [V cache]
    Shape: [block_size, num_heads, head_dim]
    """
    block_id: int          # 物理Block ID
    ref_count: int         # 引用计数
    is_allocated: bool     # 是否已分配
    device: str            # 'npu' or 'cpu'
```

### 2.3 BlockTable设计

```python
class BlockTable:
    """
    逻辑Block到物理Block的映射表
    
    示例:
    seq_id: 1
    logical_blocks: [0, 1, 2]  # 逻辑Block索引
    physical_blocks: [5, 3, 7] # 物理Block ID
    
    表示: 逻辑Block 0 -> 物理Block 5
          逻辑Block 1 -> 物理Block 3
          逻辑Block 2 -> 物理Block 7
    """
    
    def __init__(self, block_size: int, num_blocks: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.block_tables = {}  # seq_id -> List[physical_block_id]
        
    def allocate(self, seq_id: int, num_blocks: int):
        """为序列分配Block"""
        pass
        
    def free(self, seq_id: int):
        """释放序列的Block"""
        pass
```

### 2.4 内存管理策略

#### 2.4.1 水位线管理
```python
class CacheConfig:
    """KV Cache内存配置"""
    
    # GPU/NPU显存分配策略
    gpu_memory_utilization = 0.9  # 使用90%的显存
    
    # 水位线设置
    reserved_memory = 0.05  # 预留5%内存
    
    # Block池大小
    num_gpu_blocks = None   # 自动计算
    num_cpu_blocks = None   # 用于swap
```

#### 2.4.2 Swap机制
```python
class BlockAllocator:
    """
    Block分配器，支持CPU-GPU swap
    """
    
    def can_swap_in(self, block_id: int) -> bool:
        """检查是否可以swap in"""
        # 需要有足够的GPU内存
        pass
        
    def swap_in(self, block_ids: List[int]):
        """从CPU swap到GPU"""
        # 使用异步拷贝
        pass
        
    def swap_out(self, block_ids: List[int]):
        """从GPU swap到CPU"""
        # 预empted序列的Block
        pass
```

## 3. 关键代码解读

### 3.1 Block分配流程

```python
class BlockManager:
    """Block管理器核心实现"""
    
    def __init__(self, 
                 block_size: int,
                 num_gpu_blocks: int,
                 num_cpu_blocks: int):
        self.block_size = block_size
        
        # GPU Block分配器
        self.gpu_allocator = BlockAllocator(
            num_blocks=num_gpu_blocks,
            device='npu'
        )
        
        # CPU Block分配器 (用于swap)
        self.cpu_allocator = BlockAllocator(
            num_blocks=num_cpu_blocks,
            device='cpu'
        )
        
        # 每个序列的Block表
        self.block_tables: Dict[int, List[int]] = {}
        
        # 每个序列已分配的token数量 (关键：追踪实际token数)
        self.seq_num_tokens: Dict[int, int] = {}
        
        # 引用计数
        self.ref_counts: Dict[int, int] = {}
    
    def allocate_slot(self, seq_id: int) -> Tuple[int, int]:
        """
        为序列分配一个新的slot
        
        Returns:
            (block_id, slot_offset): 物理Block ID和slot偏移
        """
        # 1. 初始化序列状态
        if seq_id not in self.block_tables:
            self.block_tables[seq_id] = []
            self.seq_num_tokens[seq_id] = 0
            
        block_table = self.block_tables[seq_id]
        num_tokens = self.seq_num_tokens[seq_id]
        
        # 2. 计算slot偏移 (当前Block中的位置)
        slot_offset = num_tokens % self.block_size
        
        # 3. 检查是否需要分配新Block
        # 当 slot_offset == 0 且 num_tokens > 0 时，说明当前Block已满
        # 当 block_table 为空时，需要分配第一个Block
        if len(block_table) == 0 or slot_offset == 0:
            # 分配新的物理Block
            block_id = self.gpu_allocator.allocate()
            block_table.append(block_id)
            self.ref_counts[block_id] = 1
        else:
            # 使用最后一个Block (Block还有空闲slot)
            block_id = block_table[-1]
        
        # 4. 更新token计数
        self.seq_num_tokens[seq_id] += 1
            
        return block_id, slot_offset
    
    def fork(self, parent_seq_id: int, child_seq_id: int):
        """
        Fork一个序列（用于beam search等）
        
        关键：共享Block，增加引用计数
        """
        parent_blocks = self.block_tables[parent_seq_id]
        
        # 子序列共享父序列的Block
        self.block_tables[child_seq_id] = parent_blocks.copy()
        
        # 增加所有Block的引用计数
        for block_id in parent_blocks:
            self.ref_counts[block_id] += 1
    
    def free(self, seq_id: int):
        """
        释放序列的Block
        
        关键：减少引用计数，引用为0时才真正释放
        """
        if seq_id not in self.block_tables:
            return
            
        block_table = self.block_tables[seq_id]
        
        for block_id in block_table:
            # 减少引用计数
            self.ref_counts[block_id] -= 1
            
            # 引用为0时释放
            if self.ref_counts[block_id] == 0:
                self.gpu_allocator.free(block_id)
                del self.ref_counts[block_id]
        
        del self.block_tables[seq_id]
```

### 3.2 PagedAttention Kernel实现

#### 变量含义详解

**核心概念：连续批处理（Continuous Batching）**

在vLLM中，多个序列的token会被"打包"在一起处理，而不是传统的固定batch维度：

```
传统批处理:
  batch_size = 4, seq_len = 10
  input: [batch_size, seq_len, ...] = [4, 10, ...]
  
vLLM连续批处理:
  num_seqs = 4 (4个序列并行)
  但每个序列长度不同！
  seq1: 10 tokens
  seq2: 15 tokens  
  seq3: 8 tokens
  seq4: 12 tokens
  num_tokens = 10 + 15 + 8 + 12 = 45 (所有token打包在一起)
```

**变量对照表：**

| 变量 | 形状 | 含义 | 说明 |
|------|------|------|------|
| `num_tokens` | 标量 | **所有序列的token总数** | 不是batch_size！是所有序列token打包后的总数 |
| `num_seqs` | 标量 | **序列数量** | 类似batch_size，但每个序列长度不同 |
| `num_blocks` | 标量 | 物理Block总数 | KV Cache的总Block数量 |
| `block_tables` | `[num_seqs, max_blocks]` | 每个序列的Block映射表 | 每行是一个序列的Block ID列表 |
| `context_lens` | `[num_seqs]` | 每个序列的上下文长度 | 每个序列有多少个历史token |

**图解示例：**

```
假设: num_seqs = 3, block_size = 4

序列状态:
┌─────────────────────────────────────────────────────────┐
│ Seq 0: "Hello world!"        → 3 tokens, context_len=3 │
│ Seq 1: "How are you doing?"  → 5 tokens, context_len=5 │
│ Seq 2: "Hi"                  → 1 token,  context_len=1 │
└─────────────────────────────────────────────────────────┘

num_tokens = 3 + 5 + 1 = 9  (所有token打包在一起)

query.shape = [9, num_heads, head_dim]  ← 注意是9，不是3！

context_lens = [3, 5, 1]  ← 每个序列的长度

block_tables:
┌─────────────────────────────┐
│ Seq 0: [Block_5, Block_8]   │  ← 3 tokens需要1个Block(部分填充)
│ Seq 1: [Block_2, Block_7]   │  ← 5 tokens需要2个Block
│ Seq 2: [Block_1]            │  ← 1 token需要1个Block(部分填充)
└─────────────────────────────┘
block_tables.shape = [3, 2]  ← [num_seqs, max_num_blocks_per_seq]
```

#### Kernel实现代码

```python
def paged_attention_kernel(
    query: torch.Tensor,           # [num_tokens, num_heads, head_dim] - 所有序列的token打包
    key_cache: torch.Tensor,       # [num_blocks, block_size, num_heads, head_dim] - KV缓存块
    value_cache: torch.Tensor,     # [num_blocks, block_size, num_heads, head_dim] - KV缓存块
    block_tables: torch.Tensor,    # [num_seqs, max_num_blocks_per_seq] - 序列到Block的映射
    context_lens: torch.Tensor,    # [num_seqs] - 每个序列的上下文长度
    scale: float,                  # attention缩放因子 = 1/sqrt(head_dim)
    block_size: int,               # 每个Block的token容量
    max_context_len: int,          # 最大上下文长度
) -> torch.Tensor:
    """
    PagedAttention核心kernel
    
    关键思想：
    1. 根据block_tables找到每个token对应的物理Block
    2. 在Block内计算attention
    3. 跨Block聚合结果
    
    注意：num_tokens是所有序列token的总和，不是batch_size！
         num_seqs才是序列数量（类似batch_size）
         context_lens记录每个序列的历史长度
    """
    
    num_tokens = query.shape[0]  # 所有token总数
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    
    # 输出张量
    output = torch.zeros_like(query)
    
    # 对每个token计算attention
    for i in range(num_tokens):
        # 1. 找到该token所属的序列
        seq_id = i  # 简化，实际需要映射
        context_len = context_lens[seq_id]
        
        # 2. 获取该序列的Block表
        block_table = block_tables[seq_id]
        
        # 3. 计算attention
        # 3.1 计算query * key^T
        attn_weights = []
        for block_idx in range((context_len + block_size - 1) // block_size):
            block_id = block_table[block_idx]
            
            # 获取该Block的K cache
            key_block = key_cache[block_id]  # [block_size, num_heads, head_dim]
            
            # 计算attention score
            # query[i]: [num_heads, head_dim]
            # key_block: [block_size, num_heads, head_dim]
            attn_score = torch.matmul(
                query[i].unsqueeze(0),  # [1, num_heads, head_dim]
                key_block.transpose(-1, -2)  # [block_size, head_dim, num_heads]
            ) * scale  # [1, num_heads, block_size]
            
            attn_weights.append(attn_score)
        
        # 3.2 拼接所有Block的attention weights
        attn_weights = torch.cat(attn_weights, dim=-1)  # [1, num_heads, context_len]
        
        # 3.3 Softmax
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # 3.4 计算attention * value
        attn_output = torch.zeros(1, num_heads, head_dim)
        for block_idx in range((context_len + block_size - 1) // block_size):
            block_id = block_table[block_idx]
            
            # 获取该Block的V cache
            value_block = value_cache[block_id]  # [block_size, num_heads, head_dim]
            
            # 计算加权求和
            start = block_idx * block_size
            end = min(start + block_size, context_len)
            attn_output += torch.matmul(
                attn_weights[:, :, start:end],  # [1, num_heads, block_size]
                value_block  # [block_size, num_heads, head_dim]
            )
        
        output[i] = attn_output.squeeze(0)
    
    return output
```

### 3.3 NPU优化实现

```python
class NPUPagedAttention:
    """
    针对Ascend NPU优化的PagedAttention实现
    """
    
    def __init__(self, num_heads: int, head_dim: int, block_size: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        
        # 使用torch_npu的融合算子
        self.npu_fusion_attention = torch_npu.npu_fusion_attention
    
    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        NPU优化的PagedAttention
        
        优化点：
        1. 使用npu_fusion_attention融合算子
        2. 批量处理多个Block
        3. 减少内存访问次数
        """
        
        # 1. 根据block_tables收集所有需要的K/V cache
        # 使用索引操作，避免循环
        num_seqs = block_tables.shape[0]
        max_num_blocks = block_tables.shape[1]
        
        # 收集K cache
        # [num_seqs, max_num_blocks, block_size, num_heads, head_dim]
        gathered_key_cache = key_cache[block_tables]
        gathered_value_cache = value_cache[block_tables]
        
        # 2. 重塑为连续内存
        # [num_seqs, max_context_len, num_heads, head_dim]
        gathered_key_cache = gathered_key_cache.reshape(
            num_seqs, max_num_blocks * self.block_size, 
            self.num_heads, self.head_dim
        )
        gathered_value_cache = gathered_value_cache.reshape(
            num_seqs, max_num_blocks * self.block_size,
            self.num_heads, self.head_dim
        )
        
        # 3. 使用NPU融合attention算子
        # 转换为BNSD格式 (Batch, Num_heads, Seq_len, head_Dim)
        query = query.transpose(0, 1).unsqueeze(0)  # [1, num_heads, num_tokens, head_dim]
        gathered_key_cache = gathered_key_cache.transpose(1, 2)  # [num_seqs, num_heads, max_context_len, head_dim]
        gathered_value_cache = gathered_value_cache.transpose(1, 2)
        
        # 调用NPU融合算子
        output = self.npu_fusion_attention(
            query, gathered_key_cache, gathered_value_cache,
            head_num=self.num_heads,
            input_layout="BNSD",
            scale=1.0 / math.sqrt(self.head_dim),
            atten_mask=None,  # 使用causal mask
            keep_prob=1.0,
        )[0]
        
        return output.transpose(0, 1).squeeze(0)  # [num_tokens, num_heads, head_dim]
```

## 4. 性能优化建议

### 4.1 Block大小调优

```python
# 根据模型和硬件选择最优Block大小
def get_optimal_block_size(model_config, hardware_config):
    """
    Block大小选择指南：
    
    小Block (8-16 tokens):
    - 优点：内存利用率高，碎片少
    - 缺点：映射表大，kernel开销高
    
    大Block (32-64 tokens):
    - 优点：kernel效率高，映射表小
    - 缺点：内存浪费多，碎片多
    
    推荐值：
    - 小模型 (< 7B): 16 tokens
    - 中模型 (7B-70B): 16-32 tokens
    - 大模型 (> 70B): 32 tokens
    """
    
    if model_config.hidden_size <= 4096:
        return 16
    elif model_config.hidden_size <= 8192:
        return 16
    else:
        return 32
```

### 4.2 内存预分配

```python
class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        # 预分配所有Block，避免运行时分配
        self.blocks = torch.zeros(
            num_blocks, block_size,
            device='npu',
            dtype=torch.float16
        )
        
        # 空闲Block列表
        self.free_blocks = list(range(num_blocks))
```

### 4.3 Swap优化

```python
# 异步swap，减少阻塞
async def swap_blocks_async(block_ids: List[int], src_device: str, dst_device: str):
    """
    异步Block swap
    
    优化点：
    1. 使用异步拷贝
    2. 批量swap多个Block
    3. 与计算重叠
    """
    # 使用torch的异步拷贝
    for block_id in block_ids:
        block = get_block(block_id)
        block.to(dst_device, non_blocking=True)
```

### 4.4 KV Cache更新逻辑

KV Cache更新是PagedAttention的核心操作，发生在每次推理迭代后。

#### 更新时机

```
┌─────────────────────────────────────────────────────────────────┐
│                    KV Cache 更新时机                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Prefill阶段:                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 输入: "Hello world!"                                     │   │
│  │ 处理: 一次性计算所有token的K/V                           │   │
│  │ 更新: 将所有K/V写入Block                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Decode阶段:                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 输入: 新生成的1个token                                   │   │
│  │ 处理: 计算1个token的K/V                                  │   │
│  │ 更新: 将新K/V追加到Block                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 核心更新代码

```python
class KVCacheUpdater:
    """
    KV Cache更新器
    
    负责：
    1. 将新计算的K/V写入对应的Block slot
    2. 处理跨Block的连续写入
    3. 支持批量更新
    """
    
    def __init__(self, block_size: int, num_layers: int):
        self.block_size = block_size
        self.num_layers = num_layers
    
    def update_kv_cache(
        self,
        key: torch.Tensor,           # [num_tokens, num_heads, head_dim]
        value: torch.Tensor,         # [num_tokens, num_heads, head_dim]
        key_cache: torch.Tensor,     # [num_blocks, block_size, num_heads, head_dim]
        value_cache: torch.Tensor,   # [num_blocks, block_size, num_heads, head_dim]
        block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
        slot_mapping: torch.Tensor,  # [num_tokens] - 每个token对应的slot位置
    ):
        """
        更新KV Cache
        
        slot_mapping: 每个token应该写入的slot位置
            - slot = block_id * block_size + slot_offset
            - 由BlockManager在allocate_slot时确定
        """
        num_tokens = key.shape[0]
        
        # 方法1: 使用scatter操作批量写入 (推荐，高效)
        # 将key/value按slot_mapping写入cache
        self._scatter_update(key, key_cache, slot_mapping)
        self._scatter_update(value, value_cache, slot_mapping)
    
    def _scatter_update(
        self,
        src: torch.Tensor,      # [num_tokens, num_heads, head_dim]
        cache: torch.Tensor,    # [num_blocks, block_size, num_heads, head_dim]
        slot_mapping: torch.Tensor,  # [num_tokens]
    ):
        """
        使用scatter操作更新cache
        
        原理：
        1. 将cache展平为 [num_blocks * block_size, num_heads, head_dim]
        2. 使用slot_mapping作为索引，将src写入对应位置
        """
        num_blocks, block_size, num_heads, head_dim = cache.shape
        
        # 展平cache: [num_blocks * block_size, num_heads, head_dim]
        cache_flat = cache.view(-1, num_heads, head_dim)
        
        # 扩展slot_mapping用于scatter: [num_tokens, num_heads, head_dim]
        slot_mapping_expanded = slot_mapping.unsqueeze(-1).unsqueeze(-1)
        slot_mapping_expanded = slot_mapping_expanded.expand(-1, num_heads, head_dim)
        
        # Scatter写入
        cache_flat.scatter_(0, slot_mapping_expanded, src)
```

#### 完整更新流程

```python
class CacheEngine:
    """
    完整的Cache引擎，管理KV Cache的分配、更新、释放
    """
    
    def __init__(
        self,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_gpu_blocks: int,
    ):
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # 为每层分配KV Cache
        # shape: [num_blocks, block_size, num_heads, head_dim]
        self.gpu_cache = [
            torch.zeros(
                num_gpu_blocks, block_size, num_heads, head_dim,
                dtype=torch.float16,
                device='cuda'
            )
            for _ in range(num_layers * 2)  # K和V各num_layers个
        ]
        
        # Block Manager
        self.block_manager = BlockManager(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks
        )
    
    def update(
        self,
        seq_ids: List[int],
        new_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    ):
        """
        更新指定序列的KV Cache
        
        Args:
            seq_ids: 序列ID列表
            new_key_values: 每层的新K/V，每个元素是 (key, value) 元组
                           key/value shape: [num_new_tokens, num_heads, head_dim]
        """
        # 1. 为新token分配slot
        slot_mapping = []
        for seq_id in seq_ids:
            # 获取该序列需要分配的slot数量
            num_new_tokens = new_key_values[0][0].shape[0]
            
            for _ in range(num_new_tokens):
                block_id, slot_offset = self.block_manager.allocate_slot(seq_id)
                # 计算全局slot索引
                slot = block_id * self.block_size + slot_offset
                slot_mapping.append(slot)
        
        slot_mapping = torch.tensor(slot_mapping, device='cuda')
        
        # 2. 更新每层的KV Cache
        for layer_idx, (key, value) in enumerate(new_key_values):
            # K cache的索引
            k_cache = self.gpu_cache[layer_idx * 2]
            # V cache的索引
            v_cache = self.gpu_cache[layer_idx * 2 + 1]
            
            # 写入cache
            self._scatter_update(key, k_cache, slot_mapping)
            self._scatter_update(value, v_cache, slot_mapping)
    
    def _scatter_update(self, src, cache, slot_mapping):
        """Scatter写入"""
        num_blocks, block_size, num_heads, head_dim = cache.shape
        cache_flat = cache.view(-1, num_heads, head_dim)
        
        slot_mapping_expanded = slot_mapping.unsqueeze(-1).unsqueeze(-1)
        slot_mapping_expanded = slot_mapping_expanded.expand(-1, num_heads, head_dim)
        
        cache_flat.scatter_(0, slot_mapping_expanded, src)
```

#### Prefill阶段的批量更新

```python
def update_cache_for_prefill(
    self,
    seq_id: int,
    input_tokens: List[int],
    key_values: List[Tuple[torch.Tensor, torch.Tensor]],
):
    """
    Prefill阶段的KV Cache更新
    
    特点：
    1. 一次性写入大量token
    2. 需要分配多个Block
    3. 可能跨Block边界
    """
    num_tokens = len(input_tokens)
    num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
    
    # 1. 批量分配Block
    block_ids = []
    for _ in range(num_blocks_needed):
        block_id = self.block_manager.gpu_allocator.allocate()
        block_ids.append(block_id)
    
    self.block_manager.block_tables[seq_id] = block_ids
    
    # 2. 计算slot_mapping
    slot_mapping = []
    for i in range(num_tokens):
        block_idx = i // self.block_size
        slot_offset = i % self.block_size
        block_id = block_ids[block_idx]
        slot = block_id * self.block_size + slot_offset
        slot_mapping.append(slot)
    
    slot_mapping = torch.tensor(slot_mapping, device='cuda')
    
    # 3. 批量写入KV Cache
    for layer_idx, (key, value) in enumerate(key_values):
        k_cache = self.gpu_cache[layer_idx * 2]
        v_cache = self.gpu_cache[layer_idx * 2 + 1]
        
        self._scatter_update(key, k_cache, slot_mapping)
        self._scatter_update(value, v_cache, slot_mapping)
```

#### Decode阶段的增量更新

```python
def update_cache_for_decode(
    self,
    seq_id: int,
    new_key: torch.Tensor,   # [1, num_heads, head_dim]
    new_value: torch.Tensor, # [1, num_heads, head_dim]
    layer_idx: int,
):
    """
    Decode阶段的KV Cache更新
    
    特点：
    1. 每次只更新1个token
    2. 通常复用最后一个Block的空闲slot
    3. Block满时才分配新Block
    """
    # 1. 分配一个slot
    block_id, slot_offset = self.block_manager.allocate_slot(seq_id)
    
    # 2. 计算全局slot索引
    slot = block_id * self.block_size + slot_offset
    
    # 3. 写入对应位置
    k_cache = self.gpu_cache[layer_idx * 2]
    v_cache = self.gpu_cache[layer_idx * 2 + 1]
    
    # 直接写入指定slot
    k_cache[block_id, slot_offset] = new_key[0]
    v_cache[block_id, slot_offset] = new_value[0]
```

#### 更新流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    KV Cache 更新完整流程                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Step 1: 模型前向传播                                     │   │
│  │         输入token → Transformer → 输出K/V               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Step 2: 分配slot                                         │   │
│  │         BlockManager.allocate_slot() → (block_id, offset)│   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Step 3: 计算slot_mapping                                 │   │
│  │         slot = block_id * block_size + offset            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Step 4: Scatter写入                                      │   │
│  │         cache[slot] = K/V                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Step 5: 更新Block Table                                  │   │
│  │         记录序列到Block的映射                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 性能优化技巧

```python
# 优化1: 使用paged attention kernel直接写入
def update_with_kernel(
    self,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    使用CUDA kernel直接写入，避免Python开销
    """
    # vLLM中的实现使用自定义CUDA kernel
    # ops.reshape_and_cache(key, value, cache, slot_mapping)
    pass

# 优化2: 批量更新多个序列
def batch_update(
    self,
    seq_ids: List[int],
    keys: List[torch.Tensor],
    values: List[torch.Tensor],
):
    """
    批量更新多个序列的KV Cache
    """
    # 合并所有token
    all_keys = torch.cat(keys, dim=0)
    all_values = torch.cat(values, dim=0)
    
    # 合并所有slot_mapping
    all_slot_mapping = []
    for seq_id, key in zip(seq_ids, keys):
        num_tokens = key.shape[0]
        for _ in range(num_tokens):
            block_id, slot_offset = self.block_manager.allocate_slot(seq_id)
            all_slot_mapping.append(block_id * self.block_size + slot_offset)
    
    all_slot_mapping = torch.tensor(all_slot_mapping, device='cuda')
    
    # 一次性更新
    self._scatter_update(all_keys, self.gpu_cache[0], all_slot_mapping)
```

## 5. 实际应用案例

### 5.1 多轮对话场景

```python
# 多轮对话中的Block共享
class ConversationManager:
    def __init__(self, block_manager: BlockManager):
        self.block_manager = block_manager
        self.conversation_history = {}
    
    def add_turn(self, conversation_id: str, new_tokens: List[int]):
        """
        添加新的对话轮次
        
        关键：共享历史Block
        """
        if conversation_id in self.conversation_history:
            # Fork历史Block
            parent_seq_id = self.conversation_history[conversation_id]
            child_seq_id = generate_seq_id()
            
            self.block_manager.fork(parent_seq_id, child_seq_id)
            
            # 为新token分配Block
            for token in new_tokens:
                self.block_manager.allocate_slot(child_seq_id)
            
            # 更新conversation历史
            self.conversation_history[conversation_id] = child_seq_id
            
            # 释放旧的parent序列
            self.block_manager.free(parent_seq_id)
        else:
            # 新对话
            seq_id = generate_seq_id()
            for token in new_tokens:
                self.block_manager.allocate_slot(seq_id)
            self.conversation_history[conversation_id] = seq_id
```

### 5.2 Beam Search场景

```python
# Beam Search中的Block共享
def beam_search_with_paged_attention(
    initial_seq_id: int,
    beam_width: int,
    max_length: int
):
    """
    Beam Search实现
    
    关键优化：
    1. Fork共享Block
    2. 只为不同的token分配新Block
    """
    beam_seqs = [initial_seq_id]
    
    for step in range(max_length):
        candidates = []
        
        for seq_id in beam_seqs:
            # 生成候选token
            logits = model.forward(seq_id)
            top_k_tokens = torch.topk(logits, k=beam_width)
            
            for token in top_k_tokens:
                # Fork序列
                new_seq_id = fork_sequence(seq_id)
                
                # 为新token分配Block
                allocate_slot(new_seq_id, token)
                
                candidates.append((new_seq_id, score))
        
        # 选择top-k候选
        beam_seqs = select_top_k(candidates, k=beam_width)
    
    return beam_seqs
```

## 6. 性能监控

### 6.1 关键指标

```python
class PageAttentionMetrics:
    """性能监控指标"""
    
    # 内存利用率
    block_utilization = 0.0  # 已使用Block / 总Block
    
    # 碎片率
    fragmentation_rate = 0.0  # 未使用的slot / 总slot
    
    # Swap频率
    swap_in_count = 0
    swap_out_count = 0
    
    # 引用计数分布
    ref_count_distribution = {}  # {ref_count: num_blocks}
    
    def report(self):
        print(f"Block利用率: {self.block_utilization:.2%}")
        print(f"碎片率: {self.fragmentation_rate:.2%}")
        print(f"Swap次数: {self.swap_in_count + self.swap_out_count}")
```

## 7. 总结

Page-Attention通过操作系统级别的内存管理思想，解决了传统KV Cache管理的内存碎片化和静态分配问题。其核心创新包括：

1. **分块管理**：将KV Cache划分为固定大小的Block
2. **按需分配**：动态分配Block，避免预分配浪费
3. **引用计数**：支持多个序列共享Block
4. **Swap机制**：支持CPU-GPU内存交换

在Ascend NPU上，通过使用torch_npu的融合算子，进一步提升了Page-Attention的性能。
