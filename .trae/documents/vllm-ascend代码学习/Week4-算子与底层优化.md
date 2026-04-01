# vLLM-Ascend 框架代码学习 - Week 4

> 学习主题：算子与底层优化
> 学习目标：理解核心算子实现，掌握算子融合策略，理解内存管理机制

---

## 一、学习进度

| 日期 | 内容 | 状态 |
|------|------|------|
| Day 22-23 | 核心算子 | 待开始 |
| Day 24-25 | 算子融合 | 待开始 |
| Day 26-27 | 内存管理 | 待开始 |
| Day 28 | 本周复盘 | 待开始 |

---

## 二、代码阅读笔记

### 2.1 核心算子

#### ops/layernorm.py

**文件概述**:
- 路径: `vllm_ascend/ops/layernorm.py`
- 功能: 昇腾NPU优化的LayerNorm系列算子实现
- 依赖: torch, torch_npu, vllm.config, vllm.model_executor.layers.layernorm

**核心实现**:
```python
class AscendRMSNorm(RMSNorm):
    def __init__(
        self, hidden_size: int, eps: float = 1e-6, var_hidden_size: int | None = None,
        has_weight: bool = True, dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        # 支持量化场景下的norm.bias
        vllm_config = get_current_vllm_config()
        if vllm_config.quant_config is not None and any(
            "norm.bias" in name for name in vllm_config.quant_config.quant_description
        ):
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
            self.bias.weight_loader = self._bias_weight_loader
    
    def forward_oot(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        import torch_npu
        
        if residual is not None:
            residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            # 支持自定义算子和原生算子两种模式
            if enable_custom_op():
                x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x, residual, self.weight, self.bias, self.variance_epsilon
                )
            else:
                x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
                if self.bias is not None:
                    x.add_(self.bias)
            return x, residual
        
        x, residual = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
        if self.bias_loaded:
            x.add_(self.bias)
        
        # 权重预取优化
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mlp_weight_postprocess(x)
        return x

class AscendGemmaRMSNorm(GemmaRMSNorm):
    # Gemma模型专用的RMSNorm实现
    def forward_oot(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        import torch_npu
        
        if residual is not None:
            residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            if enable_custom_op():
                x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x, residual, 1.0 + self.weight, None, self.variance_epsilon
                )
            else:
                x, _, residual = torch_npu.npu_add_rms_norm(x, residual, 1.0 + self.weight, self.variance_epsilon)
            return x, residual
        
        x, _ = torch.ops._C_ascend.npu_gemma_rms_norm(x, self.weight, self.variance_epsilon)
        return x

class AscendRMSNormGated(RMSNormGated):
    # 带门控的RMSNorm实现
    def forward_oot(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""
        return LayerNormFn.apply(x, self.weight, self.bias, z, self.eps, self.group_size, self.norm_before_gate, True)
```

**关键问题解答**:
- **Q: 如何优化归一化计算？**
  A: 通过以下方式优化：
     1. 使用NPU融合算子`npu_add_rms_norm`和`npu_add_rms_norm_bias`
     2. 支持自定义C++算子进一步优化性能
     3. 集成权重预取机制减少内存访问延迟
     4. 支持残差连接与归一化的融合计算
  
- **Q: 支持哪些归一化类型？**
  A: 支持以下归一化类型：
     1. **RMSNorm**: 常规RMS归一化
     2. **GemmaRMSNorm**: Gemma模型专用的RMS归一化
     3. **RMSNormGated**: 带门控的RMS归一化（支持group_size配置）

---

#### ops/linear.py

**文件概述**:
- 路径: `vllm_ascend/ops/linear.py`
- 功能: 昇腾NPU优化的线性层系列实现
- 依赖: torch, torch.nn, vllm.config, vllm.distributed, vllm.model_executor.layers.linear

**核心实现**:
```python
class AscendQKVParallelLinear(QKVParallelLinear):
    """Attention层QKV变换的并行线性层"""
    
    def __init__(
        self, hidden_size: int, head_size: int, total_num_heads: int,
        total_num_kv_heads: int | None = None, bias: bool = True,
        skip_bias_add: bool = False, params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None, prefix: str = "",
        *, return_bias: bool = True, disable_tp: bool = False,
        v_head_size: int | None = None,
    ):
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        # 获取并行算子配置
        self.custom_op, _, tp_size = get_parallel_op(disable_tp, prefix, self, "column")
        # 初始化QKV并行线性层
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        
        # 计算TP分割后的参数
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        
        # 初始化父类
        AscendColumnParallelLinear.__init__(...)
    
    def forward(self, input_) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        return super().forward(input_)

class AscendColumnParallelLinear(ColumnParallelLinear):
    """列并行线性层"""
    def __init__(self, input_size: int, output_size: int, bias: bool = True,
                 gather_output: bool = False, skip_bias_add: bool = False,
                 params_dtype: torch.dtype | None = None,
                 quant_config: QuantizationConfig | None = None,
                 output_sizes: list[int] | None = None, prefix: str = "",
                 *, return_bias: bool = True, disable_tp: bool = False):
        # 获取并行算子配置
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(disable_tp, prefix, self, "column")
        # 计算TP分割后的参数
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        # 初始化父类
        AscendLinearBase.__init__(...)

class AscendRowParallelLinear(RowParallelLinear):
    """行并行线性层"""
    # 实现类似ColumnParallelLinear
```

**关键问题解答**:
- **Q: 如何调用NPU融合算子？**
  A: 通过以下方式调用：
     1. 直接调用`torch_npu`提供的NPU原生算子（如`torch_npu.npu_rms_norm`）
     2. 通过`enable_custom_op()`条件判断使用自定义C++算子（如`torch.ops._C_ascend.npu_add_rms_norm_bias`）
     3. 在linear层中通过`get_parallel_op`获取优化的并行算子
  
- **Q: 与CUDA Flash Attention的差异？**
  A: 主要差异包括：
     1. **硬件适配**：针对昇腾NPU架构优化，而非NVIDIA GPU
     2. **算子接口**：使用`torch_npu`而非`torch.cuda`
     3. **优化策略**：结合昇腾NPU的AI Core和AI CPU架构特点进行优化
     4. **内存管理**：适配昇腾NPU的内存层次结构
     5. **分布式支持**：支持昇腾特有的分布式通信方式

---

### 2.2 算子融合

#### 融合算子实现

**文件概述**:
- 路径: `vllm_ascend/ops/triton/` 和 `vllm_ascend/compilation/passes/`
- 功能: 实现各种算子融合优化
- 依赖: torch, torch_npu, triton, vllm.compilation

**核心融合算子**:

1. **QKV拆分与RMSNorm融合**:
   ```python
   # ops/triton/linearnorm/split_qkv_rmsnorm_rope.py
   def split_qkv_rmsnorm_rope(
       x, weight, cos, sin, qkv_head_num, qkv_head_dim, qkv_interleaved=False,
       q_proj=None, k_proj=None, v_proj=None, bias=None
   ):
       """融合RMSNorm + QKV拆分 + RoPE计算"""
       # 1. RMSNorm归一化
       # 2. QKV线性变换与拆分
       # 3. RoPE位置编码应用
       # 4. 返回处理后的Q, K, V张量
   ```

2. **注意力计算融合**:
   ```python
   # ops/triton/fla/ 目录下的各种注意力融合算子
   def fused_qkvzba_split_reshape(...):
       """融合QKV拆分、重塑、归一化等操作"""
   
   def chunk_scaled_dot_kkt(...):
       """融合缩放点积注意力计算"""
   ```

3. **MLP层融合**:
   ```python
   # ops/triton/activation/swiglu_quant.py
   def swiglu_quant(...):
       """融合Swish激活、线性变换和量化操作"""
   ```

**融合优化Pass**:

```python
# compilation/passes/qknorm_rope_fusion_pass.py
class QKNormRoPEFusionPass(CompilationPass):
    """融合QK归一化和RoPE计算的Pass"""
    
    def apply(self, graph: fx.Graph) -> fx.Graph:
        # 遍历计算图
        for node in graph.nodes:
            # 识别QK归一化和RoPE计算模式
            # 替换为融合算子
            pass
        return graph

# compilation/passes/allreduce_rmsnorm_fusion_pass.py
class AllReduceRMSNormFusionPass(CompilationPass):
    """融合AllReduce和RMSNorm操作的Pass"""
    
    def apply(self, graph: fx.Graph) -> fx.Graph:
        # 遍历计算图
        for node in graph.nodes:
            # 识别AllReduce后跟RMSNorm的模式
            # 替换为融合算子
            pass
        return graph
```

**关键问题解答**:
- **Q: 哪些算子可以融合？**
  A: 主要融合以下算子组合：
     1. **归一化+线性变换**：RMSNorm + QKV线性变换
     2. **注意力相关**：QKV拆分 + RoPE + 注意力计算
     3. **激活函数**：Swish/GeLU + 线性变换
     4. **分布式操作**：AllReduce + RMSNorm
     5. **序列操作**：cumsum + 其他序列处理
  
- **Q: 融合策略是什么？**
  A: 采用以下融合策略：
     1. **计算图分析**：通过编译Pass分析计算图，识别可融合模式
     2. **硬件感知**：根据昇腾NPU的架构特点选择融合方式
     3. **内存优化**：减少中间结果的内存读写
     4. **性能优先**：优先融合计算密集型算子
     5. **灵活性**：支持不同粒度的融合控制

---

#### 自定义算子开发

**文件概述**:
- 功能: 开发自定义C++/TIK算子以优化性能
- 依赖: 昇腾CANN Toolkit, TIK DSL

**自定义算子调用**:

```python
# 在Python中调用自定义算子
import torch
from vllm_ascend.utils import enable_custom_op

# 检查是否启用自定义算子
if enable_custom_op():
    # 调用自定义C++算子
    x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
        x, residual, self.weight, self.bias, self.variance_epsilon
    )
else:
    # 回退到原生算子
    x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
```

**TIK DSL使用**:

```python
# TIK DSL示例（简化版）
from te import tik

def custom_attention_kernel():
    # 创建TIK实例
    tik_instance = tik.Tik()
    
    # 定义输入输出张量
    q = tik_instance.Tensor("float16", (BATCH, HEAD, SEQ, DIM), name="q", scope=tik.scope_gm)
    k = tik_instance.Tensor("float16", (BATCH, HEAD, SEQ, DIM), name="k", scope=tik.scope_gm)
    v = tik_instance.Tensor("float16", (BATCH, HEAD, SEQ, DIM), name="v", scope=tik.scope_gm)
    out = tik_instance.Tensor("float16", (BATCH, HEAD, SEQ, DIM), name="out", scope=tik.scope_gm)
    
    # 实现注意力计算逻辑
    # 1. QK矩阵乘法
    # 2. 缩放
    # 3. Softmax
    # 4. 与V矩阵乘法
    
    # 发射指令
    tik_instance.BuildCCE(kernel_name="custom_attention", inputs=[q, k, v], outputs=[out])
    return tik_instance
```

**关键问题解答**:
- **Q: 如何开发自定义算子？**
  A: 通过以下步骤开发：
     1. 使用C++/CUDA开发基础算子
     2. 注册为PyTorch扩展算子
     3. 编写Python包装器
     4. 集成到vLLM-Ascend的算子体系中
  
- **Q: TIK DSL如何使用？**
  A: TIK DSL使用步骤：
     1. 导入tik模块
     2. 创建Tik实例
     3. 定义输入输出张量
     4. 使用TIK指令实现计算逻辑
     5. 构建CCE kernel
     6. 编译生成算子二进制文件
     7. 在Python中调用

**性能优势**:
- 减少kernel启动开销
- 优化内存访问模式
- 提高计算资源利用率
- 充分发挥昇腾NPU的架构优势

---

### 2.3 内存管理

#### device_allocator/camem.py

**文件概述**:
- 路径: `vllm_ascend/device_allocator/camem.py`
- 功能: CANN内存池管理，支持休眠/唤醒模式
- 依赖: torch, torch_npu, acl.rt, vllm.logger

**核心实现**:
```python
class CaMemAllocator:
    """CANN内存池管理器，支持休眠/唤醒模式的单例类"""
    
    instance = None
    default_tag: str = "default"
    
    @staticmethod
    def get_instance() -> "CaMemAllocator":
        """获取单例实例"""
        if CaMemAllocator.instance is None:
            CaMemAllocator.instance = CaMemAllocator()
        return CaMemAllocator.instance
    
    def __init__(self):
        # 初始化内存池配置
        conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        assert "expandable_segments:True" not in conf, "不支持可扩展段"
        
        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = CaMemAllocator.default_tag
        self.allocator_and_pools: dict[str, Any] = {}
    
    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        """使用内存池的上下文管理器"""
        if tag is None:
            tag = CaMemAllocator.default_tag
        
        old_tag = self.current_tag
        self.current_tag = tag
        with use_memory_pool_with_allocator(self.python_malloc_callback, self.python_free_callback) as data:
            self.allocator_and_pools[tag] = data
            yield
            self.current_tag = old_tag
    
    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        """休眠模式：将指定tag的内存卸载到CPU，其他内存释放"""
        if offload_tags is None:
            offload_tags = (CaMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)
        
        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            if data.tag in offload_tags:
                # 卸载到CPU内存
                size_in_bytes = handle[1]
                cpu_backup_tensor = torch.empty(size_in_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
                cpu_ptr = cpu_backup_tensor.data_ptr()
                ACL_MEMCPY_DEVICE_TO_HOST = 2
                memcpy(cpu_ptr, dest_max, ptr, size_in_bytes, ACL_MEMCPY_DEVICE_TO_HOST)
                data.cpu_backup_tensor = cpu_backup_tensor
            # 释放设备内存
            unmap_and_release(handle)
        
        gc.collect()
        torch.npu.empty_cache()
    
    def wake_up(self, tags: list[str] | None = None) -> None:
        """唤醒模式：将卸载的内存加载回NPU"""
        for ptr, data in self.pointer_to_data.items():
            if tags is None or data.tag in tags:
                handle = data.handle
                # 重新映射设备内存
                create_and_map(handle)
                if data.cpu_backup_tensor is not None:
                    # 从CPU加载数据回NPU
                    cpu_backup_tensor = data.cpu_backup_tensor
                    size_in_bytes = cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                    cpu_ptr = cpu_backup_tensor.data_ptr()
                    ACL_MEMCPY_HOST_TO_DEVICE = 1
                    memcpy(ptr, dest_max, cpu_ptr, size_in_bytes, ACL_MEMCPY_HOST_TO_DEVICE)
                    data.cpu_backup_tensor = None
```

**关键问题解答**:
- **Q: 如何优化NPU内存使用？**
  A: 通过以下方式优化：
     1. **内存池管理**：使用CAMEM内存池减少内存分配开销
     2. **休眠/唤醒机制**：支持将不活跃内存卸载到CPU，释放NPU内存
     3. **标签管理**：按标签分组管理内存，精细控制内存生命周期
     4. **内存复用**：同一标签的内存可以在不同上下文间复用
     5. **批量操作**：批量分配和释放内存，减少系统调用
  
- **Q: NUMA感知内存分配如何实现？**
  A: 通过以下机制实现：
     1. **CPU绑定**：在`cpu_binding.py`中实现NUMA节点感知的CPU绑定
     2. **内存亲和性**：确保内存分配在本地NUMA节点
     3. **分布式感知**：在分布式环境下考虑节点间的内存访问成本

---

#### 块分配器与内存碎片管理

**文件概述**:
- 路径: `vllm_ascend/worker/block_table.py` 和相关文件
- 功能: 管理KV Cache的块分配，减少内存碎片
- 依赖: numpy, torch, vllm.distributed

**核心实现**:

```python
class BlockTable:
    """管理KV Cache的块分配表"""
    
    def __init__(
        self, block_size: int, max_num_reqs: int, max_num_blocks_per_req: int,
        max_num_batched_tokens: int, pin_memory: bool, device: torch.device,
        kernel_sizes: list[int] | None = None, cp_kv_cache_interleave_size: int = 1,
        num_speculative_tokens: int = 0,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.physical_block_size = block_size
        
        # 混合块大小支持
        if kernel_sizes is None or kernel_sizes == [0]:
            self.block_size = block_size
            self.use_hybrid_blocks = False
        else:
            # 选择合适的内核大小
            selected_kernel_size = None
            for kernel_size in kernel_sizes:
                if kernel_size > 0 and self.physical_block_size % kernel_size == 0:
                    selected_kernel_size = kernel_size
                    break
            
            self.block_size = selected_kernel_size
            self.blocks_per_phys_block = self.physical_block_size // self.block_size
            self.use_hybrid_blocks = self.blocks_per_phys_block > 1
        
        # 初始化块表
        logical_table_size = max_num_blocks_per_req * (self.blocks_per_phys_block if self.use_hybrid_blocks else 1)
        self.block_table = self._make_buffer(max_num_reqs * duplicate_size, logical_table_size, dtype=torch.int32)
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
    
    def append_row(self, block_ids, row_idx: int) -> None:
        """添加块ID到指定请求的行"""
        if not block_ids:
            return
        block_ids = np.array(block_ids)
        if self.use_hybrid_blocks:
            block_ids = self._convert_physical_to_logical_blocks(block_ids)
        
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids
        self.num_blocks_per_row[row_idx] += num_blocks
```

**关键问题解答**:
- **Q: 如何减少内存碎片？**
  A: 通过以下策略减少内存碎片：
     1. **块化管理**：将内存划分为固定大小的块，统一管理
     2. **混合块大小**：支持不同大小的块，适应不同长度的序列
     3. **块合并**：定期合并连续的空闲块
     4. **延迟分配**：仅在需要时分配块，减少碎片产生
     5. **优先级回收**：优先回收长时间不使用的块
  
- **Q: Block大小如何确定？**
  A: Block大小的确定策略：
     1. **默认块大小**：由配置参数`block_size`指定
     2. **内核大小适配**：根据`kernel_sizes`选择合适的块大小
     3. **硬件特性**：考虑昇腾NPU的内存访问特性
     4. **工作负载感知**：根据实际请求的序列长度分布调整
     5. **混合块模式**：支持将物理块分割为更小的逻辑块

**性能优势**:
- 减少内存碎片，提高内存利用率
- 优化内存访问模式，提高缓存命中率
- 支持高效的块分配和回收
- 适应不同的工作负载需求

---

### 2.4 内存优化策略总结

**核心优化策略**:
1. **内存池管理**：使用CAMEM内存池减少分配开销
2. **休眠/唤醒机制**：灵活管理内存生命周期
3. **块化内存**：减少碎片，提高利用率
4. **混合块大小**：适应不同序列长度
5. **NUMA感知**：优化CPU-NPU内存访问
6. **算子融合**：减少中间结果内存占用

**与vLLM原版差异**:
| 特性 | vLLM原版 | vLLM-Ascend |
|------|----------|-------------|
| 内存池 | 基础内存池 | CAMEM高级内存池（支持休眠/唤醒） |
| 块管理 | 固定块大小 | 混合块大小支持 |
| 内存迁移 | 基本支持 | 高效CPU-NPU内存迁移 |
| NUMA支持 | 有限支持 | 完善的NUMA感知分配 |
| 自定义分配器 | CUDA专用 | CANN专用分配器 |

---

## 三、架构图

### 3.1 算子调用链路

```
待补充
```

### 3.2 内存管理流程

```
待补充
```

---

## 四、与vLLM原版差异

| 模块 | vLLM原版 | vLLM-Ascend | 差异说明 |
|------|----------|-------------|----------|
| Attention Op | FlashAttention CUDA | FlashAttention NPU | 待补充 |
| Fused Ops | CUDA fused kernels | Ascend fused kernels | 待补充 |
| Memory | CUDA memory | NPU memory | 待补充 |

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

1. 量化特性
