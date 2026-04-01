# vLLM-Ascend 框架代码学习 - Week 5

> 学习主题：量化特性
> 学习目标：理解AWQ/SmoothQuant/GPTQ/KV8量化实现，掌握量化策略选择

---

## 一、学习进度

| 日期 | 内容 | 状态 |
|------|------|------|
| Day 29-30 | W4A16-AWQ量化 | 待开始 |
| Day 31-32 | W8A8-SmoothQuant量化 | 待开始 |
| Day 33-34 | W8A16-GPTQ量化与KV8 | 待开始 |
| Day 35 | 本周复盘 | 待开始 |

---

## 二、代码阅读笔记

### 2.1 W4A16-AWQ量化

#### quantization/methods/w4a16.py

**文件概述**:
- 路径: `vllm_ascend/quantization/methods/w4a16.py`
- 功能: AWQ (Activation-aware Weight Quantization) 量化实现，支持INT4权重和FP16激活
- 依赖: torch, torch_npu, vllm.config, vllm_ascend.ascend_config

**核心实现**:
```python
# 权重打包与解包工具函数

def unpack_from_int32(
    weight: torch.Tensor,  # 打包的int32权重张量
    shape: torch.Size,     # 原始形状
    num_bits: int,         # 量化位数
    packed_dim: int = 1,   # 打包维度
) -> torch.Tensor:
    """将INT32打包的权重解包为原始格式"""
    assert weight.dtype == torch.int32, f"期望int32类型，得到{weight.dtype}"
    assert num_bits <= 8, f"期望位数≤8，得到{num_bits}"
    
    pack_factor = 32 // num_bits  # 压缩因子（8个INT4打包为1个INT32）
    mask = (1 << num_bits) - 1     # 掩码
    
    # 根据打包维度解包
    if packed_dim == 1:
        unpacked_weight = torch.zeros(
            (weight.shape[0], weight.shape[1] * pack_factor),
            device=weight.device, dtype=torch.int32
        )
        for i in range(pack_factor):
            unpacked_weight[:, i::pack_factor] = (weight >> (num_bits * i)) & mask
    else:
        # 按行打包的情况
        ...
    
    # 应用偏移校正（从无符号转为有符号）
    offset = pow(2, num_bits) // 2
    unpacked_weight = (unpacked_weight - offset).to(torch.int8)
    
    return unpacked_weight

# AWQ量化方案实现

@register_scheme("W4A16", "moe")
class AscendW4A16FusedMoEMethod(AscendMoEScheme):
    """昇腾平台的W4A16量化MoE实现"""
    
    quant_type: QuantType = QuantType.W4A16
    
    def __init__(self) -> None:
        self.transpose_weight = True
        self.num_bits = 4  # INT4量化
        self.pack_factor = 8  # 8个INT4打包为1个INT32
        
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
    
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, router_logits: torch.Tensor, ...
    ) -> torch.Tensor:
        # 选择专家
        topk_weights, topk_ids = select_experts(...)
        
        # 调用融合MoE专家计算（包含量化操作）
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight_packed,  # 打包的INT4权重
                w2=layer.w2_weight_packed,    # 打包的INT4权重
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                # 缩放因子和偏移量
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w1_offset=layer.w13_weight_offset,
                w2_offset=layer.w2_weight_offset,
                ...
            )
        )
```

**关键问题解答**:
- **Q: 如何实现权重INT4量化？**
  A: 通过以下步骤实现：
     1. **分组量化**：按group_size（默认32）将权重分组
     2. **量化计算**：对每个分组计算缩放因子和偏移量
     3. **打包存储**：将8个INT4权重打包为1个INT32，减少内存占用
     4. **运行时解包**：在计算前解包权重并应用反量化
  
- **Q: 激活缩放如何计算？**
  A: AWQ的激活缩放在量化阶段计算：
     1. **激活感知**：分析激活值分布，找出敏感的权重通道
     2. **缩放因子计算**：为不同通道分配不同的缩放因子，保护敏感通道
     3. **权重重缩放**：根据激活分布动态调整权重，减少量化误差
  
- **Q: 反量化流程是什么？**
  A: 反量化流程：
     1. **解包**：将INT32打包的权重解包为INT8格式
     2. **偏移校正**：应用偏移转换为有符号整数
     3. **缩放**：乘以缩放因子恢复到FP16范围
     4. **计算**：与FP16激活进行矩阵乘法

**性能特点**:
- 权重压缩率8倍（从FP16到INT4）
- 小并发场景下时延提升约80%
- 支持MoE模型的高效量化

---

#### quantization/utils.py

**文件概述**:
- 路径: `vllm_ascend/quantization/utils.py`
- 功能: 量化工具函数集合
- 依赖: torch, torch_npu, numpy

**核心函数**:
```python
def quantize_tensor(tensor: torch.Tensor, num_bits: int, group_size: int) -> tuple:
    """量化张量，返回量化权重、缩放因子和偏移量"""
    # 按group_size分组
    # 计算每个组的缩放因子和偏移量
    # 量化权重
    return quantized_weight, scale, offset

def dequantize_tensor(quantized_weight: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """反量化张量"""
    # 应用缩放和偏移恢复原始值
    return dequantized_weight
```

**关键问题解答**:
- **Q: 如何处理激活缩放？**
  A: 通过以下方式处理：
     1. **动态缩放**：根据实际输入激活值动态调整
     2. **预计算缩放因子**：在模型加载时预计算并存储
     3. **硬件加速**：利用昇腾NPU的量化指令优化缩放计算
  
- **Q: 量化误差如何控制？**
  A: 误差控制策略：
     1. **分组量化**：小粒度分组减少量化误差
     2. **激活感知**：保护对激活敏感的权重通道
     3. **偏移校正**：使用有符号量化减少零点偏移误差
     4. **混合精度**：关键路径保持FP16精度

---

### 2.2 W8A8-SmoothQuant量化

#### quantization/methods/w8a8_static.py

**文件概述**:
- 路径: `vllm_ascend/quantization/methods/w8a8_static.py`
- 功能: SmoothQuant量化实现，支持INT8权重和INT8激活
- 依赖: torch, torch_npu, vllm_ascend.utils

**核心实现**:
```python
@register_scheme("W8A8", "linear")
class AscendW8A8LinearMethod(AscendLinearScheme):
    """昇腾平台的W8A8静态量化线性层实现
    
    使用静态的per-tensor量化激活和per-channel量化权重
    """
    
    def __init__(self) -> None:
        pass
    
    def get_weight(
        self, input_size: int, output_size: int, params_dtype: torch.dtype = torch.bfloat16
    ) -> dict[str, Any]:
        # 定义INT8量化权重
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict
    
    def get_pertensor_param(self, params_dtype: torch.dtype) -> dict[str, Any]:
        # 定义per-tensor量化参数
        params_dict = {}
        params_dict["input_scale"] = torch.empty(1, dtype=params_dtype)
        params_dict["input_offset"] = torch.empty(1, dtype=torch.int8)
        return params_dict
    
    def get_perchannel_param(
        self, output_size: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        # 定义per-channel量化参数
        params_dict = {}
        params_dict["quant_bias"] = torch.empty(output_size, dtype=torch.int32)
        if params_dtype == torch.bfloat16:
            params_dict["deq_scale"] = torch.empty(output_size, dtype=torch.float32)
        elif params_dtype == torch.float16:
            params_dict["deq_scale"] = torch.empty(output_size, dtype=torch.int64)
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict
    
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None, tp_rank: int | None = 0
    ) -> torch.Tensor:
        # 如果输入不是INT8，先量化激活
        if x.dtype != torch.int8:
            # 权重预取优化
            weight_prefetch_method = get_weight_prefetch_method()
            weight_prefetch_method.maybe_prefetch_attn_weight_preprocess(...)
            
            # 量化输入激活
            x = torch.ops.vllm.quantize(
                x,
                layer.aclnn_input_scale,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
            )
            
            # 权重预取后处理
            weight_prefetch_method.maybe_prefetch_attn_weight_postprocess(...)
        
        # 调用NPU量化矩阵乘法
        output = torch_npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=layer.params_dtype,
        )
        return output
    
    def process_weights_after_loading(self, layer):
        # 处理加载后的权重
        expanding_factor = layer.weight.data.shape[1]
        # 扩展输入缩放因子
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor), requires_grad=False
        )
        # 计算缩放因子倒数（优化运行时性能）
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor), requires_grad=False
        )
        # 扩展输入偏移
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor), requires_grad=False
        ).to(layer.aclnn_input_scale.dtype)
        
        # 转置权重以优化内存访问
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        # 可能的内存格式转换
        layer.weight.data = maybe_trans_nz(layer.weight.data)
        # 扁平化缩放因子和偏移量
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
```

**关键问题解答**:
- **Q: 如何平衡权重和激活量化？**
  A: SmoothQuant通过以下方式平衡：
     1. **平滑因子**：在权重和激活之间插入缩放因子，重新分配量化误差
     2. **激活重缩放**：将激活的动态范围部分转移到权重
     3. **权重平滑**：减少权重的动态范围，提高量化精度
     4. **静态量化**：使用离线校准确定最佳量化参数
  
- **Q: 平滑因子如何计算？**
  A: 平滑因子计算过程：
     1. **校准数据收集**：使用校准数据集收集激活分布
     2. **动态范围分析**：计算每个通道的激活和权重的动态范围
     3. **平滑因子确定**：根据公式 s = α * (max|A|) / (max|W|) 计算
     4. **参数存储**：将平滑因子合并到权重中，减少运行时开销

**性能特点**:
- 权重和激活均为INT8，内存占用减少4倍
- 吞吐提升约30%
- 精度损失小，适合对精度要求较高的场景
- 支持分布式量化通信优化

---

### 2.3 W8A16-GPTQ量化

#### quantization/methods/w8a16.py

**文件概述**:
- 路径: `vllm_ascend/quantization/methods/w8a16.py`
- 功能: GPTQ (GPT-Q) 量化实现，支持INT8权重和FP16激活
- 依赖: torch, torch_npu, vllm_ascend.utils

**核心实现**:
```python
@register_scheme("W8A16", "linear")
class AscendW8A16LinearMethod(AscendLinearScheme):
    """昇腾平台的W8A16量化线性层实现
    
    使用8位量化权重和16位激活，适合GPTQ量化模型
    """
    
    def __init__(self) -> None:
        pass
    
    def get_weight(
        self, input_size: int, output_size: int, params_dtype: torch.dtype = torch.bfloat16
    ) -> dict[str, Any]:
        # 定义INT8量化权重
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict
    
    def get_perchannel_param(
        self, output_size: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        # 定义per-channel量化参数
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict
    
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None, tp_rank: int | None = 0
    ) -> torch.Tensor:
        # 调用NPU量化矩阵乘法
        output = torch_npu.npu_weight_quant_batchmatmul(
            x=x,                      # FP16激活
            weight=layer.weight,      # INT8量化权重
            antiquant_scale=layer.weight_scale,  # 反量化缩放因子
            antiquant_offset=layer.weight_offset, # 反量化偏移量
            bias=bias,                # 可选的bias
        )
        return output
    
    def process_weights_after_loading(self, layer):
        # 转置权重以优化内存访问
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        # 可能的内存格式转换
        layer.weight.data = maybe_trans_nz(layer.weight.data)
        # 扁平化缩放因子和偏移量
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
```

**关键问题解答**:
- **Q: GPTQ量化流程是什么？**
  A: GPTQ量化流程：
     1. **逐层量化**：按层顺序量化，利用前层输出校准后层
     2. **最优权重量化**：对每个权重矩阵，通过优化算法找到最优量化值
     3. **误差传播**：将量化误差传播到后续层，进行补偿
     4. **高效实现**：使用块优化和并行计算加速量化过程
  
- **Q: 与AWQ的差异？**
  A: GPTQ与AWQ的主要差异：
     | 特性 | GPTQ | AWQ |
     |------|------|-----|
     | 量化策略 | 逐层优化，误差传播 | 激活感知，权重均衡 |
     | 量化粒度 | per-channel | per-group |
     | 压缩率 | 4倍（INT8） | 8倍（INT4） |
     | 量化速度 | 较慢（需要优化） | 较快 |
     | 精度 | 较高 | 中等 |
     | 硬件要求 | 较低 | 较高 |

**性能特点**:
- 权重压缩率4倍（从FP16到INT8）
- 吞吐提升约20%
- 精度损失小，适合生产环境
- 实现简单，部署成本低

---

### 2.4 KV Cache量化

#### quantization/methods/kv_c8.py

**文件概述**:
- 路径: `vllm_ascend/quantization/methods/kv_c8.py`
- 功能: KV Cache量化实现，支持多种注意力机制（MLA、SFA、Dense Attention）
- 依赖: torch, vllm.config, vllm.distributed

**核心实现**:

```python
# FAKQuant - 用于MLA的KV量化

@register_scheme("FAKQuant", "attention")
class AscendFAQuantAttentionMethod:
    """MLA-based C8 (FAKQuant) KV缓存量化"""
    
    def __init__(self):
        self.transpose_weight = True
        vllm_config = get_current_vllm_config()
        config = vllm_config.model_config.hf_config
        self.kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        self.qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
    
    def create_weights(self, layer: torch.nn.Module) -> None:
        # 创建FA量化所需的参数
        extra_module_names = ["fa_q", "fa_k", "fa_v"]
        for name in extra_module_names:
            setattr(layer, name, torch.nn.Module())
        
        dtype = torch.get_default_dtype()
        # 创建缩放因子和偏移量参数
        params_dict = {
            "fa_q.scale": torch.empty((layer.num_heads, 1), dtype=dtype),
            "fa_k.scale": torch.empty((layer.num_kv_heads, 1), dtype=dtype),
            "fa_v.scale": torch.empty((layer.num_kv_heads, 1), dtype=dtype),
            "fa_q.offset": torch.empty((layer.num_heads, 1), dtype=torch.int8),
            "fa_k.offset": torch.empty((layer.num_kv_heads, 1), dtype=torch.int8),
            "fa_v.offset": torch.empty((layer.num_kv_heads, 1), dtype=torch.int8),
        }
        
        # 注册参数
        for name, weight in params_dict.items():
            module_name, weight_name = name.rsplit(".", 1)
            module = getattr(layer, module_name)
            weight_param = torch.nn.Parameter(weight, requires_grad=False)
            module.register_parameter(weight_name, weight_param)
            weight_param.weight_loader = _fa_quant_weight_loader
    
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # 处理加载后的权重，准备运行时使用
        fa_k_scale = torch.squeeze(layer.fa_k.scale).unsqueeze(0)
        layer.fak_descale_float = torch.nn.Parameter(fa_k_scale.to(torch.float), requires_grad=False)
        layer.fak_descale = torch.nn.Parameter(fa_k_scale, requires_grad=False)
        layer.fak_descale_reciprocal = 1.0 / torch.nn.Parameter(fa_k_scale, requires_grad=False)
        
        fa_k_offset = torch.squeeze(layer.fa_k.offset).unsqueeze(0)
        layer.fak_offset = torch.nn.Parameter(fa_k_offset.to(layer.fak_descale.dtype), requires_grad=False)

# C8 KV Cache - 用于Dense Attention模型

class AscendC8KVCacheAttentionMethod(AscendAttentionScheme):
    """C8 INT8 KV缓存量化（用于Dense Attention模型，如Qwen3）"""
    
    def __init__(self, quant_description: dict, prefix: str):
        self.quant_description = quant_description
        self.prefix = prefix
    
    def create_weights(self, layer: torch.nn.Module) -> None:
        # 覆盖kv_cache_torch_dtype，使Attention.get_kv_cache_spec自动返回int8
        layer.kv_cache_torch_dtype = torch.int8
        
        # 升级实现到C8特定的子类
        if hasattr(layer, "impl"):
            from vllm_ascend.attention.attention_v1 import AscendC8AttentionBackendImpl
            layer.impl.__class__ = AscendC8AttentionBackendImpl
        
        # 创建KV缓存量化参数
        layer.k_cache_scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        layer.k_cache_scale.weight_loader = _c8_kv_scale_weight_loader
        layer.k_cache_offset = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)
        layer.k_cache_offset.weight_loader = _c8_kv_scale_weight_loader
        layer.v_cache_scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        layer.v_cache_scale.weight_loader = _c8_kv_scale_weight_loader
        layer.v_cache_offset = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)
        layer.v_cache_offset.weight_loader = _c8_kv_scale_weight_loader
```

**关键问题解答**:
- **Q: 如何量化KV Cache？**
  A: KV Cache量化通过以下方式实现：
     1. **INT8量化**：将KV缓存从FP16量化为INT8
     2. **缩放因子与偏移量**：为每个头计算独立的缩放因子和偏移量
     3. **硬件加速**：利用昇腾NPU的量化指令优化计算
     4. **多种注意力支持**：为MLA、SFA和Dense Attention提供不同的量化实现
     5. **运行时反量化**：在注意力计算时高效反量化
  
- **Q: 量化对精度的影响？**
  A: KV Cache量化对精度的影响：
     1. **可忽略的精度损失**：INT8量化在KV缓存上的精度损失通常小于0.1% BLEU/PPL
     2. **上下文长度影响**：较长上下文下精度保持稳定
     3. **不同模型差异**：不同模型对KV量化的敏感性不同，需要针对性调优
     4. **量化策略选择**：可以选择不同的量化粒度（per-head、per-tensor）平衡精度和性能

**性能特点**:
- KV缓存内存占用减少4倍（从FP16到INT8）
- 吞吐提升约15-25%
- 时延降低约10-20%
- 支持动态批量处理
- 兼容所有主要注意力机制

---

## 三、架构图

### 3.1 量化实现对比

```
┌─────────────────────────────────────────────────────────────────┐
│                        量化特性对比                              │
├──────────────┬─────────────┬─────────────┬──────────────────────┤
│    特性       │   权重位宽   │   激活位宽   │       性能收益        │
├──────────────┼─────────────┼─────────────┼──────────────────────┤
│    AWQ       │    INT4     │    FP16     │ 小并发时延提升80%     │
│ SmoothQuant  │    INT8     │    INT8     │ 吞吐提升30%          │
│    GPTQ      │    INT8     │    FP16     │ 吞吐提升20%          │
│    KV8       │   KV INT8   │     -       │ 吞吐提升15-25%       │
└──────────────┴─────────────┴─────────────┴──────────────────────┘
```

---

## 四、与vLLM原版差异

| 模块 | vLLM原版 | vLLM-Ascend | 差异说明 |
|------|----------|-------------|----------|
| AWQ | CUDA实现 | NPU实现 | 待补充 |
| SmoothQuant | CUDA实现 | NPU实现 | 待补充 |
| KV Cache量化 | FP8 | INT8 | 待补充 |

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

1. 图模式特性
