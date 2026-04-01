# vLLM-Ascend 框架代码学习 - Week 6

> 学习主题：图模式特性
> 学习目标：理解Turbo-Graph/Acl-Graph/Torch.Compile实现，掌握图模式选择策略

---

## 一、学习进度

| 日期 | 内容 | 状态 |
|------|------|------|
| Day 36-37 | Ascend-Turbo-Graph | 待开始 |
| Day 38-39 | Acl-Graph | 待开始 |
| Day 40-41 | Torch.Compile与高效解码 | 待开始 |
| Day 42 | 本周复盘 | 待开始 |

---

## 二、代码阅读笔记

### 2.1 Ascend-Turbo-Graph

**概述**:
- **Ascend-Turbo-Graph**是昇腾平台的高级图优化技术，实现整图下沉和动态shape支持
- **核心特点**：整图优化、动态形状自适应、自动融合优化
- **性能收益**：Decode吞吐翻倍

**实现原理**:
1. **整图捕获**：将整个模型的前向传播捕获为单个图
2. **动态形状处理**：支持运行时动态调整输入形状
3. **自动优化**：自动进行算子融合、内存优化和并行调度
4. **硬件加速**：充分利用昇腾NPU的AI Core和AI CPU架构

**关键功能**:
- **自适应图优化**：根据输入形状自动调整图结构
- **内存池化**：共享图执行过程中的临时内存
- **异步执行**：支持图内算子的异步执行和调度
- **分布式支持**：兼容分布式训练和推理场景

### 2.2 Acl-Graph

#### compilation/acl_graph.py

**文件概述**:
- 路径: `vllm_ascend/compilation/acl_graph.py`
- 功能: ACL图模式实现，基于昇腾NPU的图优化技术
- 依赖: torch, torch_npu, vllm.compilation, vllm.config

**核心实现**:
```python
@dataclasses.dataclass
class ACLGraphEntry:
    """ACL图入口，管理图的批处理描述符、图对象和输出"""
    batch_descriptor: BatchDescriptor
    aclgraph: torch.npu.NPUGraph | None = None  # NPU图对象
    output: Any | None = None                     # 图执行输出
    input_addresses: list[int] | None = None     # 调试用：记录输入地址

class ACLGraphWrapper:
    """ACL图包装器，为可执行对象添加图捕获和重放能力"""
    
    def __init__(
        self, runnable: Callable, vllm_config: VllmConfig, runtime_mode: CUDAGraphMode
    ):
        self.runnable = runnable                     # 原始可执行对象
        self.vllm_config = vllm_config               # vLLM配置
        self.runtime_mode = runtime_mode             # 运行时模式（FULL/PIECEWISE）
        self.graph_pool = current_platform.get_global_graph_pool()  # 图池
        self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry] = {}  # 图入口缓存
    
    def __call__(self, *args, **kwargs):
        """执行图捕获或重放"""
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode
        
        # 检查是否需要使用图模式
        if aclgraph_runtime_mode == CUDAGraphMode.NONE or aclgraph_runtime_mode != self.runtime_mode:
            return self.runnable(*args, **kwargs)
        
        # 获取或创建图入口
        if batch_descriptor not in self.concrete_aclgraph_entries:
            self.concrete_aclgraph_entries[batch_descriptor] = ACLGraphEntry(batch_descriptor=batch_descriptor)
        
        entry = self.concrete_aclgraph_entries[batch_descriptor]
        
        # 捕获新图
        if entry.aclgraph is None:
            logger.debug("Capturing a aclgraph on (%s,%s)", self.runtime_mode.name, entry.batch_descriptor)
            
            # 记录输入地址（调试用）
            input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            entry.input_addresses = input_addresses
            
            # 创建NPU图对象
            aclgraph = torch.npu.NPUGraph()
            
            with torch.npu.graph(aclgraph, pool=self.graph_pool):
                # 在图上下文中执行原始函数
                output = self.runnable(*args, **kwargs)
                
            # 保存图和输出
            entry.output = weak_ref_tensors(output)  # 使用弱引用节省内存
            entry.aclgraph = aclgraph
            compilation_counter.num_cudagraph_captured += 1
            
            return output
        
        # 重放已有图
        logger.info_once("Replaying aclgraph")
        torch.npu.current_stream().synchronize()  # 确保顺序执行
        entry.aclgraph.replay()  # 重放图
        return entry.output
```

**关键问题解答**:
- **Q: 如何实现分段图执行？**
  A: 通过以下方式实现：
     1. **PIECEWISE模式**：将模型拆分为多个子图，每个子图单独捕获和重放
     2. **层粒度捕获**：大致按层粒度划分图，减少单次捕获的复杂度
     3. **图间通信**：使用事件和流同步机制确保子图间的正确执行顺序
     4. **内存优化**：使用弱引用管理图的工作区和输出，减少内存占用
  
- **Q: 与Turbo Graph的差异？**
  A: 与Turbo Graph的差异：
     | 特性 | Turbo Graph | ACL Graph |
     |------|-------------|-----------|
     | 图粒度 | 整图 | 分段/整图 |
     | 实现方式 | 高级图优化 | 基于NPUGraph |
     | 性能提升 | 更高（Decode吞吐翻倍） | 中等（40-60%） |
     | 灵活性 | 较低 | 较高 |
     | 适用场景 | 固定形状 | 动态形状 |

**性能特点**:
- 相比eager模式提升40-60%
- 支持FULL和PIECEWISE两种图模式
- 自动管理图缓存和内存
- 兼容动态批量大小
- 支持分布式环境

---

#### 图缓存管理

**文件概述**:
- 功能: 管理编译后的图缓存，提高图重用率
- 依赖: torch.npu, vllm.envs

**核心实现**:
```python
def set_graph_params(aclgraph_capture_sizes: list[int]):
    """设置图参数，为不同大小的输入预分配资源"""
    global _graph_params
    _graph_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},  # 事件
        {size: None for size in aclgraph_capture_sizes}, # 工作区
        {size: [] for size in aclgraph_capture_sizes},  # 句柄
        {size: [] for size in aclgraph_capture_sizes},  # 注意力参数
    )

def update_graph_params_workspaces(num_tokens: int, workspace: torch.Tensor):
    """更新图工作区，根据实际输入大小调整"""
    global _graph_params
    if _graph_params is not None:
        _graph_params.workspaces[num_tokens] = workspace
```

**关键问题解答**:
- **Q: 如何缓存编译后的图？**
  A: 图缓存策略：
     1. **基于BatchDescriptor**：使用批处理描述符作为缓存键
     2. **多尺寸预分配**：根据配置的`cudagraph_capture_sizes`预分配图资源
     3. **弱引用管理**：使用弱引用减少内存占用，支持自动回收
     4. **全局图池**：使用平台提供的全局图池管理图对象
  
- **Q: 缓存失效策略是什么？**
  A: 缓存失效策略：
     1. **内存压力触发**：当内存不足时，自动释放不活跃的图
     2. **显式清理**：在模型卸载时显式清理所有图资源
     3. **弱引用自动回收**：当图不再被引用时自动回收
     4. **配置限制**：通过配置参数限制最大缓存图数量

---

### 2.3 Torch.Compile

#### compilation/compiler_interface.py

**文件概述**:
- 路径: `vllm_ascend/compilation/compiler_interface.py`
- 功能: Torch编译接口实现，对接torch.compile和昇腾后端
- 依赖: torch, torch.fx, vllm.compilation, vllm_ascend.ascend_config

**核心实现**:
```python
class AscendCompiler(CompilerInterface):
    """昇腾平台的自定义编译器接口"""
    
    name = "AscendCompiler"
    
    def compile(
        self, graph: fx.GraphModule, example_inputs: list[Any], compiler_config: dict[str, Any],
        compile_range: Range, key: str | None = None
    ) -> tuple[Callable | None, Any | None]:
        # 深拷贝图以避免被inductor修改
        graph = copy.deepcopy(graph)
        
        # 获取昇腾编译配置
        ascend_compilation_config = get_ascend_config().ascend_compilation_config
        
        if ascend_compilation_config.enable_npugraph_ex:
            # 使用NPUGraph EX模式
            logger.info("enable_npugraph_ex is enabled, which will bring graph compilation optimization.")
            return npugraph_ex_compile(
                graph, example_inputs, compiler_config, self.vllm_config, ascend_compilation_config, compile_range, key
            )
        else:
            # 使用融合Pass模式
            return fusion_pass_compile(graph, example_inputs, compiler_config, compile_range, key)

def npugraph_ex_compile(
    graph: fx.GraphModule, example_inputs: list[Any], compiler_config: dict[str, Any],
    vllm_config: VllmConfig, ascend_compilation_config: AscendCompilationConfig,
    compile_range: Range, key: str | None = None
) -> tuple[Callable | None, Any | None]:
    """使用NPUGraph EX编译FX图"""
    import torchair
    
    # 配置torchair编译器
    torch.npu.set_compile_mode(jit_compile=False)
    config = torchair.CompilerConfig()
    config.mode = "reduce-overhead"  # ACL图模式
    config.debug.run_eagerly = True  # 先在eager模式下执行以优化FX图
    
    # 配置静态形状内核（如果启用）
    if ascend_compilation_config.enable_static_kernel:
        logger.info("enable_static_kernel is enabled, static shape kernel will be used.")
        config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        # 设置静态形状范围
        # ...
    
    # 获取NPU后端
    npugraph_ex = torchair.get_npu_backend(compiler_config=config)
    
    # 确保图返回元组
    if not graph_returns_tuple(graph):
        return make_graph_return_tuple(graph, example_inputs, npugraph_ex), None
    return npugraph_ex(graph, example_inputs), None
```

**关键问题解答**:
- **Q: 如何对接torch.compile？**
  A: 通过以下方式对接：
     1. **实现CompilerInterface**：继承vLLM的CompilerInterface接口
     2. **FX图处理**：接收PyTorch FX图并进行优化
     3. **后端选择**：根据配置选择使用NPUGraph EX或融合Pass
     4. **结果返回**：返回编译后的可执行函数
  
- **Q: 后端转换流程是什么？**
  A: 后端转换流程：
     1. **FX图捕获**：torch.compile捕获模型的FX图
     2. **图优化**：应用自定义的融合Pass优化图结构
     3. **后端转换**：将FX图转换为昇腾NPU可执行的形式
     4. **编译执行**：在NPU上编译并执行优化后的图

**性能特点**:
- 推理性能提升30-50%
- 与torch.compile无缝集成
- 支持多种优化策略
- 兼容昇腾NPU特性

---

### 2.4 高效解码特性

#### prefix_caching（前缀缓存）

**文件概述**:
- 路径: `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/`
- 功能: 前缀缓存（Auto-Prefix-Caching），识别和复用请求中的公共前缀
- 依赖: torch, zmq, vllm.config, multiprocessing.shared_memory

**核心实现**:
```python
class MetadataServer:
    """元数据服务器，管理前缀缓存的元数据和CPU KV缓存"""
    
    def __init__(self, vllm_config: VllmConfig):
        self.world_size = vllm_config.parallel_config.world_size
        self.pipeline_parallel_size = vllm_config.parallel_config.pipeline_parallel_size
        # 配置CPU交换空间
        available_memory_gb = kv_transfer_config.get_from_extra_config("cpu_swap_space_gb", 800)
        self.available_memory = available_memory_gb * 1024 * 1024 * 1024
        
        # ZMQ通信设置
        self.ctx = zmq.Context()
        self.socket = make_zmq_socket(
            self.ctx, "ipc:///tmp/metadata.ipc", zmq.ROUTER, bind=True
        )
        
        # 注册服务函数
        self.functions = {
            "init_cpu_kv_caches": self.init_cpu_kv_caches,
            "get_matched_num_and_touch": self.cpu_block_manager.get_matched_num_and_touch,
            "allocate_slots": self.cpu_block_manager.allocate_slots,
            "record_request_cache_and_free_slots": self.cpu_block_manager.record_request_cache_and_free_slots,
        }
    
    def init_cpu_kv_caches(
        self, pp_rank: int, tp_rank: int, kv_cache_specs: dict[str, AttentionSpec], mla_config: MLAConfig
    ) -> tuple[dict[str, SharedMemory], tuple[int, ...], torch.dtype, MLAConfig]:
        """初始化CPU KV缓存"""
        # 根据配置和可用内存计算块数量
        layer = next(iter(kv_cache_specs.values()))
        available_memory = self.available_memory // self.world_size // len(kv_cache_specs)
        num_blocks = available_memory // layer.page_size_bytes
        
        # 创建共享内存
        layer_size = (2, num_blocks, layer.block_size, layer.num_kv_heads, layer.head_size)
        nbytes = math.prod(layer_size) * get_dtype_size(layer.dtype)
        
        shared_memory_dict = {}
        for layer_name in kv_cache_specs:
            shared_memory_dict[layer_name] = MetadataServer._safe_create_shared_memory(
                f"cpu_kv_cache_{pp_rank}_{tp_rank}_{layer_name}", nbytes
            )
        
        return shared_memory_dict, layer_size, layer.dtype, None
```

**关键问题解答**:
- **Q: 如何识别和复用前缀？**
  A: 通过以下机制实现：
     1. **元数据服务器**：集中管理所有请求的前缀元数据
     2. **前缀匹配算法**：快速查找请求间的公共前缀
     3. **共享内存缓存**：将KV缓存存储在共享内存中，实现跨进程复用
     4. **引用计数**：跟踪前缀的使用情况，实现高效的内存管理
  
- **Q: 缓存命中率如何优化？**
  A: 命中率优化策略：
     1. **LRU淘汰策略**：优先淘汰最久未使用的前缀
     2. **多级缓存**：结合GPU和CPU缓存，平衡速度和容量
     3. **批量处理**：将具有相同前缀的请求批处理
     4. **动态调整**：根据请求模式动态调整缓存大小和策略

**性能特点**:
- 减少重复计算，提高吞吐量
- 支持超长上下文的高效处理
- 跨进程前缀共享，减少内存占用
- 自适应缓存管理，平衡性能和资源

---

#### chunked_prefill（分块预填充）

**概述**:
- **Chunked-Prefill (Split-Fuse)**：将长序列的预填充阶段分为多个块处理
- **核心功能**：实现全量增量同时推理，提高GPU资源利用率
- **性能收益**：显著提升长上下文场景下的吞吐量

**实现原理**:
1. **序列分块**：将长序列拆分为多个固定大小的块
2. **并行处理**：多个块并行处理，充分利用GPU计算资源
3. **结果合并**：将各块的处理结果合并为完整序列
4. **动态调度**：根据序列长度和可用资源动态调整块大小

**关键优势**:
- **资源利用率提升**：避免长序列预填充时的GPU资源闲置
- **延迟优化**：减少长序列的整体处理时间
- **内存管理**：降低峰值内存占用，支持更长的上下文
- **兼容性**：与现有动态批处理机制无缝集成

---

#### speculative_decoding（投机解码）

**文件概述**:
- 路径: `vllm_ascend/spec_decode/`
- 功能: 投机解码，使用小模型提前预测，大模型验证
- 依赖: torch, torch_npu, vllm.config

**实现原理**:
1. **大小模型协作**：小模型快速生成多个候选tokens，大模型验证
2. **接受机制**：大模型验证并接受或拒绝候选tokens
3. **并行计算**：小模型和大模型并行计算，提高吞吐量
4. **动态调整**：根据接受率动态调整候选token数量

**核心实现**:
```python
# 简化的投机解码流程
def speculative_decoding(large_model, small_model, input_ids, max_tokens):
    generated_tokens = []
    current_ids = input_ids
    
    while len(generated_tokens) < max_tokens:
        # 小模型生成候选tokens
        small_output = small_model(current_ids, max_new_tokens=4)  # 生成4个候选
        candidate_tokens = small_output.sequences[0, -4:]
        
        # 大模型验证候选tokens
        large_input = torch.cat([current_ids, candidate_tokens], dim=1)
        large_output = large_model(large_input)
        
        # 计算接受概率和选择接受数量
        acceptance_mask = compute_acceptance_mask(large_output.logits)
        num_accepted = compute_num_accepted(acceptance_mask)
        
        # 更新生成的tokens
        if num_accepted > 0:
            accepted_tokens = candidate_tokens[:num_accepted]
            generated_tokens.extend(accepted_tokens.tolist())
            current_ids = torch.cat([current_ids, accepted_tokens], dim=1)
        else:
            # 没有接受任何token，使用大模型的第一个预测
            next_token = large_output.sequences[0, -1:]
            generated_tokens.append(next_token.item())
            current_ids = torch.cat([current_ids, next_token], dim=1)
    
    return generated_tokens
```

**关键问题解答**:
- **Q: 大小模型投机如何实现？**
  A: 通过以下步骤实现：
     1. **模型选择**：选择一个小模型（如原模型的蒸馏版本）和一个大模型
     2. **候选生成**：小模型基于当前上下文生成多个候选tokens
     3. **并行验证**：大模型并行验证所有候选tokens
     4. **接受决策**：根据验证结果决定接受多少个候选tokens
  
- **Q: 接受率如何优化？**
  A: 接受率优化策略：
     1. **候选数量调整**：根据历史接受率动态调整候选token数量
     2. **小模型优化**：选择与大模型行为接近的小模型
     3. **温度调整**：调整小模型的生成温度，平衡多样性和准确性
     4. **长度惩罚**：对候选序列长度进行惩罚，避免过长序列

**性能特点**:
- 吞吐量提升2-3倍
- 延迟降低50%以上
- 精度损失可接受（通常小于0.5%）
- 支持动态调整候选数量

**适用场景**:
- 高吞吐量要求的生成任务
- 对延迟敏感的应用
- 长文本生成场景

---

## 三、架构图

### 3.1 图模式实现对比

```
┌─────────────────────────────────────────────────────────────────┐
│                        图模式特性对比                            │
├──────────────┬──────────────────┬───────────────────────────────┤
│    特性       │      特点        │          性能收益             │
├──────────────┼──────────────────┼───────────────────────────────┤
│ Turbo-Graph  │ 整图下沉，动态shape│ Decode吞吐翻倍               │
│  Acl-Graph   │ 分段图执行        │ 相比eager提升40-60%          │
│Torch.Compile │ Torch dynamo构图  │ 推理性能提升30-50%           │
└──────────────┴──────────────────┴───────────────────────────────┘
```

---

## 四、与vLLM原版差异

| 模块 | vLLM原版 | vLLM-Ascend | 差异说明 |
|------|----------|-------------|----------|
| Graph Mode | CUDA Graph | Turbo/Acl Graph | 待补充 |
| Speculative | CUDA实现 | NPU实现 | 待补充 |

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

1. 分布式特性
