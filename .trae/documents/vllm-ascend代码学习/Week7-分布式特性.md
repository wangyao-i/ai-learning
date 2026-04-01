# vLLM-Ascend 框架代码学习 - Week 7

> 学习主题：分布式特性
> 学习目标：理解TP/PP/EP/CP实现，掌握分布式策略选择

---

## 一、学习进度

| 日期 | 内容 | 状态 |
|------|------|------|
| Day 43-44 | Tensor Parallelism | 待开始 |
| Day 45-46 | Pipeline Parallelism | 待开始 |
| Day 47-48 | Expert Parallelism | 待开始 |
| Day 49 | 本周复盘 | 待开始 |

---

## 二、代码阅读笔记

### 2.1 Tensor Parallelism (张量并行)

#### distributed/parallel_state.py

**文件概述**:
- 路径: `vllm_ascend/distributed/parallel_state.py`
- 功能: 张量并行组的初始化和管理
- 依赖: torch, vllm.config, vllm.distributed.parallel_state

**核心实现**:
```python
def init_ascend_model_parallel(parallel_config: ParallelConfig):
    """初始化昇腾模型并行组"""
    if model_parallel_initialized():
        return
    
    # 获取全局并行配置
    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)
    global_tp_size = parallel_config.tensor_parallel_size
    global_dp_size = parallel_config.data_parallel_size
    global_pp_size = parallel_config.pipeline_parallel_size
    global_pcp_size = parallel_config.prefill_context_parallel_size
    
    # 构建所有rank的网格布局
    all_ranks = torch.arange(world_size).reshape(
        -1, global_dp_size, global_pp_size, global_pcp_size, global_tp_size
    )
    
    # 初始化细粒度张量并行组
    # 1. MLP层张量并行
    # 2. 输出投影层张量并行
    # 3. LM Head张量并行
    # 4. Embedding层张量并行
    
    def _create_or_get_group(group_size: int, group_name: str) -> GroupCoordinator:
        """创建或获取并行组"""
        if group_size is None:
            return None
        if group_size not in _group_cache:
            rank_grid = torch.arange(world_size).reshape(global_pp_size, global_dp_size, global_tp_size)
            num_chunks = global_dp_size // group_size
            group_ranks = []
            for pp_idx in range(global_pp_size):
                stage_ranks = rank_grid[pp_idx]  # (dp, tp)
                for chunk in range(num_chunks):
                    for tp_idx in range(global_tp_size):
                        group = stage_ranks[chunk * group_size : (chunk + 1) * group_size, tp_idx].tolist()
                        group_ranks.append(group)
            pg = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name=group_name)
            _group_cache[group_size] = pg
        return _group_cache[group_size]
    
    # 初始化各组件的张量并行组
    otp_size = get_ascend_config().finegrained_tp_config.oproj_tensor_parallel_size
    lmhead_tp_size = get_ascend_config().finegrained_tp_config.lmhead_tensor_parallel_size
    embedding_tp_size = get_ascend_config().finegrained_tp_config.embedding_tensor_parallel_size
    mlp_tp_size = get_ascend_config().finegrained_tp_config.mlp_tensor_parallel_size
    
    global _OTP, _LMTP, _EMBED_TP, _MLP_TP
    
    if otp_size > 0:
        _OTP = _create_or_get_group(otp_size, "otp")
    if lmhead_tp_size > 0:
        _LMTP = _create_or_get_group(lmhead_tp_size, "lmheadtp")
    if embedding_tp_size > 0:
        _EMBED_TP = _create_or_get_group(embedding_tp_size, "emtp")
    if mlp_tp_size > 0:
        _MLP_TP = _create_or_get_group(mlp_tp_size, "mlptp")
```

**关键问题解答**:
- **Q: 如何切分权重？**
  A: 权重切分策略：
     1. **列切分**：将权重矩阵按列维度切分，用于Q、K、V投影层和MLP的第一个线性层
     2. **行切分**：将权重矩阵按行维度切分，用于输出投影层和MLP的第二个线性层
     3. **细粒度切分**：支持对不同组件（MLP、O Proj、LM Head、Embedding）使用不同的TP大小
     4. **自动切分**：根据配置自动计算切分方式和大小
  
- **Q: 列切分和行切分的差异？**
  A: 主要差异：
     | 特性 | 列切分 | 行切分 |
     |------|--------|--------|
     | 切分维度 | 输入维度 | 输出维度 |
     | 通信操作 | AllReduce（前向） | AllGather（前向） |
     | 适用层 | Q/K/V投影、MLP第一线性层 | 输出投影、MLP第二线性层 |
     | 内存占用 | 减少输入维度的内存 | 减少输出维度的内存 |

---

#### distributed/device_communicators/npu_communicator.py

**文件概述**:
- 路径: `vllm_ascend/distributed/device_communicators/npu_communicator.py`
- 功能: 昇腾NPU的通信实现
- 依赖: torch, torch_npu, vllm.distributed

**核心实现**:
```python
# HCCL通信实现
def allreduce(tensor: torch.Tensor, group: GroupCoordinator = None, async_op: bool = False) -> torch.Tensor:
    """执行AllReduce操作"""
    if group is None:
        group = get_tp_group()
    
    # 使用HCCL后端执行AllReduce
    if async_op:
        work = torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group.device_group, async_op=True)
        return work
    else:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group.device_group)
        return tensor

def allgather(tensor_list: list[torch.Tensor], tensor: torch.Tensor, group: GroupCoordinator = None, async_op: bool = False) -> torch.Tensor:
    """执行AllGather操作"""
    if group is None:
        group = get_tp_group()
    
    # 使用HCCL后端执行AllGather
    if async_op:
        work = torch.distributed.all_gather(tensor_list, tensor, group=group.device_group, async_op=True)
        return work
    else:
        torch.distributed.all_gather(tensor_list, tensor, group=group.device_group)
        return tensor
```

**关键问题解答**:
- **Q: 如何实现AllReduce？**
  A: 通过以下方式实现：
     1. **HCCL后端**：使用昇腾HCCL库提供的高效AllReduce实现
     2. **分组通信**：支持在不同的并行组上执行AllReduce
     3. **同步/异步**：支持同步和异步两种执行模式
     4. **自动选择**：根据张量类型和大小自动选择最优的通信算法
  
- **Q: HCCL通信如何优化？**
  A: HCCL通信优化策略：
     1. **通信与计算重叠**：使用异步通信操作重叠通信和计算
     2. **张量融合**：合并小张量通信，减少通信次数
     3. **拓扑感知**：根据集群拓扑选择最优通信路径
     4. **批量通信**：批量处理多个通信请求
     5. **硬件加速**：利用昇腾NPU的硬件通信引擎

**性能特点**:
- 支持细粒度张量并行配置
- 优化的HCCL通信实现
- 支持多种并行组配置
- 高效的内存使用
- 与昇腾硬件深度优化

---

### 2.2 Pipeline Parallelism (流水线并行)

**概述**:
- **功能**：将模型的不同层分配到不同的设备上，实现层间并行计算
- **核心原理**：将完整的模型分割成多个阶段（stage），每个阶段包含连续的几层，并分配到不同的设备
- **微批次**：将大批次输入分割成多个微批次，实现流水线执行

**文件位置**:
vLLM-Ascend基于vLLM的流水线并行实现，主要通过以下方式集成：
- 继承vLLM的PipelineParallelEngine
- 使用昇腾优化的通信后端
- 支持与TP、DP等并行策略的混合使用

**核心配置**:
```python
# 配置示例
parallel_config = ParallelConfig(
    tensor_parallel_size=2,  # 张量并行大小
    pipeline_parallel_size=2,  # 流水线并行大小（阶段数）
    data_parallel_size=1,     # 数据并行大小
    prefill_context_parallel_size=1  # 预填充上下文并行大小
)
```

**关键问题解答**:
- **Q: 如何切分层？**
  A: 层切分策略：
     1. **连续层切分**：将模型的层按顺序连续切分到不同阶段
     2. **平衡切分**：根据每层的计算量和内存占用自动平衡各阶段的负载
     3. **自定义切分**：支持用户指定层到阶段的映射关系
     4. **混合并行**：支持与TP、DP等并行策略的混合使用
  
- **Q: 微批次如何调度？**
  A: 微批次调度机制：
     1. **1F1B调度**：使用1F1B（One Forward, One Backward）调度算法，最大化流水线利用率
     2. **自动批处理**：根据设备内存和计算能力自动调整微批次大小
     3. **重叠通信**：利用异步通信重叠不同阶段间的数据传输
     4. **动态调整**：根据运行时负载动态调整微批次数量

**实现原理**:
```
阶段0 (设备0): [Embedding层] -> [Transformer层1-8]
       ↓
阶段1 (设备1): [Transformer层9-16] -> [LM Head]

微批次调度流程:
1. 微批次1在阶段0执行前向计算
2. 微批次1转移到阶段1执行前向计算，同时微批次2在阶段0执行前向计算
3. 微批次1在阶段1执行反向计算，同时微批次2转移到阶段1执行前向计算
4. 微批次1的梯度从阶段1转移到阶段0，同时微批次2在阶段1执行反向计算
5. 微批次1在阶段0执行反向计算，同时微批次2的梯度从阶段1转移到阶段0
```

**性能特点**:
- 支持大规模模型的分布式训练和推理
- 平衡各设备的计算负载和内存占用
- 优化的阶段间通信
- 支持与其他并行策略混合使用
- 自动调整微批次大小以最大化性能

**使用场景**:
- 超大模型（百亿/千亿参数）的分布式部署
- 内存受限场景下的模型推理
- 需要与TP、DP混合使用的场景

---

### 2.3 Expert Parallelism (专家并行)

#### eplb/core/policy/policy_default_eplb.py

**文件概述**:
- 路径: `vllm_ascend/eplb/core/policy/policy_default_eplb.py`
- 功能: 默认专家负载均衡策略实现
- 依赖: numpy, collections

**核心实现**:
```python
class DefaultEplb(EplbPolicy):
    """默认专家负载均衡策略"""
    
    def rebalance_experts(self, current_expert_table, expert_workload):
        """重新平衡专家分布"""
        info = DynamicTable()
        info.workload_table = np.array(expert_workload)
        info.placement_table = np.array(current_expert_table)
        
        # 获取当前配置
        layer_num, num_npus, experts_per_npu = info.workload_table.shape
        expert_ids, counts = np.unique(info.placement_table[0], return_counts=True)
        num_redundancy_expert = self.get_redundant_num(num_npus, counts)
        num_original_expert = len(expert_ids)
        
        # 计算每层的专家负载
        layer_workloads = self.add_redundant(info.placement_table, info.workload_table, num_original_expert)
        max_heat_per_layer_before = self.calculate_max_heat_per_layer(info.workload_table, layer_num)
        npu_heat_all_origin = sum(max_heat_per_layer_before)
        
        # 为每层执行负载均衡
        global_deployment = [[[] for _ in range(num_npus)] for _ in range(layer_num)]
        max_heat_per_layer_after = np.zeros([layer_num])
        
        for layer in range(layer_num):
            # 获取当前层的专家ID和对应的负载
            weights = np.zeros((num_original_expert,), dtype="object")
            for expert_id, workload_weight in enumerate(layer_workloads[layer]):
                weights[expert_id] = (expert_id, workload_weight)
            
            # 执行负载均衡算法
            result, layer_deployment = self.original_compute_balanced_pack_redundancy(
                weights, num_npus, num_redundancy_expert
            )
            
            global_deployment[layer] = layer_deployment
            max_heat_per_layer_after[layer] = max(result, key=lambda x: x["total_weight"])["total_weight"]
        
        # 约束专家本地交换
        new_global_deployment = self.constraint_expert_local_exchange(current_expert_table, global_deployment)
        
        # 计算层变化比例
        layer_changed_ratio = []
        for layer_idx in range(layer_num):
            layer_changed_ratio.append(max_heat_per_layer_after[layer_idx] / max_heat_per_layer_before[layer_idx])
        
        per_layer_priority = np.argsort(layer_changed_ratio)
        npu_heat_all_after = sum(max_heat_per_layer_after)
        
        # 决定是否需要改变专家分布
        change = 0
        if npu_heat_all_after < 0.95 * npu_heat_all_origin:
            change = 1
        
        return change, per_layer_priority, np.array(new_global_deployment).tolist()
    
    @staticmethod
    def compute_balanced_pack_redundancy(origin_weights, card_num, num_redundancy_expert):
        """计算带有冗余专家的负载均衡分布"""
        # 步骤1: 为热专家创建冗余专家
        # 步骤2: 计算每卡专家数量
        # 步骤3: 初始化专家分布
        # 步骤4: 按负载分配专家
        # 步骤5: 返回分布结果
        # ... 具体实现见代码
```

**关键问题解答**:
- **Q: 如何实现专家负载均衡？**
  A: 负载均衡实现策略：
     1. **热专家拆分**：将负载过高的专家拆分为多个冗余专家
     2. **权重排序**：按专家负载权重降序排序
     3. **贪心分配**：将专家分配到负载最低的设备上
     4. **约束优化**：考虑专家本地交换的约束条件
     5. **动态调整**：根据负载变化动态调整专家分布
  
- **Q: All2All通信如何优化？**
  A: All2All通信优化：
     1. **专家分组**：将相关专家分组，减少通信量
     2. **批量通信**：合并多个All2All操作
     3. **硬件加速**：利用昇腾NPU的硬件通信引擎
     4. **异步通信**：重叠通信和计算
     5. **拓扑感知**：根据集群拓扑优化通信路径

#### eplb/core/policy/policy_factory.py

**文件概述**:
- 路径: `vllm_ascend/eplb/core/policy/policy_factory.py`
- 功能: 负载均衡策略工厂，管理不同的负载均衡策略
- 依赖: policy_abstract, policy_default_eplb, policy_flashlb, policy_random, policy_swift_balancer

**核心实现**:
```python
class EplbPolicyFactory:
    """专家负载均衡策略工厂"""
    
    @staticmethod
    def create_policy(policy_name: str, config: DynamicConfig) -> EplbPolicy:
        """创建指定类型的负载均衡策略"""
        if policy_name == "default_eplb":
            return DefaultEplb(config)
        elif policy_name == "flashlb":
            return Flashlb(config)
        elif policy_name == "random":
            return RandomEplb(config)
        elif policy_name == "swift_balancer":
            return SwiftBalancer(config)
        else:
            raise ValueError(f"Unknown policy name: {policy_name}")
```

**关键问题解答**:
- **Q: 动态偏置路由如何实现？**
  A: 动态偏置路由实现：
     1. **路由监控**：实时监控每个专家的负载情况
     2. **动态调整**：根据专家负载动态调整路由概率
     3. **流量引导**：将请求引导到负载较低的专家
     4. **平滑过渡**：避免路由突变导致的性能波动
  
- **Q: 专家负载如何监控？**
  A: 专家负载监控机制：
     1. **实时统计**：统计每个专家的请求次数和处理时间
     2. **热图生成**：生成专家负载热图
     3. **阈值触发**：当负载超过阈值时触发重平衡
     4. **历史分析**：分析历史负载模式，预测未来负载

**性能特点**:
- 支持多种负载均衡策略
- 动态调整专家分布
- 支持冗余专家提高容错性
- 优化的All2All通信
- 与昇腾硬件深度集成

**适用场景**:
- MoE（Mixture of Experts）模型
- 专家负载不均衡的场景
- 需要高吞吐量的生成任务
- 大规模分布式部署

---

### 2.4 Context Parallelism (上下文并行)

#### attention/context_parallel/attention_cp.py

**文件概述**:
- 路径: `vllm_ascend/attention/context_parallel/attention_cp.py`
- 功能: 上下文并行注意力机制实现
- 依赖: torch, torch.distributed, torch_npu, vllm.config, vllm.distributed

**核心实现**:
```python
class AscendAttentionCPImpl(AscendAttentionBackendImpl):
    """支持上下文并行的注意力实现"""
    
    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int, **kwargs):
        super().__init__(num_heads, head_size, scale, num_kv_heads, **kwargs)
        self.pcp_size = get_pcp_group().world_size  # 预填充上下文并行大小
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.pcp_group = get_pcp_group().device_group if self.pcp_size > 1 else None
        
        self.dcp_size = get_decode_context_model_parallel_world_size()  # 解码上下文并行大小
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0
        self.dcp_group = get_dcp_group().device_group if self.dcp_size > 1 else None
    
    def _forward_prefill_cp(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata) -> torch.Tensor:
        """执行预填充阶段的上下文并行注意力计算"""
        # 步骤1: 预处理QKV，分为头部和尾部
        data_head, data_tail = self._forward_prefill_cp_pre(query, key, value, attn_metadata)
        
        # 步骤2: 对头部和尾部分别执行注意力计算
        output_head, lse_head = self._forward_prefill_cp_attn(data_head, True, attn_metadata)
        output_tail, lse_tail = self._forward_prefill_cp_attn(data_tail, False, attn_metadata)
        
        # 步骤3: 合并结果
        output, attn_lse = self._forward_prefill_cp_post(
            [output_head, output_tail],
            [lse_head, lse_tail],
            attn_metadata,
        )
        return output, attn_lse
    
    def _forward_decode_pcp_dcp(self, query: torch.Tensor, attn_metadata: AscendMetadata) -> torch.Tensor:
        """执行解码阶段的PCP/DCP并行注意力计算"""
        # DCP处理：按头维度并行
        if self.dcp_size > 1:
            query = get_dcp_group().all_gather(query.contiguous(), 1)
            num_heads = self.num_heads * self.dcp_size
        else:
            num_heads = self.num_heads
        
        # 执行融合注意力计算
        common_kwargs = {
            "num_heads": num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "input_layout": "TND",
            "atten_mask": None,
            "scale": self.scale,
            "softmax_lse_flag": True,
            "block_table": attn_metadata.decode_meta.block_tables,
            "block_size": self.key_cache.shape[1],
            "actual_seq_lengths_kv": attn_metadata.decode_meta.num_computed_tokens_of_pcp_dcp[:, self.pcp_rank, self.dcp_rank],
            "actual_seq_lengths": torch.arange(attn_metadata.num_decodes_flatten) + 1,
        }
        
        attn_out, attn_lse = torch_npu.npu_fused_infer_attention_score(query, self.key_cache.view(self.key_cache.shape[0], self.key_cache.shape[1], -1), 
                                                                       self.value_cache.view(self.key_cache.shape[0], self.key_cache.shape[1], -1), 
                                                                       **common_kwargs)
        
        # 处理注意力输出和LSE
        attn_out_lse = _process_attn_out_lse(attn_out, attn_lse)
        attn_out = _npu_attention_update(self.head_size, attn_out_lse)
        return attn_out
```

**关键问题解答**:
- **Q: 如何切分长序列？**
  A: 长序列切分策略：
     1. **上下文维度切分**：将长序列按上下文维度切分为多个片段
     2. **头部尾部划分**：将序列分为头部（共享部分）和尾部（私有部分）
     3. **块大小控制**：根据设备内存和计算能力动态调整块大小
     4. **重叠处理**：处理块之间的重叠部分，确保注意力计算的正确性
  
- **Q: 序列维度如何并行？**
  A: 序列维度并行实现：
     1. **PCP (Prefill Context Parallelism)**：预填充阶段的上下文并行
     2. **DCP (Decode Context Parallelism)**：解码阶段的上下文并行
     3. **混合并行**：同时支持PCP和DCP
     4. **环形通信**：使用环形通信模式减少通信开销

**实现原理**:
```
上下文并行处理流程：
1. 序列切分：将长序列按上下文维度切分为多个片段
2. 并行计算：不同设备处理不同的序列片段
3. 通信合并：通过AllGather/AllReduce等操作合并计算结果
4. 结果恢复：恢复完整的序列表示

预填充阶段(PCP)：
- 将查询分为头部和尾部
- 对每个部分执行注意力计算
- 合并结果并恢复完整序列

解码阶段(DCP)：
- 按头维度并行计算
- 使用环形通信减少开销
- 合并结果生成最终输出
```

**性能特点**:
- 支持超长上下文处理（可达百万级tokens）
- 内存使用与序列长度线性增长
- 通信开销优化
- 与现有注意力机制无缝集成
- 支持与TP/PP/DP混合使用

**适用场景**:
- 超长文本生成
- 文档摘要和理解
- 多轮对话
- 需要处理长上下文的复杂任务

---

## 三、架构图

### 3.1 分布式并行对比

```
┌─────────────────────────────────────────────────────────────────┐
│                        分布式并行对比                            │
├──────────────┬──────────────────┬───────────────────────────────┤
│    特性       │      切分方式     │          通信方式             │
├──────────────┼──────────────────┼───────────────────────────────┤
│     TP       │ 层内切分（权重）   │ HCCL AllReduce               │
│     PP       │ 层间切分（层）     │ NPU间点对点通信               │
│     EP       │ 专家切分          │ All2All                      │
│     CP       │ 序列维度切分      │ Ring-Attention               │
└──────────────┴──────────────────┴───────────────────────────────┘
```

---

## 四、与vLLM原版差异

| 模块 | vLLM原版 | vLLM-Ascend | 差异说明 |
|------|----------|-------------|----------|
| TP | NCCL AllReduce | HCCL AllReduce | 待补充 |
| PP | CUDA Stream | NPU Stream | 待补充 |
| EP | CUDA实现 | NPU实现 | 待补充 |

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

1. MoE优化与通信
