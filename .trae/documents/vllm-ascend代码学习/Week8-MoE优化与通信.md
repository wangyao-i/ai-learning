# vLLM-Ascend 框架代码学习 - Week 8

> 学习主题：MoE优化与通信
> 学习目标：理解MoE核心实现，掌握通信优化策略，理解PD分离部署

---

## 一、学习进度

| 日期 | 内容 | 状态 |
|------|------|------|
| Day 50-51 | MoE核心实现 | 待开始 |
| Day 52-53 | 通信优化 | 待开始 |
| Day 54-55 | PD分离部署与性能调优 | 待开始 |
| Day 56 | 本周复盘与总结 | 待开始 |

---

## 二、代码阅读笔记

### 2.1 MoE核心实现

#### ops/fused_moe/fused_moe.py

**文件概述**:
- 路径: `vllm_ascend/ops/fused_moe/fused_moe.py`
- 功能: 融合MoE实现，包括专家网络、路由和通信优化
- 依赖: torch, torch_npu, vllm.config, vllm.distributed, vllm_ascend.eplb

**核心类**:
```python
class AscendFusedMoE(FusedMoE):
    """昇腾平台的融合MoE实现"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化MoE配置
        num_experts = kwargs["num_experts"]
        intermediate_size = kwargs["intermediate_size"]
        num_shared_experts = kwargs.get("n_shared_experts", 0)
        
        # 设置并行组
        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.mc2_group = get_mc2_group()
        
        # 初始化EPLB配置（专家负载均衡）
        eplb_config = get_ascend_config().eplb_config
        self.mix_placement = getattr(ascend_config, "mix_placement", False)
        self.n_shared_experts = num_shared_experts
        num_experts += num_shared_experts if self.mix_placement else 0
        self.moe_config.num_experts = num_experts
        
        # 初始化专家映射和负载均衡
        self.global_expert_map, self._expert_map, self.log2phy, self.global_redundant_expert_num = init_eplb_config(
            eplb_config, self.moe_instance_id, self.moe_config, self.mix_placement, num_shared_experts
        )
        self.global_num_experts = num_experts + self.global_redundant_expert_num
        self.dynamic_eplb = eplb_config.dynamic_eplb and (self.log2phy is not None)
        self.local_num_experts = self.global_num_experts // self.ep_size
        
        # 设置量化方法和运行器
        setup_moe_comm_method(self.moe_config)
        self.quant_type = self._get_quant_type()
        self.runner = self._init_runner()
    
    def forward_impl(self, hidden_states: torch.Tensor, router_logits: torch.Tensor, return_with_event: bool = False) -> torch.Tensor | FusedMoEResult:
        """MoE前向传播实现"""
        # 准备阶段
        prepare_output = _EXTRA_CTX.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=_EXTRA_CTX.flash_comm_v1_enabled,
            enable_shared_expert_dp=self.enable_shared_expert_dp,
            quant_type=self.quant_type,
        )
        
        # 专家计算
        fused_experts_results = self.quant_method.apply(
            layer=self,
            x=prepare_output.hidden_states,
            router_logits=prepare_output.router_logits,
            pertoken_scale=prepare_output.pertoken_scale,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self._expert_map,
            # ... 其他参数
        )
        
        # 负载均衡更新
        if self.dynamic_eplb:
            expert_tokens = fused_experts_results.expert_tokens
            group_list_type = fused_experts_results.group_list_type
            local_load = expert_tokens if group_list_type == 1 else torch.cat([expert_tokens[:1], expert_tokens[1:] - expert_tokens[:-1]])
            self.moe_load.add_(local_load)
        
        # 最终处理
        routed_out = _EXTRA_CTX.moe_comm_method.finalize(
            hidden_states=fused_experts_results.routed_out,
            reduce_results=self.reduce_results,
            padded_hidden_states_shape=prepare_output.padded_hidden_states_shape,
        )
        
        return routed_out
```

```python
class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """非量化的MoE方法实现"""
    
    def apply(self, layer: torch.nn.Module, x: torch.Tensor, router_logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """应用MoE计算"""
        # 专家选择
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=kwargs["top_k"],
            use_grouped_topk=kwargs["use_grouped_topk"],
            renormalize=kwargs["renormalize"],
            # ... 其他参数
        )
        
        # 零专家处理
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)
        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=kwargs["global_num_experts"],
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )
        
        # 融合专家计算
        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                w1_bias=layer.w13_bias if self.moe.has_bias else None,
                w2_bias=layer.w2_bias if self.moe.has_bias else None,
                quant_type=QuantType.NONE,
                # ... 其他参数
            )
        )
        
        # 合并零专家结果
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        
        return final_hidden_states
```

**关键问题解答**:
- **Q: 如何实现专家网络？**
  A: 专家网络实现：
     1. **双层MLP结构**：每个专家是一个两层MLP网络（w13和w2权重）
     2. **权重格式化**：权重转换为NZ格式以支持融合操作
     3. **零专家支持**：支持零专家（不进行计算的专家）以优化负载均衡
     4. **共享专家**：支持共享专家，多个MoE层共享部分专家网络
  
- **Q: 细粒度专家分工如何实现？**
  A: 细粒度专家分工：
     1. **动态负载均衡**：基于EPLB（Expert Parallel Load Balance）实现动态专家分配
     2. **专家映射**：使用log2phy映射表管理专家的物理位置
     3. **冗余专家**：为热点专家创建冗余副本，分散负载
     4. **混合放置**：支持共享专家和私有专家的混合放置策略

#### ops/fused_moe/experts_selector.py

**文件概述**:
- 路径: `vllm_ascend/ops/fused_moe/experts_selector.py`
- 功能: 专家选择和路由实现
- 依赖: torch, torch_npu

**核心实现**:
```python
def select_experts(hidden_states: torch.Tensor, router_logits: torch.Tensor, top_k: int, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """选择Top-K专家"""
    # 路由选择逻辑
    # ...
    return topk_weights, topk_ids

def zero_experts_compute(expert_indices: torch.Tensor, expert_scales: torch.Tensor, num_experts: int, 
                        zero_expert_type: str, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """零专家计算"""
    # 零专家处理逻辑
    # ...
    return expert_indices, expert_scales, zero_expert_result
```

**关键问题解答**:
- **Q: 如何实现Top-K路由？**
  A: Top-K路由实现：
     1. **路由网络**：使用神经网络生成专家选择概率
     2. **Top-K选择**：为每个token选择概率最高的K个专家
     3. **权重归一化**：对选择的专家权重进行归一化
     4. **分组Top-K**：支持分组Top-K以优化性能
  
- **Q: 动态偏置路由如何实现？**
  A: 动态偏置路由：
     1. **偏置校正**：使用e_score_correction_bias校正专家选择偏置
     2. **负载监控**：实时监控专家负载
     3. **动态调整**：根据负载动态调整路由概率
     4. **平滑过渡**：避免路由突变导致的性能波动

**融合策略**:
1. **计算融合**：将专家选择、权重查找和矩阵乘法融合为单个操作
2. **通信融合**：将All2All通信与计算重叠
3. **内存融合**：优化内存访问模式，减少数据移动
4. **多流并行**：使用多流技术并行执行不同阶段的计算

**性能特点**:
- 支持多种专家并行策略
- 动态负载均衡提高资源利用率
- 融合操作减少计算和通信开销
- 支持共享专家和私有专家
- 与昇腾硬件深度优化

---

### 2.2 通信优化

#### ops/fused_moe/moe_comm_method.py

**文件概述**:
- 路径: `vllm_ascend/ops/fused_moe/moe_comm_method.py`
- 功能: MoE通信方法实现，包括AllGather、All2All和MC2等通信策略
- 依赖: torch, torch_npu, vllm.model_executor.layers.fused_moe

**核心实现**:
```python
class MoECommMethod(ABC):
    """MoE通信方法基类"""
    
    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config
        self.token_dispatcher = self._get_token_dispatcher()
        self.prepare_finalize = self._get_prepare_finalize()
        self.use_fusion_ops = set_gmmswigluquant_method()
    
    def fused_experts(self, fused_experts_input: MoEFusedExpertsInput) -> FusedExpertsResult:
        """融合专家计算"""
        # 1. 令牌分发前的准备
        before_dispatch_evt = torch.npu.current_stream().record_event()
        routed_topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            routed_topk_ids = fused_experts_input.routing.log2phy[routed_topk_ids]
        
        # 2. 令牌分发
        token_dispatch_input = build_token_dispatch_input(
            fused_experts_input=fused_experts_input,
            topk_ids=routed_topk_ids,
        )
        token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)
        
        # 3. MLP计算
        mlp_compute_input = build_mlp_compute_input(
            fused_experts_input=fused_experts_input,
            token_dispatch_output=token_dispatch_output,
            use_fusion_ops=self.use_fusion_ops,
        )
        mlp_output = self._apply_mlp(mlp_compute_input)
        
        # 4. 令牌合并
        before_combine_evt = torch.npu.current_stream().record_event()
        routed_out = self.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            combine_metadata=token_dispatch_output.combine_metadata,
        )
        
        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=token_dispatch_output.group_list_type,
            expert_tokens=token_dispatch_output.group_list,
        )
```

```python
class AlltoAllCommImpl(MoECommMethod):
    """使用All2All通信的MoE实现"""
    
    def _get_token_dispatcher(self):
        return TokenDispatcherWithAll2AllV(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )
    
    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)

class MC2CommImpl(MoECommMethod):
    """使用MC2（通信计算并行）的MoE实现"""
    
    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()
    
    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)

class FusedMC2CommImpl(MoECommMethod):
    """融合的MC2通信实现"""
    
    def fused_experts(self, fused_experts_input: MoEFusedExpertsInput):
        # 使用昇腾融合操作dispatch_ffn_combine或dispatch_gmm_combine_decode
        if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
            out = torch.empty_like(fused_experts_input.hidden_states)
            torch.ops._C_ascend.dispatch_ffn_combine(
                x=fused_experts_input.hidden_states,
                weight1=fused_experts_input.weights.w1,
                weight2=fused_experts_input.weights.w2,
                expert_idx=topk_ids,
                scale1=fused_experts_input.weights.w1_scale,
                scale2=fused_experts_input.weights.w2_scale,
                probs=fused_experts_input.topk_weights.to(torch.float32),
                group=self.token_dispatcher.moe_all_to_all_group_name,
                max_output_size=65536,
                out=out,
                expert_token_nums=self.expert_token_nums,
            )
        elif envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 2:
            out, expert_tokens = torch.ops._C_ascend.dispatch_gmm_combine_decode(
                x=fused_experts_input.hidden_states,
                expert_ids=topk_ids,
                gmm1_permuted_weight=fused_experts_input.weights.w1,
                gmm1_permuted_weight_scale=fused_experts_input.weights.w1_scale,
                gmm2_weight=fused_experts_input.weights.w2,
                gmm2_weight_scale=fused_experts_input.weights.w2_scale,
                expert_smooth_scales=None,
                expert_scales=fused_experts_input.topk_weights.to(torch.float32),
                group_ep=self.token_dispatcher.moe_all_to_all_group_name,
                ep_rank_size=self.token_dispatcher.ep_world_size,
                ep_rank_id=self.token_dispatcher.ep_rank_id,
                moe_expert_num=self.moe_config.num_experts,
                global_bs=self.token_dispatcher.global_bs,
            )
        return FusedExpertsResult(routed_out=out, expert_tokens=expert_tokens)
```

**关键问题解答**:
- **Q: 如何优化All2All？**
  A: All2All通信优化策略：
     1. **融合操作**：使用融合的All2All操作，减少通信次数
     2. **批量通信**：合并多个小的All2All请求
     3. **拓扑感知**：根据集群拓扑选择最优通信路径
     4. **硬件加速**：利用昇腾NPU的硬件通信引擎
     5. **异步通信**：使用异步All2All操作，重叠通信和计算
  
- **Q: CP通信剪枝如何实现？**
  A: CP通信剪枝实现：
     1. **稀疏路由**：只与选定的专家通信，减少通信量
     2. **令牌过滤**：过滤不需要的令牌，减少通信数据量
     3. **专家分组**：将专家分组，减少All2All的维度
     4. **动态调整**：根据负载动态调整通信策略

**计算通信重叠策略**:
1. **多流并行**：使用多个NPU流并行执行通信和计算操作
2. **事件同步**：使用事件（Event）机制精确控制通信和计算的时序
3. **流水线执行**：将MoE计算分解为多个阶段，流水线执行
4. **异步通信**：使用异步通信API，在通信进行时执行其他计算

**Flash Comm共享专家混置**:
1. **专家混置**：将共享专家和私有专家混合放置，减少通信开销
2. **多流执行**：使用独立的流执行共享专家计算
3. **重叠通信**：将共享专家计算与其他通信操作重叠
4. **结果缓存**：缓存共享专家的计算结果，避免重复计算

**性能特点**:
- 支持多种通信策略（AllGather、All2All、MC2）
- 融合操作减少通信和计算开销
- 计算通信重叠隐藏通信延迟
- 与昇腾硬件深度优化
- 支持动态调整通信策略

**适用场景**:
- 大规模MoE模型训练和推理
- 高通信压力的场景
- 对延迟敏感的应用
- 需要高吞吐量的生成任务

---

### 2.3 PD分离部署

#### examples/epd_disaggregated/epd_disaggregated_guide.md

**文件概述**:
- 路径: `vllm_ascend/examples/epd_disaggregated/epd_disaggregated_guide.md`
- 功能: EPD（Encoder-Prefill-Decode）分离部署指南
- 依赖: Python >= 3.10, CANN == 8.5.0, PyTorch == 2.8.0, torch-npu == 2.8.0

**核心概念**:
EPD分离部署将模型推理过程分解为三个阶段，在不同的节点上执行：
1. **Encoder阶段**：负责视觉编码计算（如处理图像输入）
2. **Prefill阶段**：负责文本前缀填充和初始注意力计算
3. **Decode阶段**：负责自回归解码生成后续token

**部署架构**:
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   Encoder节点   ├─────►   Prefill节点   ├─────►   Decode节点   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
          ▲                                               │
          │                                               ▼
┌─────────────────┐                                 ┌─────────────────┐
│                 │                                 │                 │
│   Proxy节点     │◄─────────────────────────────────┤   客户端请求    │
│                 │                                 │                 │
└─────────────────┘                                 └─────────────────┘
```

**核心配置**:

1. **Encoder节点配置**:
```shell
vllm serve "/your/model/path" \
    --gpu-memory-utilization 0.01 \
    --port "23001" \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_producer",
        "ec_connector_extra_config": {
            "shared_storage_path": "/data/ec_cache"
        }
    }'
```

2. **Prefill节点配置**:
```shell
vllm serve "/your/model/path" \
    --gpu-memory-utilization 0.7 \
    --port "33003" \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_consumer",
        "ec_connector_extra_config": {
            "shared_storage_path": "/data/ec_cache"
        }
    }' \
    --kv-transfer-config '{"kv_connector": "MooncakeLayerwiseConnector",
      "kv_role": "kv_producer",
      "kv_port": "50001",
      "engine_id": "0",
      "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
      "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {
                        "dp_size": 1,
                        "tp_size": 1
                 },
                 "decode": {
                        "dp_size": 1,
                        "tp_size": 1
                 }
          }
      }'
```

3. **Decode节点配置**:
```shell
vllm serve "/your/model/path" \
    --gpu-memory-utilization 0.7 \
    --port "33006" \
    --kv-transfer-config '{"kv_connector": "MooncakeLayerwiseConnector",
        "kv_role": "kv_consumer",
        "kv_port": "50001",
        "engine_id": "1",
        "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
        "kv_connector_extra_config": {
                  "use_ascend_direct": true,
                  "prefill": {
                          "dp_size": 1,
                          "tp_size": 1
                   },
                   "decode": {
                          "dp_size": 1,
                          "tp_size": 1
                    }
                }
        }'
```

4. **Proxy节点配置**:
```shell
python3 epd_load_balance_proxy_layerwise_server_example.py \
    --encoder-hosts 127.0.0.1 \
    --encoder-ports 23001 \
    --prefiller-hosts 127.0.0.1 \
    --prefiller-ports 33003 \
    --decoder-hosts 127.0.0.1 \
    --decoder-ports 33006 \
    --host 127.0.0.1 \
    --port 8001
```

**关键问题解答**:
- **Q: Prefill-Decode如何分离？**
  A: PD分离实现：
     1. **阶段分解**：将推理过程分解为Prefill和Decode两个独立阶段
     2. **节点分离**：在不同的硬件节点上部署Prefill和Decode服务
     3. **KV传输**：使用MooncakeLayerwiseConnector在Prefill和Decode节点之间传输KV缓存
     4. **代理协调**：使用代理节点协调客户端请求和各阶段节点
  
- **Q: 资源利用率如何提升？**
  A: 资源利用率优化策略：
     1. **负载均衡**：根据工作负载动态分配请求到不同节点
     2. **资源隔离**：为不同阶段分配最优的硬件资源
     3. **内存优化**：根据阶段特点调整GPU内存利用率
     4. **批量处理**：在Prefill阶段使用大批次，在Decode阶段优化延迟

**性能特点**:
- **提高吞吐量**：Prefill和Decode阶段可以独立扩展
- **降低延迟**：Decode阶段可以专注于生成延迟优化
- **资源优化**：根据不同阶段的资源需求分配硬件
- **弹性扩展**：可以根据负载动态调整各阶段节点数量

**适用场景**:
- 大规模语言模型推理
- 多模态模型推理（如图文理解）
- 高并发请求场景
- 资源受限的部署环境

---

### 2.4 性能调优实践

#### Profiling工具使用

**关键问题**:
- [ ] 如何使用msprof分析性能？
- [ ] 如何识别计算/内存/通信瓶颈？

#### 调优策略

**关键问题**:
- [ ] 如何选择最优配置？
- [ ] 常见性能问题如何解决？

---

## 三、架构图

### 3.1 MoE优化特性

```
┌─────────────────────────────────────────────────────────────────┐
│                        MoE优化特性                               │
├──────────────────┬──────────────────────────────────────────────┤
│       特性        │                  性能收益                    │
├──────────────────┼──────────────────────────────────────────────┤
│ 细粒度专家分工     │ 参数利用率提升20-30%                        │
│ 共享专家隔离       │ 减少参数冗余40-50%                          │
│ 动态偏置路由       │ 专家负载均衡度提升60%+                      │
│ Flash Comm混置    │ 专家间通信开销降低50-70%                    │
│ CP通信剪枝        │ All-to-All通信性能提升8x                    │
└──────────────────┴──────────────────────────────────────────────┘
```

---

## 四、与vLLM原版差异

| 模块 | vLLM原版 | vLLM-Ascend | 差异说明 |
|------|----------|-------------|----------|
| MoE | CUDA实现 | NPU实现 | 待补充 |
| 通信优化 | NCCL优化 | HCCL优化 | 待补充 |
| PD分离 | 无 | 支持 | 待补充 |

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

---

## 七、8周学习总结

### 7.1 核心收获

1. 待补充

### 7.2 能力提升

1. 待补充

### 7.3 后续计划

1. 待补充
