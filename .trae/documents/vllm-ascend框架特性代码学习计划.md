# vLLM-Ascend 框架特性代码学习计划

> **学习目标**: 深入理解vLLM-Ascend框架代码实现，掌握核心特性原理与代码细节
> **学习周期**: 8周 (可根据实际情况调整)
> **学习方式**: 按文件结构学习框架代码 + 按特性学习特性代码
> **官方文档**: https://docs.vllm.ai/projects/ascend/

---

## 一、vLLM-Ascend 完整特性列表

> 来源: 官方文档 https://docs.vllm.ai/projects/ascend/ 及华为云文档

### 1.1 调度特性

| 特性名称 | 特性说明 | 性能收益 |
|----------|----------|----------|
| **Page-Attention** | 分块管理KV Cache，提升吞吐 | 同等硬件条件下吞吐量提升30-50% |
| **Continuous Batching** | 迭代级调度，动态调整batch，降低延迟，提升吞吐 | 降低延迟15-30%，提升吞吐20-40% |
| **Multi-step** | 一次调度多次推理，降低调度上的CPU开销 | CPU调度开销降低60%+ |

### 1.2 量化特性

| 特性名称 | 特性说明 | 性能收益 |
|----------|----------|----------|
| **W4A16-AWQ** | 权重Int4量化，降低显存消耗和时延 | 小并发时延提升80%，精度损失2%以内，显存占用降低75% |
| **W8A8-SmoothQuant** | 权重Int8量化，激活Int8量化 | 吞吐提升30%，精度损失1.5%以内，显存占用降低50% |
| **W8A16-GPTQ** | 权重Int8量化，激活FP16/BF16 | 吞吐提升20%，精度损失1%以内，显存占用降低50% |
| **KV8** | KV Cache 8位量化 | 吞吐提升15-25%，支持更长上下文长度，显存占用降低50% |

### 1.3 高效解码特性

| 特性名称 | 特性说明 | 性能收益 |
|----------|----------|----------|
| **Auto-Prefix-Caching** | 前缀缓存，降低首token时延 | 首token时延降低40-60%，内存占用优化30%+ |
| **Chunked-Prefill** | 又名Split-Fuse，全量增量同时推理 | 资源利用率提升25-35%，吞吐提升20-30% |
| **Speculative Decoding** | 支持大小模型投机推理和eager模式投机 | 推理性能提升1.5-2.5x |

### 1.4 图模式特性

| 特性名称 | 特性说明 | 性能收益 |
|----------|----------|----------|
| **Ascend-Turbo-Graph** | 记录算子执行的依赖关系构图；消除python host耗时；支持动态shape | Decode吞吐翻倍提升 |
| **Acl-Graph** | 对标cuda-graph的piece-wise graph | 相比eager模式提升40-60% |
| **Torch.Compile** | Torch.dynamo构图，转ascend-GE后端推理 | 编译后推理性能提升30-50% |

### 1.5 控制输出特性

| 特性名称 | 特性说明 | 应用场景 |
|----------|----------|----------|
| **Guided Decoding** | 通过特定模式控制模型输出 | JSON输出、结构化数据生成 |
| **Beam Search** | 通过beam search输出多个候选结果 | 高质量文本生成、翻译 |

### 1.6 实例复用特性

| 特性名称 | 特性说明 | 性能收益 |
|----------|----------|----------|
| **Multi-LoRA** | 多LoRA挂载，多个不同微调模型共用一份权重同时部署 | 显存占用降低60-80%，支持数十个LoRA同时部署 |

### 1.7 分离部署特性

| 特性名称 | 特性说明 | 性能收益 |
|----------|----------|----------|
| **PD分离部署** | Prefill-Decode分离部署，提高资源利用率 | 资源利用率提升30-40%，用户体验提升 |

### 1.8 分布式特性

| 特性名称 | 特性说明 | 通信方式 |
|----------|----------|----------|
| **Tensor Parallelism (TP)** | 模型层内切分，支持多NPU并行计算 | HCCL AllReduce |
| **Pipeline Parallelism (PP)** | 模型层间切分，不同NPU处理不同层 | NPU间点对点通信 |
| **Expert Parallelism (EP)** | MoE模型专家并行，支持DeepSeek等MoE架构 | All2All |
| **Context Parallelism (CP)** | 长序列场景下的序列维度切分，支持超长上下文 | Ring-Attention |

### 1.9 MoE优化特性

| 特性名称 | 特性说明 | 性能收益 |
|----------|----------|----------|
| **细粒度专家分工** | 将传统MoE中的专家进一步拆分为更小的单元 | 参数利用率提升20-30% |
| **共享专家隔离** | 共享专家与路由专家在参数和计算流程上隔离 | 减少参数冗余40-50% |
| **动态偏置路由** | 在路由决策中引入可学习的偏置项 | 专家负载均衡度提升60%+ |
| **Flash Comm共享专家混置** | 通过Flash通信机制实现专家间高效数据共享 | 专家间通信开销降低50-70% |
| **CP特性 (通信剪枝)** | 优化MoE模型中的All-to-All通信 | All-to-All通信性能提升8x |

### 1.10 内存优化特性

| 特性名称 | 特性说明 | 性能收益 |
|----------|----------|----------|
| **NPU感知的Block管理** | 根据NPU内存特性调整block大小 | 内存访问效率提升20-30% |
| **NUMA感知内存分配** | 考虑NPU的NUMA拓扑结构优化内存访问 | 内存访问延迟降低15-25% |

### 1.11 算子优化特性

| 特性名称 | 特性说明 | 性能收益 |
|----------|----------|----------|
| **FlashAttention适配** | 使用TIK DSL开发高效的FlashAttention算子 | Attention计算速度提升2-3x |
| **算子融合策略** | 将多个小算子融合为单个高效算子 | Kernel launch开销降低50%+ |

### 1.12 通信优化特性

| 特性名称 | 特性说明 | 性能收益 |
|----------|----------|----------|
| **零冗余TP转EP通信优化** | 优化张量并行到专家并行的通信转换 | 通信开销降低30-40% |
| **计算通信重叠** | 利用NPU的异步执行能力重叠计算和通信 | 端到端性能提升15-20% |

---

## 二、学习路径总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        vLLM-Ascend 代码学习路径                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1: 框架基础 (Week 1-2)                                           │
│  ├── 项目结构与入口                                                      │
│  ├── 平台抽象层                                                          │
│  └── Worker与模型加载                                                    │
│                                                                         │
│  Phase 2: 核心特性 (Week 3-4)                                           │
│  ├── Page-Attention实现                                                 │
│  ├── Continuous Batching                                                │
│  └── KV Cache管理                                                       │
│                                                                         │
│  Phase 3: 高级特性 (Week 5-6)                                           │
│  ├── 量化特性 (AWQ/SmoothQuant/GPTQ/KV8)                                │
│  ├── 图模式 (Turbo-Graph/Acl-Graph/Torch.Compile)                       │
│  └── 高效解码 (Prefix-Cache/Chunked-Prefill/Speculative)                │
│                                                                         │
│  Phase 4: 分布式与优化 (Week 7-8)                                       │
│  ├── 分布式并行 (TP/PP/EP/CP)                                           │
│  ├── MoE优化与通信                                                       │
│  ├── PD分离部署                                                          │
│  └── 性能调优实践                                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 三、vLLM-Ascend 代码结构

### 3.1 项目目录结构

```
vllm-ascend/
├── vllm_ascend/                  # 核心代码目录
│   ├── __init__.py              # 包入口
│   ├── platform.py              # 平台抽象层
│   ├── worker/                  # Worker实现
│   │   ├── worker.py           # 主Worker类
│   │   ├── model_runner.py     # 模型运行器
│   │   └── ...
│   ├── attention/               # Attention实现
│   │   ├── attention.py        # 基础Attention
│   │   ├── mla_v1.py           # Multi-Latent Attention
│   │   ├── sfa_v1.py           # Sparse Flash Attention
│   │   └── ...
│   ├── ops/                     # 算子实现
│   │   ├── attention.py        # Attention算子
│   │   ├── layernorm.py        # LayerNorm算子
│   │   └── ...
│   ├── torchair/                # TorchAir图编译
│   │   ├── torchair_mla.py     # MLA图模式
│   │   └── ...
│   ├── eplb/                    # 专家并行负载均衡
│   │   ├── core/
│   │   │   └── policy/         # 负载均衡策略
│   │   └── ...
│   ├── quantization/            # 量化实现
│   │   ├── awq.py
│   │   ├── smoothquant.py
│   │   └── ...
│   └── utils/                   # 工具函数
├── csrc/                        # C++扩展
│   ├── tiling_base.h           # 算子分块
│   └── ...
├── benchmarks/                  # 性能测试
├── tests/                       # 测试用例
└── examples/                    # 示例代码
```

### 3.2 核心模块依赖关系

```
                    ┌─────────────┐
                    │  platform.py │ ← 平台抽象层
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   worker/   │    │ attention/  │    │quantization/│
│  Worker     │    │ Attention   │    │  Quantize   │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └─────────────────┬┴──────────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │    ops/     │ ← 底层算子
                  │  Operators  │
                  └─────────────┘
```

---

## 四、按文件学习框架代码

### Week 1: 项目结构与平台抽象层

#### Day 1-2: 项目入口与初始化

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `vllm_ascend/__init__.py` | 包入口、版本信息、导出接口 | 如何注册为vLLM插件？ |
| `vllm_ascend/platform.py` | 平台抽象层实现 | 如何抽象NPU硬件差异？ |

**代码阅读重点**:
```python
# platform.py 核心类
class AscendPlatform(Platform):
    @classmethod
    def get_device_name(cls) -> str:
        return "ascend"
    
    @classmethod
    def is_async_output_supported(cls) -> bool:
        return True
```

#### Day 3-4: Worker架构

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `worker/worker.py` | Worker主类实现 | Worker如何管理模型推理？ |
| `worker/model_runner.py` | 模型运行器 | 如何执行前向传播？ |

**代码阅读重点**:
```python
# worker.py 核心流程
class AscendWorker(WorkerBase):
    def init_model(self):
        # 模型初始化
        
    def execute_model(self, scheduler_output):
        # 执行推理
```

#### Day 5-6: 模型加载与权重管理

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `model_executor/` | 模型执行器 | 如何加载和切分模型？ |
| `weight_utils.py` | 权重工具 | 如何处理权重格式转换？ |

#### Day 7: 本周复盘

- 整理框架架构图
- 完成代码阅读笔记
- 自测：能否描述Worker初始化流程？

---

### Week 2: Attention与KV Cache核心

#### Day 8-9: Attention Backend

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `attention/attention.py` | 基础Attention实现 | 如何实现PagedAttention？ |
| `attention/backend.py` | Attention后端抽象 | 如何支持多种Attention后端？ |

**代码阅读重点**:
```python
# attention.py 核心实现
class AscendAttentionBackend(AttentionBackend):
    @staticmethod
    def get_impl_cls():
        return AscendAttentionImpl
```

#### Day 10-11: MLA与稀疏Attention

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `attention/mla_v1.py` | Multi-Latent Attention | MLA如何压缩KV Cache？ |
| `attention/sfa_v1.py` | Sparse Flash Attention | 稀疏Attention如何优化？ |

#### Day 12-13: KV Cache管理

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `kv_cache/` | KV Cache管理 | Block如何分配和回收？ |
| `block_manager.py` | Block管理器 | 如何实现Copy-on-Write？ |

#### Day 14: 本周复盘

- 整理Attention实现对比表
- 完成KV Cache管理流程图
- 自测：能否描述Block分配算法？

---

### Week 3: 调度与批处理

#### Day 15-16: 调度器实现

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `scheduler/` | 调度器实现 | 如何实现Continuous Batching？ |
| `scheduler_output.py` | 调度输出 | 调度决策如何传递给Worker？ |

#### Day 17-18: 批处理优化

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `batcher/` | 批处理器 | 如何动态调整batch？ |
| `chunked_prefill.py` | 分块预填充 | Chunked Prefill如何实现？ |

#### Day 19-20: 前缀缓存

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `prefix_caching/` | 前缀缓存 | 如何识别和复用前缀？ |
| `cache_engine.py` | 缓存引擎 | 缓存如何管理生命周期？ |

#### Day 21: 本周复盘

- 整理调度流程图
- 完成批处理策略对比
- 自测：能否描述调度决策过程？

---

### Week 4: 算子与底层优化

#### Day 22-23: 核心算子

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `ops/attention.py` | Attention算子 | 如何调用NPU融合算子？ |
| `ops/layernorm.py` | LayerNorm算子 | 如何优化归一化计算？ |

#### Day 24-25: 算子融合

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `ops/fused_*.py` | 融合算子 | 哪些算子可以融合？ |
| `csrc/` | C++扩展 | 如何开发自定义算子？ |

#### Day 26-27: 内存管理

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `memory/` | 内存管理 | 如何优化NPU内存使用？ |
| `block_allocator.py` | Block分配器 | 如何减少内存碎片？ |

#### Day 28: 本周复盘

- 整理算子融合策略
- 完成内存管理流程图
- 自测：能否描述算子调用链路？

---

## 四、按特性学习特性代码

### Week 5: 量化特性

#### 5.1 W4A16-AWQ量化

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `quantization/awq.py` | AWQ量化实现 | 如何实现权重INT4量化？ |
| `quantization/awq_utils.py` | AWQ工具函数 | 如何处理激活缩放？ |

**代码阅读重点**:
```python
# AWQ量化核心
class AWQLinearMethod:
    def apply(self, layer, x):
        # INT4权重反量化
        # INT4 * scale -> FP16
        # 矩阵乘法
```

#### 5.2 W8A8-SmoothQuant量化

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `quantization/smoothquant.py` | SmoothQuant实现 | 如何平衡权重和激活量化？ |

#### 5.3 KV Cache量化

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `quantization/kv_cache.py` | KV Cache量化 | 如何量化KV Cache？ |

---

### Week 6: 图模式特性

#### 6.1 Ascend-Turbo-Graph

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `torchair/turbo_graph.py` | Turbo Graph实现 | 如何实现整图下沉？ |
| `torchair/graph_cache.py` | 图缓存 | 如何缓存编译后的图？ |

**代码阅读重点**:
```python
# Turbo Graph核心
class TurboGraph:
    def capture(self, model):
        # 捕获计算图
        
    def run(self, inputs):
        # 图模式执行
```

#### 6.2 Acl-Graph

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `torchair/acl_graph.py` | ACL图模式 | 如何实现分段图执行？ |

#### 6.3 Torch.Compile

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `torchair/torch_compile.py` | Torch编译 | 如何对接torch.compile？ |

---

### Week 7: 分布式特性

#### 7.1 Tensor Parallelism

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `distributed/tensor_parallel.py` | TP实现 | 如何切分权重？ |
| `distributed/tp_comm.py` | TP通信 | 如何实现AllReduce？ |

**代码阅读重点**:
```python
# TP核心实现
class ColumnParallelLinear:
    def forward(self, x):
        # 列切分线性层
        y = F.linear(x, self.weight, self.bias)
        return y

class RowParallelLinear:
    def forward(self, x):
        # 行切分线性层 + AllReduce
        y = F.linear(x, self.weight)
        y = all_reduce(y)
        return y
```

#### 7.2 Expert Parallelism

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `eplb/core/` | 专家并行 | 如何实现专家负载均衡？ |
| `eplb/core/policy/` | 负载均衡策略 | 动态偏置路由如何实现？ |

#### 7.3 Context Parallelism

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `distributed/context_parallel.py` | CP实现 | 如何切分长序列？ |
| `distributed/ring_attention.py` | Ring Attention | 如何实现环形通信？ |

---

### Week 8: MoE优化与通信

#### 8.1 MoE核心实现

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `moe/experts.py` | 专家实现 | 如何实现专家网络？ |
| `moe/router.py` | 路由实现 | 如何实现Top-K路由？ |
| `moe/fused_moe.py` | 融合MoE | 如何融合MoE计算？ |

#### 8.2 通信优化

| 文件 | 学习内容 | 核心问题 |
|------|----------|----------|
| `communication/all_to_all.py` | All2All | 如何优化All2All？ |
| `communication/overlap.py` | 计算通信重叠 | 如何隐藏通信延迟？ |
| `communication/cp_optimization.py` | CP通信优化 | 如何优化All2All 8x？ |

#### 8.3 性能调优实践

| 内容 | 学习重点 |
|------|----------|
| Profiling工具 | 如何使用msprof分析性能？ |
| 性能瓶颈定位 | 如何识别计算/内存/通信瓶颈？ |
| 调优策略 | 如何选择最优配置？ |

---

## 五、每日学习模板

### 工作日学习流程

```
┌────────────────────────────────────────────────────────────┐
│                     每日学习流程                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  上午 (工作中碎片时间 30min)                                │
│  ├── 阅读目标文件代码                                       │
│  ├── 标记不理解的部分                                       │
│  └── 记录代码结构笔记                                       │
│                                                            │
│  晚上 (系统学习 1.5h)                                       │
│  ├── 深入理解核心逻辑                                       │
│  ├── 绘制流程图/架构图                                      │
│  ├── 对比vLLM原版实现差异                                   │
│  └── 整理学习笔记                                           │
│                                                            │
│  周末 (深入实践 3h)                                         │
│  ├── 运行示例代码                                           │
│  ├── 调试关键函数                                           │
│  ├── 尝试修改代码验证理解                                   │
│  └── 完成周复盘                                             │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 代码阅读笔记模板

```markdown
# [文件名] 学习笔记

## 1. 文件概述
- 路径: 
- 功能: 
- 依赖: 

## 2. 核心类/函数

### 2.1 [类名]
```python
class ClassName:
    """功能描述"""
    def __init__(self):
        pass
    def method(self):
        pass
```

## 3. 关键流程
1. 步骤1
2. 步骤2
3. ...

## 4. 与vLLM原版差异
- 差异1
- 差异2

## 5. 疑问与待深入
- [ ] 问题1
- [ ] 问题2
```

---

## 六、学习产出清单

### 6.1 必须产出

| 周次 | 产出物 | 验收标准 |
|------|--------|----------|
| Week 1 | 框架架构图 | 能清晰展示模块依赖关系 |
| Week 2 | Attention实现对比表 | 对比MLA/SFA/PA差异 |
| Week 3 | 调度流程图 | 完整展示调度决策过程 |
| Week 4 | 算子调用链路图 | 从API到NPU算子 |
| Week 5 | 量化实现对比表 | 对比AWQ/SQ/GPTQ |
| Week 6 | 图模式实现对比表 | 对比Turbo/Acl/Compile |
| Week 7 | 分布式并行对比表 | 对比TP/PP/EP/CP |
| Week 8 | 性能调优指南 | 包含常见问题与解决方案 |

### 6.2 可选产出

- [ ] 为vLLM-Ascend贡献PR
- [ ] 撰写技术博客
- [ ] 制作学习分享PPT
- [ ] 完善项目文档

---

## 七、学习资源

### 7.1 代码仓库

| 资源 | 链接 | 说明 |
|------|------|------|
| vLLM-Ascend | https://github.com/vllm-project/vllm-ascend | 主仓库 |
| vLLM | https://github.com/vllm-project/vllm | 上游仓库 |
| nano-vllm-ascend | https://github.com/linzm1007/nano-vllm-ascend | 简化学习版 |

### 7.2 文档资源

| 资源 | 链接 | 说明 |
|------|------|------|
| 官方文档 | https://docs.vllm.ai/projects/ascend/ | 官方文档 |
| CANN文档 | https://www.hiascend.com/document | 昇腾开发文档 |
| TorchAir文档 | https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/ | 图编译文档 |

### 7.3 已有学习材料

| 文件 | 路径 | 说明 |
|------|------|------|
| 特性总览 | `.trae/documents/vllm-ascend特性详解/vllm-ascend加速特性总览.md` | 特性概览 |
| Page-Attention | `.trae/documents/vllm-ascend特性详解/01-Page-Attention分页注意力.md` | PA详解 |
| Continuous-Batching | `.trae/documents/vllm-ascend特性详解/02-Continuous-Batching连续批处理.md` | CB详解 |
| 量化特性 | `.trae/documents/vllm-ascend特性详解/03-量化特性详解.md` | 量化详解 |
| MoE优化 | `.trae/documents/vllm-ascend特性详解/04-MoE优化特性详解.md` | MoE详解 |

---

## 八、自测问题清单

### Phase 1 自测 (Week 1-2)

1. vLLM-Ascend如何注册为vLLM的硬件插件？
2. Worker初始化流程是什么？
3. PagedAttention的Block如何分配？
4. MLA如何压缩KV Cache？

### Phase 2 自测 (Week 3-4)

1. Continuous Batching如何动态调整batch？
2. Chunked Prefill如何平衡prefill和decode？
3. 前缀缓存如何识别可复用的前缀？
4. 算子融合有哪些常见模式？

### Phase 3 自测 (Week 5-6)

1. AWQ量化的激活缩放如何计算？
2. SmoothQuant如何平衡权重和激活量化？
3. Turbo Graph如何实现整图下沉？
4. ACL Graph的分段执行策略是什么？

### Phase 4 自测 (Week 7-8)

1. TP的权重切分策略是什么？
2. EP的All2All通信如何优化？
3. CP的Ring Attention如何工作？
4. 如何定位和解决性能瓶颈？

---

## 九、下一步行动

1. **创建学习目录**: 在 `.trae/documents/` 下创建 `vllm-ascend代码学习/` 目录
2. **克隆代码仓库**: `git clone https://github.com/vllm-project/vllm-ascend.git`
3. **开始Week 1学习**: 从 `platform.py` 开始阅读

---

*学习原则: 代码为主，文档为辅；理解原理，动手实践*
