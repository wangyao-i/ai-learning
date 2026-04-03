# vLLM-Ascend 代码学习项目

> 基于大模型交互的 vLLM-Ascend 框架学习与实践项目

## 项目简介

本项目旨在通过大模型交互的方式，帮助开发者深入学习 vLLM-Ascend 框架的核心原理和实现细节。项目包含完整的学习资料、代码解读和实践练习，支持社区协作和内容补充。

**参考项目**：
- [InfraTech](https://github.com/wangyao-i/InfraTech) - AI Infra知识分享与代码练习
- [ai-infra-learning](https://github.com/wangyao-i/ai-infra-learning) - AI Infra学习会议资料

---

## 📚 完整目录结构

```
ai-learning/
├── .trae/
│   ├── documents/                           # 📖 学习文档总目录
│   │   │
│   │   ├── vllm-ascend代码学习/              # 🎯 核心学习路径 (8周计划)
│   │   │   ├── Week1-项目结构与平台抽象层.md
│   │   │   │   ├── Day 1-2: 项目入口与初始化
│   │   │   │   │   ├── vllm_ascend/__init__.py
│   │   │   │   │   ├── setup.py (entry_points机制)
│   │   │   │   │   └── 插件注册流程
│   │   │   │   ├── Day 3-4: Worker架构
│   │   │   │   │   ├── worker/worker.py
│   │   │   │   │   ├── worker/model_runner.py
│   │   │   │   │   └── Worker初始化流程
│   │   │   │   ├── Day 5-6: 模型加载与权重管理
│   │   │   │   │   ├── model_executor/
│   │   │   │   │   ├── weight_utils.py
│   │   │   │   │   └── 权重切分策略
│   │   │   │   └── Day 7: 本周复盘
│   │   │   │
│   │   │   ├── Week2-Attention与KV-Cache核心.md
│   │   │   │   ├── Day 8-9: Attention Backend
│   │   │   │   │   ├── attention/attention.py
│   │   │   │   │   ├── attention/backend.py
│   │   │   │   │   └── PagedAttention实现
│   │   │   │   ├── Day 10-11: MLA与稀疏Attention
│   │   │   │   │   ├── attention/mla_v1.py
│   │   │   │   │   ├── attention/sfa_v1.py
│   │   │   │   │   └── KV Cache压缩原理
│   │   │   │   ├── Day 12-13: KV Cache管理
│   │   │   │   │   ├── kv_cache/
│   │   │   │   │   ├── block_manager.py
│   │   │   │   │   └── Copy-on-Write机制
│   │   │   │   └── Day 14: 本周复盘
│   │   │   │
│   │   │   ├── Week3-调度与批处理.md
│   │   │   │   ├── Day 15-16: 调度器实现
│   │   │   │   │   ├── scheduler/
│   │   │   │   │   ├── scheduler_output.py
│   │   │   │   │   └── Continuous Batching
│   │   │   │   ├── Day 17-18: 批处理优化
│   │   │   │   │   ├── batcher/
│   │   │   │   │   ├── chunked_prefill.py
│   │   │   │   │   └── Chunked Prefill实现
│   │   │   │   ├── Day 19-20: 前缀缓存
│   │   │   │   │   ├── prefix_caching/
│   │   │   │   │   ├── cache_engine.py
│   │   │   │   │   └── 前缀识别与复用
│   │   │   │   └── Day 21: 本周复盘
│   │   │   │
│   │   │   ├── Week4-算子与底层优化.md
│   │   │   │   ├── Day 22-23: 核心算子
│   │   │   │   │   ├── ops/attention.py
│   │   │   │   │   ├── ops/layernorm.py
│   │   │   │   │   └── NPU融合算子
│   │   │   │   ├── Day 24-25: 算子融合
│   │   │   │   │   ├── ops/fused_*.py
│   │   │   │   │   ├── csrc/
│   │   │   │   │   └── 自定义算子开发
│   │   │   │   ├── Day 26-27: 内存管理
│   │   │   │   │   ├── memory/
│   │   │   │   │   ├── block_allocator.py
│   │   │   │   │   └── NPU内存优化
│   │   │   │   └── Day 28: 本周复盘
│   │   │   │
│   │   │   ├── Week5-量化特性.md
│   │   │   │   ├── 5.1 W4A16-AWQ量化
│   │   │   │   │   ├── quantization/awq.py
│   │   │   │   │   ├── quantization/awq_utils.py
│   │   │   │   │   └── INT4权重量化
│   │   │   │   ├── 5.2 W8A8-SmoothQuant量化
│   │   │   │   │   ├── quantization/smoothquant.py
│   │   │   │   │   └── 权重激活量化
│   │   │   │   ├── 5.3 W8A16-GPTQ量化
│   │   │   │   │   ├── quantization/gptq.py
│   │   │   │   │   └── 权重量化
│   │   │   │   ├── 5.4 KV Cache量化
│   │   │   │   │   ├── quantization/kv_cache.py
│   │   │   │   │   └── KV8量化
│   │   │   │   └── 5.5 量化对比与实践
│   │   │   │
│   │   │   ├── Week6-图模式特性.md
│   │   │   │   ├── 6.1 Ascend-Turbo-Graph
│   │   │   │   │   ├── torchair/turbo_graph.py
│   │   │   │   │   ├── torchair/graph_cache.py
│   │   │   │   │   └── 整图下沉
│   │   │   │   ├── 6.2 Acl-Graph
│   │   │   │   │   ├── torchair/acl_graph.py
│   │   │   │   │   └── 分段图执行
│   │   │   │   ├── 6.3 Torch.Compile
│   │   │   │   │   ├── torchair/torch_compile.py
│   │   │   │   │   └── Torch编译优化
│   │   │   │   └── 6.4 图模式对比与实践
│   │   │   │
│   │   │   ├── Week7-分布式特性.md
│   │   │   │   ├── 7.1 Tensor Parallelism (TP)
│   │   │   │   │   ├── distributed/tensor_parallel.py
│   │   │   │   │   ├── distributed/tp_comm.py
│   │   │   │   │   └── 张量并行实现
│   │   │   │   ├── 7.2 Pipeline Parallelism (PP)
│   │   │   │   │   ├── distributed/pipeline_parallel.py
│   │   │   │   │   └── 流水线并行实现
│   │   │   │   ├── 7.3 Expert Parallelism (EP)
│   │   │   │   │   ├── eplb/core/
│   │   │   │   │   ├── eplb/core/policy/
│   │   │   │   │   └── 专家并行实现
│   │   │   │   ├── 7.4 Context Parallelism (CP)
│   │   │   │   │   ├── distributed/context_parallel.py
│   │   │   │   │   ├── distributed/ring_attention.py
│   │   │   │   │   └── 上下文并行实现
│   │   │   │   └── 7.5 分布式对比与实践
│   │   │   │
│   │   │   └── Week8-MoE优化与通信.md
│   │   │       ├── 8.1 MoE核心实现
│   │   │       │   ├── moe/experts.py
│   │   │       │   ├── moe/router.py
│   │   │       │   ├── moe/fused_moe.py
│   │   │       │   └── MoE计算流程
│   │   │       ├── 8.2 MoE优化特性
│   │   │       │   ├── 细粒度专家分工
│   │   │       │   ├── 共享专家隔离
│   │   │       │   ├── 动态偏置路由
│   │   │       │   └── Flash Comm共享专家混置
│   │   │       ├── 8.3 通信优化
│   │   │       │   ├── communication/all_to_all.py
│   │   │       │   ├── communication/overlap.py
│   │   │       │   ├── communication/cp_optimization.py
│   │   │       │   └── All2All优化8x
│   │   │       └── 8.4 性能调优实践
│   │   │
│   │   ├── vllm-ascend特性详解/              # 📋 特性专题详解
│   │   │   ├── vllm-ascend加速特性总览.md     # 特性总览
│   │   │   ├── 01-Page-Attention分页注意力.md
│   │   │   │   ├── PagedAttention原理
│   │   │   │   ├── Block管理机制
│   │   │   │   ├── 内存共享与Copy-on-Write
│   │   │   │   └── 性能收益分析
│   │   │   ├── 02-Continuous-Batching连续批处理.md
│   │   │   │   ├── Continuous Batching原理
│   │   │   │   ├── 迭代级调度
│   │   │   │   ├── 动态batch调整
│   │   │   │   └── 性能收益分析
│   │   │   ├── 03-量化特性详解.md
│   │   │   │   ├── AWQ量化原理
│   │   │   │   ├── SmoothQuant量化原理
│   │   │   │   ├── GPTQ量化原理
│   │   │   │   ├── KV Cache量化
│   │   │   │   └── 量化对比分析
│   │   │   └── 04-MoE优化特性详解.md
│   │   │       ├── MoE架构原理
│   │   │       ├── 专家并行策略
│   │   │       ├── 负载均衡机制
│   │   │       └── 通信优化技术
│   │   │
│   │   └── 其他学习资料/                     # 📖 辅助学习资料
│   │       ├── AI-Infra面试学习计划3个月.md   # 面试准备
│   │       ├── Encoder_Decoder_深入解析.md    # 模型架构
│   │       ├── LLM推理核心参数与使用场景详解.md # 推理参数
│   │       ├── PagedAttention变量详解.md      # PA变量详解
│   │       ├── Python初级语法-C++转Python指南.md # Python入门
│   │       ├── Python高级语法-vLLM代码实例讲解.md # Python进阶
│   │       ├── 位置编码详解.md                # 位置编码
│   │       ├── Week1-LLM模型结构基础.md       # LLM基础
│   │       ├── Week2-主流LLM模型结构.md       # 主流模型
│   │       ├── Week3-LLM推理核心原理.md       # 推理原理
│   │       ├── Week4-vLLM架构深入.md          # vLLM架构
│   │       ├── Week5-NPU推理深入与性能优化.md # NPU优化
│   │       ├── Week6-vLLM调度器与KV Cache源码深挖.md # 源码深挖
│   │       ├── Week7-vLLM Worker与分布式推理.md # 分布式
│   │       ├── Week8-推理优化专项-量化与算子优化.md # 优化专项
│   │       └── Week9-推理服务化与部署.md      # 服务化部署
│   │
│   ├── skills/                              # 🤖 自动化技能
│   │   └── git-archive/                     # Git归档技能
│   │       └── SKILL.md
│   │
│   └── .ignore
│
├── README.md                                # 项目说明
└── .git/                                    # Git版本控制
```

---

## 🎯 学习计划概览

### Phase 1: 框架基础 (Week 1-2)

| 周次 | 学习主题 | 核心内容 | 难度 | 学习产出 |
|------|---------|----------|------|----------|
| Week 1 | 项目结构与平台抽象层 | 插件注册机制、Platform抽象、Worker初始化 | ⚡️ | 框架架构图 |
| Week 2 | Attention与KV-Cache核心 | PagedAttention、MLA/SFA、Block管理 | ⚡️⚡️ | Attention实现对比表 |

### Phase 2: 核心特性 (Week 3-4)

| 周次 | 学习主题 | 核心内容 | 难度 | 学习产出 |
|------|---------|----------|------|----------|
| Week 3 | 调度与批处理 | Continuous Batching、Chunked Prefill、前缀缓存 | ⚡️⚡️ | 调度流程图 |
| Week 4 | 算子与底层优化 | 自定义算子、性能优化、内存管理 | ⚡️⚡️ | 算子调用链路图 |

### Phase 3: 高级特性 (Week 5-6)

| 周次 | 学习主题 | 核心内容 | 难度 | 学习产出 |
|------|---------|----------|------|----------|
| Week 5 | 量化特性 | AWQ、SmoothQuant、GPTQ、KV8量化推理 | ⚡️⚡️ | 量化实现对比表 |
| Week 6 | 图模式特性 | TorchAir、ACL Graph、Turbo Graph | ⚡️⚡️⚡️ | 图模式实现对比表 |

### Phase 4: 分布式与优化 (Week 7-8)

| 周次 | 学习主题 | 核心内容 | 难度 | 学习产出 |
|------|---------|----------|------|----------|
| Week 7 | 分布式特性 | TP/PP/EP/CP分布式推理 | ⚡️⚡️ | 分布式并行对比表 |
| Week 8 | MoE优化与通信 | 专家并行、负载均衡、通信优化 | ⚡️⚡️⚡️ | 性能调优指南 |

---

## 📖 知识体系详解

### 🔍 推理基础知识

| 主题 | 说明 | 核心文件 | 难度 |
|------|------|----------|------|
| **PagedAttention** | 分页注意力机制，KV Cache分块管理 | `attention/attention.py` | ⚡️⚡️ |
| **Continuous Batching** | 连续批处理，动态调整batch | `scheduler/` | ⚡️⚡️ |
| **Chunked Prefill** | 分块预填充，全量增量同时推理 | `chunked_prefill.py` | ⚡️⚡️ |
| **Prefix Caching** | 前缀缓存，降低首token时延 | `prefix_caching/` | ⚡️⚡️ |
| **Speculative Decoding** | 投机推理，大小模型协同 | `speculative/` | ⚡️⚡️ |
| **KV Cache管理** | Block分配、Copy-on-Write | `kv_cache/`, `block_manager.py` | ⚡️⚡️ |
| **LLM Sampling** | 推理采样策略 | `sampling/` | ⚡️ |

### 🧩 并行推理策略

| 策略 | 说明 | 核心文件 | 通信方式 | 难度 |
|------|------|----------|----------|------|
| **Data Parallelism (DP)** | 数据并行，多副本推理 | `distributed/` | 无需通信 | ⚡️ |
| **Tensor Parallelism (TP)** | 张量并行，层内切分 | `distributed/tensor_parallel.py` | HCCL AllReduce | ⚡️⚡️ |
| **Pipeline Parallelism (PP)** | 流水线并行，层间切分 | `distributed/pipeline_parallel.py` | NPU间点对点 | ⚡️⚡️ |
| **Expert Parallelism (EP)** | 专家并行，MoE专用 | `eplb/core/` | All2All | ⚡️⚡️ |
| **Context Parallelism (CP)** | 上下文并行，长序列支持 | `distributed/context_parallel.py` | Ring-Attention | ⚡️⚡️⚡️ |

### 🚀 性能优化

| 主题 | 说明 | 核心文件 | 性能收益 | 难度 |
|------|------|----------|----------|------|
| **FlashAttention** | 融合注意力算子 | `ops/attention.py` | Attention速度提升2-3x | ⚡️⚡️ |
| **MLA** | 潜向量注意力，KV压缩 | `attention/mla_v1.py` | KV Cache减少4-8倍 | ⚡️⚡️⚡️ |
| **SFA** | 稀疏注意力优化 | `attention/sfa_v1.py` | 计算量减少50%+ | ⚡️⚡️⚡️ |
| **算子融合** | 多算子融合优化 | `ops/fused_*.py` | Kernel launch开销降低50%+ | ⚡️⚡️ |
| **内存管理** | NPU内存优化 | `memory/` | 内存访问效率提升20-30% | ⚡️⚡️ |
| **通信优化** | 计算通信重叠 | `communication/overlap.py` | 端到端性能提升15-20% | ⚡️⚡️⚡️ |

### 🔧 量化技术

| 技术 | 说明 | 核心文件 | 性能收益 | 难度 |
|------|------|----------|----------|------|
| **W4A16-AWQ** | 权重INT4量化 | `quantization/awq.py` | 显存降低75%，时延提升80% | ⚡️⚡️ |
| **W8A8-SmoothQuant** | 权重激活INT8量化 | `quantization/smoothquant.py` | 吞吐提升30%，显存降低50% | ⚡️⚡️ |
| **W8A16-GPTQ** | 权重INT8量化 | `quantization/gptq.py` | 吞吐提升20%，显存降低50% | ⚡️⚡️ |
| **KV8** | KV Cache INT8量化 | `quantization/kv_cache.py` | 吞吐提升15-25%，显存降低50% | ⚡️⚡️ |

### 🎯 图模式加速

| 技术 | 说明 | 核心文件 | 性能收益 | 难度 |
|------|------|----------|----------|------|
| **Turbo Graph** | 整图下沉，消除Python开销 | `torchair/turbo_graph.py` | Decode吞吐翻倍 | ⚡️⚡️⚡️ |
| **ACL Graph** | 分段图模式 | `torchair/acl_graph.py` | 提升40-60% | ⚡️⚡️ |
| **Torch.Compile** | Torch编译优化 | `torchair/torch_compile.py` | 提升30-50% | ⚡️⚡️ |

---

## 🛠️ 如何使用

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/wangyao-i/ai-learning.git
cd ai-learning

# 安装依赖
pip install -r requirements.txt

# 安装 vllm-ascend (可选，如需在NPU上实践)
pip install vllm-ascend
```

### 2. 学习方式

#### 方式一：按周学习（推荐）
1. 从Week 1开始，逐步深入学习
2. 每天阅读目标文件代码
3. 使用大模型工具进行问答和内容补充
4. 完成每周的实践练习

#### 方式二：按特性学习
1. 选择感兴趣的特性专题
2. 阅读特性详解文档
3. 深入相关代码实现
4. 完成实践练习

#### 方式三：问题驱动学习
1. 从自测问题清单开始
2. 带着问题阅读代码
3. 整理学习笔记
4. 与社区分享学习心得

### 3. 学习路径

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        vLLM-Ascend 学习路径                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1: 框架基础 (Week 1-2)                                           │
│  ├── 项目结构与入口                                                      │
│  │   ├── entry_points机制                                               │
│  │   ├── 插件注册流程                                                    │
│  │   └── 平台抽象层                                                      │
│  ├── Worker架构                                                         │
│  │   ├── Worker初始化                                                   │
│  │   ├── ModelRunner实现                                                │
│  │   └── 模型加载                                                        │
│  └── Attention核心                                                      │
│      ├── PagedAttention实现                                             │
│      ├── MLA压缩机制                                                    │
│      └── KV Cache管理                                                   │
│                                                                         │
│  Phase 2: 核心特性 (Week 3-4)                                           │
│  ├── 调度系统                                                           │
│  │   ├── Continuous Batching                                            │
│  │   ├── Chunked Prefill                                                │
│  │   └── 前缀缓存                                                        │
│  └── 算子优化                                                           │
│      ├── 核心算子实现                                                    │
│      ├── 算子融合策略                                                    │
│      └── 内存管理                                                        │
│                                                                         │
│  Phase 3: 高级特性 (Week 5-6)                                           │
│  ├── 量化技术                                                           │
│  │   ├── AWQ量化                                                        │
│  │   ├── SmoothQuant量化                                                │
│  │   ├── GPTQ量化                                                       │
│  │   └── KV Cache量化                                                   │
│  └── 图模式                                                             │
│      ├── Turbo Graph                                                    │
│      ├── ACL Graph                                                      │
│      └── Torch.Compile                                                  │
│                                                                         │
│  Phase 4: 分布式与优化 (Week 7-8)                                       │
│  ├── 分布式并行                                                         │
│  │   ├── Tensor Parallelism                                             │
│  │   ├── Pipeline Parallelism                                           │
│  │   ├── Expert Parallelism                                             │
│  │   └── Context Parallelism                                            │
│  ├── MoE优化                                                            │
│  │   ├── 专家并行策略                                                    │
│  │   ├── 负载均衡机制                                                    │
│  │   └── 通信优化技术                                                    │
│  └── 性能调优                                                           │
│      ├── Profiling工具                                                  │
│      ├── 性能瓶颈定位                                                    │
│      └── 调优策略                                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 核心特性列表

### vLLM-Ascend 完整特性

| 特性类别 | 特性名称 | 说明 | 性能收益 |
|----------|----------|------|----------|
| **调度特性** | Page-Attention | 分块管理KV Cache | 吞吐提升30-50% |
| | Continuous Batching | 迭代级调度，动态调整batch | 延迟降低15-30%，吞吐提升20-40% |
| | Multi-step | 一次调度多次推理 | CPU调度开销降低60%+ |
| **量化特性** | W4A16-AWQ | 权重INT4量化 | 显存降低75%，时延提升80% |
| | W8A8-SmoothQuant | 权重激活INT8量化 | 吞吐提升30%，显存降低50% |
| | W8A16-GPTQ | 权重INT8量化 | 吞吐提升20%，显存降低50% |
| | KV8 | KV Cache INT8量化 | 吞吐提升15-25%，显存降低50% |
| **高效解码** | Auto-Prefix-Caching | 前缀缓存 | 首token时延降低40-60% |
| | Chunked-Prefill | 全量增量同时推理 | 资源利用率提升25-35% |
| | Speculative Decoding | 投机推理 | 推理性能提升1.5-2.5x |
| **图模式** | Turbo-Graph | 整图下沉 | Decode吞吐翻倍 |
| | ACL-Graph | 分段图模式 | 提升40-60% |
| | Torch.Compile | Torch编译优化 | 提升30-50% |
| **分布式** | Tensor Parallelism | 张量并行 | 支持多NPU并行 |
| | Pipeline Parallelism | 流水线并行 | 支持大模型部署 |
| | Expert Parallelism | 专家并行 | 支持MoE架构 |
| | Context Parallelism | 上下文并行 | 支持超长上下文 |
| **MoE优化** | 细粒度专家分工 | 专家拆分 | 参数利用率提升20-30% |
| | 动态偏置路由 | 负载均衡 | 负载均衡度提升60%+ |
| | CP通信剪枝 | All2All优化 | 通信性能提升8x |

---

## 🤝 社区贡献

我们欢迎社区贡献：

1. **内容补充**：完善学习文档和代码解读
2. **实践案例**：分享实际使用场景和性能调优经验
3. **问题反馈**：报告学习过程中遇到的问题
4. **功能建议**：提出新的学习主题和内容

### 如何贡献

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 发起Pull Request

---

## 📚 学习资源

### 官方文档
- [vLLM官方文档](https://vllm.readthedocs.io/)
- [vLLM-Ascend GitHub](https://github.com/vllm-project/vllm-ascend)
- [华为昇腾开发者文档](https://www.hiascend.com/)
- [PyTorch官方文档](https://pytorch.org/docs/)

### 推荐学习项目
- [InfraTech](https://github.com/wangyao-i/InfraTech) - AI Infra知识分享与代码练习
- [ai-infra-learning](https://github.com/wangyao-i/ai-infra-learning) - AI Infra学习会议资料

### 代码仓库
- [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend) - 主仓库
- [vLLM](https://github.com/vllm-project/vllm) - 上游仓库

---

## 📞 联系我们

- **GitHub Issues**：提交问题和建议
- **Discussions**：讨论学习心得和技术问题

---

**让我们一起探索vLLM-Ascend的无限可能！🚀**