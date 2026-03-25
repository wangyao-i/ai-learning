# vLLM-Ascend 框架代码学习

> **学习目标**: 深入理解vLLM-Ascend框架代码实现，掌握核心特性原理与代码细节
> **学习周期**: 8周
> **官方文档**: https://docs.vllm.ai/projects/ascend/

---

## 学习计划

详细学习计划请参考：[vllm-ascend框架特性代码学习计划.md](../vllm-ascend框架特性代码学习计划.md)

---

## 学习进度

| 周次 | 主题 | 学习笔记 | 状态 |
|------|------|----------|------|
| Week 1 | 项目结构与平台抽象层 | [Week1-项目结构与平台抽象层.md](./Week1-项目结构与平台抽象层.md) | 待开始 |
| Week 2 | Attention与KV Cache核心 | [Week2-Attention与KV-Cache核心.md](./Week2-Attention与KV-Cache核心.md) | 待开始 |
| Week 3 | 调度与批处理 | [Week3-调度与批处理.md](./Week3-调度与批处理.md) | 待开始 |
| Week 4 | 算子与底层优化 | [Week4-算子与底层优化.md](./Week4-算子与底层优化.md) | 待开始 |
| Week 5 | 量化特性 | [Week5-量化特性.md](./Week5-量化特性.md) | 待开始 |
| Week 6 | 图模式特性 | [Week6-图模式特性.md](./Week6-图模式特性.md) | 待开始 |
| Week 7 | 分布式特性 | [Week7-分布式特性.md](./Week7-分布式特性.md) | 待开始 |
| Week 8 | MoE优化与通信 | [Week8-MoE优化与通信.md](./Week8-MoE优化与通信.md) | 待开始 |

---

## 特性列表

### 调度特性
- Page-Attention
- Continuous Batching
- Multi-step

### 量化特性
- W4A16-AWQ
- W8A8-SmoothQuant
- W8A16-GPTQ
- KV8

### 高效解码特性
- Auto-Prefix-Caching
- Chunked-Prefill
- Speculative Decoding

### 图模式特性
- Ascend-Turbo-Graph
- Acl-Graph
- Torch.Compile

### 控制输出特性
- Guided Decoding
- Beam Search

### 实例复用特性
- Multi-LoRA

### 分离部署特性
- PD分离部署

### 分布式特性
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Expert Parallelism (EP)
- Context Parallelism (CP)

### MoE优化特性
- 细粒度专家分工
- 共享专家隔离
- 动态偏置路由
- Flash Comm共享专家混置
- CP特性 (通信剪枝)

---

## 学习资源

- [vLLM-Ascend 官方文档](https://docs.vllm.ai/projects/ascend/)
- [vLLM-Ascend GitHub](https://github.com/vllm-project/vllm-ascend)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [CANN 文档](https://www.hiascend.com/document)
- [TorchAir 文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/)

---

## 相关学习材料

- [vllm-ascend特性详解](../vllm-ascend特性详解/)
  - [Page-Attention分页注意力](../vllm-ascend特性详解/01-Page-Attention分页注意力.md)
  - [Continuous-Batching连续批处理](../vllm-ascend特性详解/02-Continuous-Batching连续批处理.md)
  - [量化特性详解](../vllm-ascend特性详解/03-量化特性详解.md)
  - [MoE优化特性详解](../vllm-ascend特性详解/04-MoE优化特性详解.md)
