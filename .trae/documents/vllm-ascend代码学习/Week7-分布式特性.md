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

### 2.1 Tensor Parallelism

#### distributed/tensor_parallel.py

**文件概述**:
- 路径: `vllm_ascend/distributed/tensor_parallel.py`
- 功能: TP实现
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何切分权重？
- [ ] 列切分和行切分的差异？

---

#### distributed/tp_comm.py

**文件概述**:
- 路径: `vllm_ascend/distributed/tp_comm.py`
- 功能: TP通信
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何实现AllReduce？
- [ ] HCCL通信如何优化？

---

### 2.2 Pipeline Parallelism

#### distributed/pipeline_parallel.py

**文件概述**:
- 路径: `vllm_ascend/distributed/pipeline_parallel.py`
- 功能: PP实现
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何切分层？
- [ ] 微批次如何调度？

---

### 2.3 Expert Parallelism

#### eplb/core/

**文件概述**:
- 路径: `vllm_ascend/eplb/core/`
- 功能: 专家并行
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何实现专家负载均衡？
- [ ] All2All通信如何优化？

---

#### eplb/core/policy/

**文件概述**:
- 路径: `vllm_ascend/eplb/core/policy/`
- 功能: 负载均衡策略
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] 动态偏置路由如何实现？
- [ ] 专家负载如何监控？

---

### 2.4 Context Parallelism

#### distributed/context_parallel.py

**文件概述**:
- 路径: `vllm_ascend/distributed/context_parallel.py`
- 功能: CP实现
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何切分长序列？
- [ ] 序列维度如何并行？

---

#### distributed/ring_attention.py

**文件概述**:
- 路径: `vllm_ascend/distributed/ring_attention.py`
- 功能: Ring Attention
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何实现环形通信？
- [ ] 通信开销如何优化？

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
