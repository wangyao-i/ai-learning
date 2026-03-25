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

#### ops/attention.py

**文件概述**:
- 路径: `vllm_ascend/ops/attention.py`
- 功能: Attention算子
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何调用NPU融合算子？
- [ ] 与CUDA Flash Attention的差异？

---

#### ops/layernorm.py

**文件概述**:
- 路径: `vllm_ascend/ops/layernorm.py`
- 功能: LayerNorm算子
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何优化归一化计算？
- [ ] 支持哪些归一化类型？

---

### 2.2 算子融合

#### ops/fused_*.py

**文件概述**:
- 路径: `vllm_ascend/ops/fused_*.py`
- 功能: 融合算子
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] 哪些算子可以融合？
- [ ] 融合策略是什么？

---

#### csrc/

**文件概述**:
- 路径: `vllm_ascend/csrc/`
- 功能: C++扩展
- 依赖: 

**核心内容**:
```cpp
// 待补充
```

**关键问题**:
- [ ] 如何开发自定义算子？
- [ ] TIK DSL如何使用？

---

### 2.3 内存管理

#### memory/

**文件概述**:
- 路径: `vllm_ascend/memory/`
- 功能: 内存管理
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何优化NPU内存使用？
- [ ] NUMA感知内存分配如何实现？

---

#### block_allocator.py

**文件概述**:
- 路径: `vllm_ascend/block_allocator.py`
- 功能: Block分配器
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何减少内存碎片？
- [ ] Block大小如何确定？

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
