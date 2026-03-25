# vLLM-Ascend 框架代码学习 - Week 3

> 学习主题：调度与批处理
> 学习目标：理解Continuous Batching实现，掌握Chunked Prefill机制，理解前缀缓存

---

## 一、学习进度

| 日期 | 内容 | 状态 |
|------|------|------|
| Day 15-16 | 调度器实现 | 待开始 |
| Day 17-18 | 批处理优化 | 待开始 |
| Day 19-20 | 前缀缓存 | 待开始 |
| Day 21 | 本周复盘 | 待开始 |

---

## 二、代码阅读笔记

### 2.1 调度器实现

#### scheduler/

**文件概述**:
- 路径: `vllm_ascend/scheduler/`
- 功能: 调度器实现
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何实现Continuous Batching？
- [ ] 调度策略有哪些？

---

#### scheduler_output.py

**文件概述**:
- 路径: `vllm_ascend/scheduler_output.py`
- 功能: 调度输出
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] 调度决策如何传递给Worker？
- [ ] 包含哪些调度信息？

---

### 2.2 批处理优化

#### batcher/

**文件概述**:
- 路径: `vllm_ascend/batcher/`
- 功能: 批处理器
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何动态调整batch？
- [ ] batch大小如何确定？

---

#### chunked_prefill.py

**文件概述**:
- 路径: `vllm_ascend/chunked_prefill.py`
- 功能: 分块预填充
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] Chunked Prefill如何实现？
- [ ] 如何平衡prefill和decode？

---

### 2.3 前缀缓存

#### prefix_caching/

**文件概述**:
- 路径: `vllm_ascend/prefix_caching/`
- 功能: 前缀缓存
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何识别和复用前缀？
- [ ] 缓存命中率如何优化？

---

#### cache_engine.py

**文件概述**:
- 路径: `vllm_ascend/cache_engine.py`
- 功能: 缓存引擎
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] 缓存如何管理生命周期？
- [ ] 如何处理缓存失效？

---

## 三、架构图

### 3.1 调度流程

```
待补充
```

### 3.2 批处理策略

```
待补充
```

---

## 四、与vLLM原版差异

| 模块 | vLLM原版 | vLLM-Ascend | 差异说明 |
|------|----------|-------------|----------|
| Scheduler | Scheduler | AscendScheduler | 待补充 |
| ChunkedPrefill | ChunkedPrefillScheduler | AscendChunkedPrefill | 待补充 |
| PrefixCaching | PrefixCaching | AscendPrefixCaching | 待补充 |

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

1. 算子与底层优化
