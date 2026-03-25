# vLLM-Ascend 框架代码学习 - Week 2

> 学习主题：Attention与KV Cache核心
> 学习目标：理解PagedAttention实现，掌握MLA/SFA差异，理解Block分配算法

---

## 一、学习进度

| 日期 | 内容 | 状态 |
|------|------|------|
| Day 8-9 | Attention Backend | 待开始 |
| Day 10-11 | MLA与稀疏Attention | 待开始 |
| Day 12-13 | KV Cache管理 | 待开始 |
| Day 14 | 本周复盘 | 待开始 |

---

## 二、代码阅读笔记

### 2.1 Attention Backend

#### attention/attention.py

**文件概述**:
- 路径: `vllm_ascend/attention/attention.py`
- 功能: 基础Attention实现
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何实现PagedAttention？
- [ ] 与Flash Attention的关系？

---

#### attention/backend.py

**文件概述**:
- 路径: `vllm_ascend/attention/backend.py`
- 功能: Attention后端抽象
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何支持多种Attention后端？
- [ ] 后端选择策略是什么？

---

### 2.2 MLA与稀疏Attention

#### attention/mla_v1.py

**文件概述**:
- 路径: `vllm_ascend/attention/mla_v1.py`
- 功能: Multi-Latent Attention
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] MLA如何压缩KV Cache？
- [ ] 潜向量如何计算？

---

#### attention/sfa_v1.py

**文件概述**:
- 路径: `vllm_ascend/attention/sfa_v1.py`
- 功能: Sparse Flash Attention
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] 稀疏Attention如何优化？
- [ ] 稀疏模式有哪些？

---

### 2.3 KV Cache管理

#### kv_cache/

**文件概述**:
- 路径: `vllm_ascend/kv_cache/`
- 功能: KV Cache管理
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] Block如何分配和回收？
- [ ] 如何处理Copy-on-Write？

---

#### block_manager.py

**文件概述**:
- 路径: `vllm_ascend/block_manager.py`
- 功能: Block管理器
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] Block分配算法是什么？
- [ ] 如何处理Block碎片？

---

## 三、架构图

### 3.1 Attention实现对比

```
待补充
```

### 3.2 KV Cache管理流程

```
待补充
```

---

## 四、与vLLM原版差异

| 模块 | vLLM原版 | vLLM-Ascend | 差异说明 |
|------|----------|-------------|----------|
| AttentionBackend | FlashAttentionBackend | AscendAttentionBackend | 待补充 |
| BlockManager | BlockManager | AscendBlockManager | 待补充 |
| KV Cache | GPU KV Cache | NPU KV Cache | 待补充 |

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

1. 调度与批处理
