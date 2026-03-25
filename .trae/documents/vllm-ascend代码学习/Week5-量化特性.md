# vLLM-Ascend 框架代码学习 - Week 5

> 学习主题：量化特性
> 学习目标：理解AWQ/SmoothQuant/GPTQ/KV8量化实现，掌握量化策略选择

---

## 一、学习进度

| 日期 | 内容 | 状态 |
|------|------|------|
| Day 29-30 | W4A16-AWQ量化 | 待开始 |
| Day 31-32 | W8A8-SmoothQuant量化 | 待开始 |
| Day 33-34 | W8A16-GPTQ量化与KV8 | 待开始 |
| Day 35 | 本周复盘 | 待开始 |

---

## 二、代码阅读笔记

### 2.1 W4A16-AWQ量化

#### quantization/awq.py

**文件概述**:
- 路径: `vllm_ascend/quantization/awq.py`
- 功能: AWQ量化实现
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何实现权重INT4量化？
- [ ] 激活缩放如何计算？
- [ ] 反量化流程是什么？

---

#### quantization/awq_utils.py

**文件概述**:
- 路径: `vllm_ascend/quantization/awq_utils.py`
- 功能: AWQ工具函数
- 依赖: 

**核心函数**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何处理激活缩放？
- [ ] 量化误差如何控制？

---

### 2.2 W8A8-SmoothQuant量化

#### quantization/smoothquant.py

**文件概述**:
- 路径: `vllm_ascend/quantization/smoothquant.py`
- 功能: SmoothQuant实现
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何平衡权重和激活量化？
- [ ] 平滑因子如何计算？

---

### 2.3 W8A16-GPTQ量化

#### quantization/gptq.py

**文件概述**:
- 路径: `vllm_ascend/quantization/gptq.py`
- 功能: GPTQ量化实现
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] GPTQ量化流程是什么？
- [ ] 与AWQ的差异？

---

### 2.4 KV Cache量化

#### quantization/kv_cache.py

**文件概述**:
- 路径: `vllm_ascend/quantization/kv_cache.py`
- 功能: KV Cache量化
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何量化KV Cache？
- [ ] 量化对精度的影响？

---

## 三、架构图

### 3.1 量化实现对比

```
┌─────────────────────────────────────────────────────────────────┐
│                        量化特性对比                              │
├──────────────┬─────────────┬─────────────┬──────────────────────┤
│    特性       │   权重位宽   │   激活位宽   │       性能收益        │
├──────────────┼─────────────┼─────────────┼──────────────────────┤
│    AWQ       │    INT4     │    FP16     │ 小并发时延提升80%     │
│ SmoothQuant  │    INT8     │    INT8     │ 吞吐提升30%          │
│    GPTQ      │    INT8     │    FP16     │ 吞吐提升20%          │
│    KV8       │   KV INT8   │     -       │ 吞吐提升15-25%       │
└──────────────┴─────────────┴─────────────┴──────────────────────┘
```

---

## 四、与vLLM原版差异

| 模块 | vLLM原版 | vLLM-Ascend | 差异说明 |
|------|----------|-------------|----------|
| AWQ | CUDA实现 | NPU实现 | 待补充 |
| SmoothQuant | CUDA实现 | NPU实现 | 待补充 |
| KV Cache量化 | FP8 | INT8 | 待补充 |

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

1. 图模式特性
