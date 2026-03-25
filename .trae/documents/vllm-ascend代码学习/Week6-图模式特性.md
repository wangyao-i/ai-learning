# vLLM-Ascend 框架代码学习 - Week 6

> 学习主题：图模式特性
> 学习目标：理解Turbo-Graph/Acl-Graph/Torch.Compile实现，掌握图模式选择策略

---

## 一、学习进度

| 日期 | 内容 | 状态 |
|------|------|------|
| Day 36-37 | Ascend-Turbo-Graph | 待开始 |
| Day 38-39 | Acl-Graph | 待开始 |
| Day 40-41 | Torch.Compile与高效解码 | 待开始 |
| Day 42 | 本周复盘 | 待开始 |

---

## 二、代码阅读笔记

### 2.1 Ascend-Turbo-Graph

#### torchair/turbo_graph.py

**文件概述**:
- 路径: `vllm_ascend/torchair/turbo_graph.py`
- 功能: Turbo Graph实现
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何实现整图下沉？
- [ ] 如何支持动态shape？
- [ ] 图捕获流程是什么？

---

#### torchair/graph_cache.py

**文件概述**:
- 路径: `vllm_ascend/torchair/graph_cache.py`
- 功能: 图缓存
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何缓存编译后的图？
- [ ] 缓存失效策略是什么？

---

### 2.2 Acl-Graph

#### torchair/acl_graph.py

**文件概述**:
- 路径: `vllm_ascend/torchair/acl_graph.py`
- 功能: ACL图模式
- 依赖: 

**核心类**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何实现分段图执行？
- [ ] 与Turbo Graph的差异？

---

### 2.3 Torch.Compile

#### torchair/torch_compile.py

**文件概述**:
- 路径: `vllm_ascend/torchair/torch_compile.py`
- 功能: Torch编译
- 依赖: 

**核心实现**:
```python
# 待补充
```

**关键问题**:
- [ ] 如何对接torch.compile？
- [ ] 后端转换流程是什么？

---

### 2.4 高效解码特性

#### prefix_caching/

**文件概述**:
- 路径: `vllm_ascend/prefix_caching/`
- 功能: 前缀缓存（Auto-Prefix-Caching）
- 依赖: 

**关键问题**:
- [ ] 如何识别和复用前缀？
- [ ] 缓存命中率如何优化？

---

#### chunked_prefill.py

**文件概述**:
- 路径: `vllm_ascend/chunked_prefill.py`
- 功能: Chunked-Prefill (Split-Fuse)
- 依赖: 

**关键问题**:
- [ ] 如何实现全量增量同时推理？
- [ ] 资源利用率如何提升？

---

#### speculative_decoding/

**文件概述**:
- 路径: `vllm_ascend/speculative/`
- 功能: 投机解码
- 依赖: 

**关键问题**:
- [ ] 大小模型投机如何实现？
- [ ] 接受率如何优化？

---

## 三、架构图

### 3.1 图模式实现对比

```
┌─────────────────────────────────────────────────────────────────┐
│                        图模式特性对比                            │
├──────────────┬──────────────────┬───────────────────────────────┤
│    特性       │      特点        │          性能收益             │
├──────────────┼──────────────────┼───────────────────────────────┤
│ Turbo-Graph  │ 整图下沉，动态shape│ Decode吞吐翻倍               │
│  Acl-Graph   │ 分段图执行        │ 相比eager提升40-60%          │
│Torch.Compile │ Torch dynamo构图  │ 推理性能提升30-50%           │
└──────────────┴──────────────────┴───────────────────────────────┘
```

---

## 四、与vLLM原版差异

| 模块 | vLLM原版 | vLLM-Ascend | 差异说明 |
|------|----------|-------------|----------|
| Graph Mode | CUDA Graph | Turbo/Acl Graph | 待补充 |
| Speculative | CUDA实现 | NPU实现 | 待补充 |

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

1. 分布式特性
