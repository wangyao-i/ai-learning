# HCCL 快速上手路径

## 学习目标

快速掌握 HCCL 通信库的核心概念和开发方法，能够在昇腾 NPU 集群中进行分布式通信开发。

---

## 学习计划（2周）

### 第1周：核心概念与基础使用
- HCCL 概述与核心概念
- 基本API使用方法
- 常用通信算子
- 完整代码示例

### 第2周：开发实践与优化
- 算法选择策略
- 性能调优方法
- 常见问题解决
- 实战案例分析

---

## 进度追踪

| 周次 | 学习内容 | 完成状态 | 学习产出 |
|------|----------|----------|----------|
| 1 | 核心概念与基础使用 | 待完成 | 可运行的示例代码 |
| 2 | 开发实践与优化 | 待完成 | 性能优化方案 |

---

## 核心特性速览

### 1. 通信算子
| 算子 | 功能 | 使用场景 |
|------|------|----------|
| AllReduce | 梯度聚合 | 分布式训练 |
| AllGather | 数据收集 | 模型并行 |
| Broadcast | 参数分发 | 模型初始化 |
| ReduceScatter | 梯度分片 | 分布式损失计算 |

### 2. 通信算法
| 算法 | 适用场景 | 特点 |
|------|----------|------|
| Ring | 中小规模 | 抗拥塞 |
| Mesh | 大规模单机 | 并行度高 |
| RHD | 多机通信 | 步数少 |

### 3. 关键API
```cpp
// 初始化通信域
HcclCommInitAll(&comm, deviceCount);

// AllReduce
HcclAllReduce(sendBuf, recvBuf, count, dataType, op, comm, stream);

// 销毁通信域
HcclCommDestroy(comm);
```

---

## 快速开始

### 1. 环境准备
```bash
# 安装 CANN 工具包
# 安装 HCCL 通信库
# 配置环境变量
export HCCL_BUFFSIZE=524288000  # 500MB
```

### 2. 编写第一个程序
```cpp
#include "hccl/hccl.h"

int main() {
    HcclComm comm;
    HcclCommInitAll(&comm, 8);
    
    // 执行通信
    HcclAllReduce(...);
    
    HcclCommDestroy(comm);
    return 0;
}
```

---

## 学习资源

- **官方文档**：[cann-hccl](https://gitee.com/ascend/cann-hccl)
- **示例代码**：`examples/` 目录
- **调试工具**：MindStudio、性能分析工具

---

## 总结

通过本快速上手路径，您将：
1. 理解 HCCL 的核心概念
2. 掌握基本 API 的使用方法
3. 能够进行性能调优
4. 解决常见开发问题

开始学习吧！