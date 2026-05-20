# Week1 - 核心概念与基础使用

## 1. HCCL 概述

**HCCL**（Huawei Collective Communication Library）是基于昇腾 AI 处理器的高性能集合通信库，为分布式深度学习训练提供通信支持。

### 核心功能
- 提供多种集合通信算子
- 支持多种通信算法
- 适配昇腾 NPU 硬件

---

## 2. 核心概念

### 2.1 通信域
通信域是通信操作的上下文，管理参与通信的进程和资源。

```cpp
// 初始化通信域（单机多卡）
HcclComm comm;
HcclCommInitAll(&comm, deviceCount);

// 获取当前rank
int rank;
HcclGetRank(comm, &rank);

// 获取通信域大小
int size;
HcclGetSize(comm, &size);
```

### 2.2 Rank
通信域中每个进程的唯一标识，从 0 开始编号。

### 2.3 数据类型
```cpp
typedef enum {
    HCCL_DATA_TYPE_FLOAT32 = 0,      // 32位浮点
    HCCL_DATA_TYPE_FLOAT16 = 1,      // 16位浮点
    HCCL_DATA_TYPE_INT32 = 2,        // 32位整数
    HCCL_DATA_TYPE_INT8 = 3,         // 8位整数
} HcclDataType;
```

### 2.4 归约操作
```cpp
typedef enum {
    HCCL_REDUCE_SUM = 0,             // 求和
    HCCL_REDUCE_PROD = 1,            // 求积
    HCCL_REDUCE_MAX = 2,             // 求最大值
    HCCL_REDUCE_MIN = 3,             // 求最小值
} HcclReduceOp;
```

---

## 3. 核心 API 详解

### 3.1 AllReduce - 全归约

**功能**：所有进程数据归约，结果广播到所有进程

**应用场景**：梯度聚合、参数同步

**API**：
```cpp
HcclResult HcclAllReduce(
    void *sendBuf,                   // 发送缓冲区
    void *recvBuf,                   // 接收缓冲区
    uint64_t count,                  // 数据元素个数
    HcclDataType dataType,           // 数据类型
    HcclReduceOp op,                 // 归约操作
    HcclComm comm,                   // 通信域
    aclrtStream stream               // 执行流
);
```

**完整示例**：
```cpp
#include "hccl/hccl.h"
#include <vector>
#include <iostream>

int main() {
    // 1. 初始化设备
    int deviceCount = 0;
    aclrtGetDeviceCount(&deviceCount);
    
    // 2. 初始化通信域
    HcclComm comm;
    HcclCommInitAll(&comm, deviceCount);
    
    // 3. 获取当前rank
    int rank;
    HcclGetRank(comm, &rank);
    
    // 4. 准备数据
    const int size = 1024 * 1024;  // 1MB
    std::vector<float> sendBuf(size, static_cast<float>(rank));
    std::vector<float> recvBuf(size);
    
    // 5. 创建执行流
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    // 6. 执行AllReduce（求和）
    HcclResult result = HcclAllReduce(
        sendBuf.data(),
        recvBuf.data(),
        size,
        HCCL_DATA_TYPE_FLOAT32,
        HCCL_REDUCE_SUM,
        comm,
        stream
    );
    
    if (result != HCCL_SUCCESS) {
        std::cerr << "AllReduce failed: " << result << std::endl;
        return -1;
    }
    
    // 7. 同步等待完成
    aclrtSynchronizeStream(stream);
    
    // 8. 验证结果
    std::cout << "Rank " << rank << ": result[0] = " << recvBuf[0] << std::endl;
    
    // 9. 清理资源
    aclrtDestroyStream(stream);
    HcclCommDestroy(comm);
    
    return 0;
}
```

### 3.2 AllGather - 全收集

**功能**：收集所有进程数据到每个进程

**应用场景**：模型并行中的参数收集

**API**：
```cpp
HcclResult HcclAllGather(
    void *sendBuf,                   // 发送缓冲区
    void *recvBuf,                   // 接收缓冲区
    uint64_t sendCount,              // 发送数据个数
    HcclDataType dataType,           // 数据类型
    HcclComm comm,                   // 通信域
    aclrtStream stream               // 执行流
);
```

**示例**：
```cpp
// 每个rank发送1MB数据，接收8MB数据
const int sendSize = 1024 * 1024;
const int recvSize = sendSize * deviceCount;

std::vector<float> sendBuf(sendSize, static_cast<float>(rank));
std::vector<float> recvBuf(recvSize);

HcclAllGather(
    sendBuf.data(),
    recvBuf.data(),
    sendSize,
    HCCL_DATA_TYPE_FLOAT32,
    comm,
    stream
);
```

### 3.3 Broadcast - 广播

**功能**：从根进程广播数据到所有进程

**应用场景**：参数初始化、模型分发

**API**：
```cpp
HcclResult HcclBroadcast(
    void *sendBuf,                   // 发送缓冲区（根进程）
    void *recvBuf,                   // 接收缓冲区（非根进程）
    uint64_t count,                  // 数据个数
    HcclDataType dataType,           // 数据类型
    int root,                        // 根进程rank
    HcclComm comm,                   // 通信域
    aclrtStream stream               // 执行流
);
```

**示例**：
```cpp
// 从rank 0广播参数到所有rank
std::vector<float> params(paramSize);

if (rank == 0) {
    // 根进程初始化参数
    InitParams(params.data(), paramSize);
}

HcclBroadcast(
    params.data(),                   // 根进程发送
    params.data(),                   // 非根进程接收
    paramSize,
    HCCL_DATA_TYPE_FLOAT32,
    0,                               // root rank
    comm,
    stream
);
```

### 3.4 ReduceScatter - 归约分散

**功能**：归约所有进程数据，按块分散到各进程

**应用场景**：分布式损失计算、梯度分片

**API**：
```cpp
HcclResult HcclReduceScatter(
    void *sendBuf,                   // 发送缓冲区
    void *recvBuf,                   // 接收缓冲区
    uint64_t recvCount,              // 接收数据个数
    HcclDataType dataType,           // 数据类型
    HcclReduceOp op,                 // 归约操作
    HcclComm comm,                   // 通信域
    aclrtStream stream               // 执行流
);
```

**示例**：
```cpp
// 每个rank发送8MB数据，接收1MB数据
const int sendSize = 1024 * 1024 * 8;
const int recvSize = sendSize / deviceCount;

std::vector<float> sendBuf(sendSize);
std::vector<float> recvBuf(recvSize);

HcclReduceScatter(
    sendBuf.data(),
    recvBuf.data(),
    recvSize,
    HCCL_DATA_TYPE_FLOAT32,
    HCCL_REDUCE_SUM,
    comm,
    stream
);
```

---

## 4. 通信算法

### 4.1 算法对比

| 算法 | 适用场景 | 特点 |
|------|----------|------|
| Ring | 中小规模集群 | 抗拥塞 |
| Mesh | 大规模单机 | 并行度高 |
| RHD | 多机通信 | 步数少 |
| Star | 小规模 | 延迟低 |

### 4.2 算法选择策略

```cpp
// HCCL会自动选择最优算法，也可以手动指定

// 通过环境变量指定
export HCCL_ALGO_SELECT=ring

// 通过代码指定
SetAlgorithm("mesh");
```

---

## 5. 完整开发流程

### 5.1 标准开发流程

```cpp
#include "hccl/hccl.h"

int main() {
    // 1. 初始化设备
    int deviceCount;
    aclrtGetDeviceCount(&deviceCount);
    aclrtSetDevice(0);
    
    // 2. 初始化通信域
    HcclComm comm;
    HcclCommInitAll(&comm, deviceCount);
    
    // 3. 获取rank信息
    int rank, size;
    HcclGetRank(comm, &rank);
    HcclGetSize(comm, &size);
    
    // 4. 分配内存
    size_t dataSize = 1024 * 1024;  // 1MB
    void* sendBuf = AllocDeviceMem(dataSize);
    void* recvBuf = AllocDeviceMem(dataSize);
    
    // 5. 初始化数据
    InitData(sendBuf, dataSize, rank);
    
    // 6. 创建流
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    // 7. 执行通信
    HcclAllReduce(sendBuf, recvBuf, dataSize, 
                  HCCL_DATA_TYPE_FLOAT32,
                  HCCL_REDUCE_SUM, comm, stream);
    
    // 8. 同步等待
    aclrtSynchronizeStream(stream);
    
    // 9. 验证结果
    VerifyResult(recvBuf, dataSize, size);
    
    // 10. 清理资源
    aclrtDestroyStream(stream);
    FreeDeviceMem(sendBuf);
    FreeDeviceMem(recvBuf);
    HcclCommDestroy(comm);
    
    return 0;
}
```

### 5.2 错误处理

```cpp
HcclResult result = HcclAllReduce(...);
if (result != HCCL_SUCCESS) {
    const char* errorMsg = HcclGetErrorString(result);
    std::cerr << "Error: " << errorMsg << std::endl;
    
    // 常见错误处理
    switch (result) {
        case HCCL_E_INVALID_ARG:
            // 参数错误
            break;
        case HCCL_E_MEMORY:
            // 内存错误
            break;
        case HCCL_E_TIMEOUT:
            // 超时
            break;
        default:
            // 其他错误
            break;
    }
}
```

---

## 6. 常见问题

### 6.1 内存对齐
```cpp
// 确保内存按设备要求对齐
size_t alignment = 4096;  // 4KB对齐
size_t alignedSize = (dataSize + alignment - 1) / alignment * alignment;
void* buf = AllocAlignedMem(alignedSize, alignment);
```

### 6.2 流同步
```cpp
// 确保在通信前完成依赖操作
aclrtSynchronizeStream(stream);

// 确保在通信后等待完成
aclrtSynchronizeStream(stream);
```

### 6.3 性能优化
```cpp
// 使用环境变量优化
export HCCL_BUFFSIZE=524288000  // 500MB
export HCCL_PIPELINE_DEPTH=8    // 流水线深度

// 原地操作减少内存拷贝
HcclAllReduce(buf, buf, count, dataType, op, comm, stream);
```

---

## 7. 学习要点

### 7.1 核心API
- HcclCommInitAll - 初始化通信域
- HcclAllReduce - 全归约
- HcclAllGather - 全收集
- HcclBroadcast - 广播
- HcclReduceScatter - 归约分散

### 7.2 开发流程
1. 初始化设备和通信域
2. 分配内存并初始化数据
3. 创建执行流
4. 执行通信操作
5. 同步等待完成
6. 验证结果
7. 清理资源

### 7.3 注意事项
- 确保所有rank执行相同的操作
- 正确处理错误码
- 合理配置内存大小
- 注意流同步

---

## 8. 下周学习计划

**Week2 - 开发实践与优化**
- 算法选择策略
- 性能调优方法
- 常见问题解决
- 实战案例分析

继续学习，掌握HCCL的高级开发技巧！