# AllReduce 算子详解

## 1. 算子概述

**AllReduce** 是分布式训练中最核心的集合通信算子，执行以下两个阶段：

1. **归约（Reduce）**：将所有进程的数据按指定操作合并
2. **广播（Broadcast）**：将归约结果发送到所有进程

**应用场景**：梯度聚合、参数同步、损失计算

---

## 2. API 接口

```cpp
HcclResult HcclAllReduce(
    void *sendBuf,                   // 输入：发送缓冲区
    void *recvBuf,                   // 输出：接收缓冲区
    uint64_t count,                  // 输入：数据元素个数
    HcclDataType dataType,           // 输入：数据类型
    HcclReduceOp op,                 // 输入：归约操作类型
    HcclComm comm,                   // 输入：通信域
    aclrtStream stream               // 输入：执行流
);
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| sendBuf | void* | 发送缓冲区指针，每个进程存放本地数据 |
| recvBuf | void* | 接收缓冲区指针，存放归约结果 |
| count | uint64_t | 每个进程发送的数据元素个数 |
| dataType | HcclDataType | 数据类型（FLOAT32/FLOAT16/INT32等） |
| op | HcclReduceOp | 归约操作（SUM/PROD/MAX/MIN） |
| comm | HcclComm | 通信域句柄 |
| stream | aclrtStream | 执行流 |

### 数据类型支持

```cpp
typedef enum {
    HCCL_DATA_TYPE_FLOAT32 = 0,      // 32位浮点
    HCCL_DATA_TYPE_FLOAT16 = 1,      // 16位浮点
    HCCL_DATA_TYPE_INT32 = 2,        // 32位整数
    HCCL_DATA_TYPE_INT8 = 3,         // 8位整数
    HCCL_DATA_TYPE_UINT8 = 4,        // 无符号8位整数
    HCCL_DATA_TYPE_BF16 = 5,         // BF16浮点
} HcclDataType;
```

### 归约操作类型

```cpp
typedef enum {
    HCCL_REDUCE_SUM = 0,             // 求和（最常用）
    HCCL_REDUCE_PROD = 1,            // 求积
    HCCL_REDUCE_MAX = 2,             // 求最大值
    HCCL_REDUCE_MIN = 3,             // 求最小值
} HcclReduceOp;
```

---

## 3. 算法实现

### 3.1 Ring AllReduce

**阶段1：Reduce-Scatter**

```cpp
void RingAllReduce::ReduceScatter(
    std::vector<Buffer>& buffers,
    int rank, int size, size_t chunkSize) {
    
    // 初始化：复制自己的数据块
    memcpy(buffers[rank].Data() + rank * chunkSize,
           sendBuffers[rank].Data() + rank * chunkSize,
           chunkSize);
    
    for (int step = 0; step < size - 1; step++) {
        // 计算发送和接收目标
        int sendRank = (rank - step + size) % size;
        int recvRank = (rank - step - 1 + size) % size;
        
        // 异步发送
        SendAsync(buffers[rank].Data() + sendRank * chunkSize, 
                  chunkSize, sendRank);
        
        // 异步接收
        RecvAsync(buffers[rank].Data() + recvRank * chunkSize, 
                  chunkSize, recvRank);
        
        // 等待接收完成
        WaitRecv();
        
        // 归约：recv = recv OP send
        ReduceInPlace(buffers[rank].Data() + recvRank * chunkSize,
                      buffers[rank].Data() + sendRank * chunkSize,
                      chunkSize, reduceOp_);
    }
}
```

**阶段2：AllGather**

```cpp
void RingAllReduce::AllGather(
    std::vector<Buffer>& buffers,
    int rank, int size, size_t chunkSize) {
    
    for (int step = 0; step < size - 1; step++) {
        int sendRank = (rank - step + size) % size;
        int recvRank = (rank - step - 1 + size) % size;
        
        // 发送当前拥有的数据块
        SendAsync(buffers[rank].Data() + sendRank * chunkSize,
                  chunkSize, sendRank);
        
        // 接收数据块
        RecvAsync(buffers[rank].Data() + recvRank * chunkSize,
                  chunkSize, recvRank);
        
        // 等待完成
        WaitAll();
    }
}
```

### 3.2 RHD AllReduce

```cpp
void RHDAllReduce::Execute(
    const std::vector<Buffer>& sendBuffers,
    std::vector<Buffer>& recvBuffers,
    HcclComm comm) {
    
    int rank = comm->GetRank();
    int size = comm->GetSize();
    
    // 复制输入数据
    memcpy(recvBuffers[rank].Data(), sendBuffers[rank].Data(), totalSize_);
    
    // Reduce阶段：递归二分归约
    for (int step = 1; step < size; step *= 2) {
        int partner = rank ^ step;
        
        if (partner < size) {
            // 发送数据到partner
            Send(recvBuffers[rank].Data(), totalSize_, partner);
            
            // 接收partner数据
            Buffer temp(totalSize_);
            Recv(temp.Data(), totalSize_, partner);
            
            // 归约
            ReduceInPlace(recvBuffers[rank].Data(), temp.Data(), totalSize_, reduceOp_);
        }
    }
    
    // Broadcast阶段：递归二分广播
    for (int step = size / 2; step > 0; step /= 2) {
        int partner = rank ^ step;
        
        if (partner < size && rank < partner) {
            Send(recvBuffers[rank].Data(), totalSize_, partner);
        } else if (partner < size) {
            Recv(recvBuffers[rank].Data(), totalSize_, partner);
        }
    }
}
```

---

## 4. 性能分析

### 4.1 时间复杂度

| 算法 | 通信步数 | 复杂度 |
|------|----------|--------|
| Ring | 2*(n-1) | O(n) |
| RHD | 2*log2(n) | O(log n) |
| Star | 1 | O(1) |

### 4.2 Hockney 模型

```cpp
// Ring AllReduce 时间估算
double RingAllReduceTime(int n, double alpha, double beta, size_t dataSize) {
    size_t chunkSize = dataSize / n;
    // Reduce-Scatter: (n-1)*(α + β*chunkSize)
    // AllGather: (n-1)*(α + β*chunkSize)
    return 2 * (n - 1) * (alpha + beta * chunkSize);
}

// RHD AllReduce 时间估算
double RHDAllReduceTime(int n, double alpha, double beta, size_t dataSize) {
    int steps = static_cast<int>(std::log2(n));
    // Reduce: steps*(α + β*dataSize)
    // Broadcast: steps*(α + β*dataSize)
    return 2 * steps * (alpha + beta * dataSize);
}
```

### 4.3 性能对比

```
数据量: 1MB, 节点数: 8

Ring:
  chunkSize = 128KB
  时间 = 2*7*(α + β*128KB)
       ≈ 14*(1μs + 128KB/10GB/s)
       ≈ 14*(1μs + 12.8μs)
       ≈ 193μs

RHD:
  steps = 3
  时间 = 2*3*(α + β*1MB)
       ≈ 6*(1μs + 100μs)
       ≈ 606μs
```

---

## 5. 使用示例

### 5.1 基本使用

```cpp
#include "hccl/hccl.h"
#include <vector>

int main() {
    // 初始化设备
    int deviceCount = 0;
    aclrtGetDeviceCount(&deviceCount);
    
    // 初始化通信域
    HcclComm comm;
    HcclCommInitAll(&comm, deviceCount);
    
    // 获取当前rank
    int rank;
    HcclGetRank(comm, &rank);
    
    // 准备数据
    const int size = 1024 * 1024;  // 1MB
    std::vector<float> sendBuf(size, static_cast<float>(rank));
    std::vector<float> recvBuf(size);
    
    // 创建执行流
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    // 执行AllReduce（求和）
    HcclResult result = HcclAllReduce(
        sendBuf.data(),
        recvBuf.data(),
        size,
        HCCL_DATA_TYPE_FLOAT32,
        HCCL_REDUCE_SUM,
        comm,
        stream
    );
    
    // 同步等待完成
    aclrtSynchronizeStream(stream);
    
    // 验证结果（所有rank的recvBuf[0]应该相等）
    std::cout << "Rank " << rank << ": result[0] = " << recvBuf[0] << std::endl;
    
    // 清理资源
    aclrtDestroyStream(stream);
    HcclCommDestroy(comm);
    
    return 0;
}
```

### 5.2 高级配置

```cpp
// 设置环境变量优化性能
setenv("HCCL_BUFFSIZE", "524288000", 1);  // 500MB
setenv("HCCL_PIPELINE_DEPTH", "8", 1);     // 流水线深度

// 创建大缓冲区
const size_t largeSize = 100 * 1024 * 1024;  // 100MB
std::vector<float> largeBuf(largeSize);

// 执行AllReduce（求最大值）
HcclAllReduce(
    largeBuf.data(),
    largeBuf.data(),  // 原地操作
    largeSize,
    HCCL_DATA_TYPE_FLOAT32,
    HCCL_REDUCE_MAX,
    comm,
    stream
);
```

---

## 6. 常见问题与优化

### 6.1 性能瓶颈分析

| 瓶颈类型 | 表现 | 解决方法 |
|----------|------|----------|
| 网络带宽 | 带宽利用率低 | 使用Mesh/RHD算法 |
| 内存带宽 | 内存拷贝慢 | 启用零拷贝 |
| 算法选择 | 不适合当前场景 | 动态算法选择 |
| 同步开销 | 等待时间长 | 异步通信 |

### 6.2 优化建议

1. **选择合适算法**：根据数据量和节点数选择
2. **启用流水线**：大数据量时启用
3. **调整Buffer大小**：根据数据量配置
4. **异步执行**：使用stream并发执行

---

## 7. 总结

AllReduce 是分布式训练的核心算子，HCCL 提供了多种优化算法：

- **Ring**：适用于中小规模集群
- **RHD**：适用于大规模多机集群
- **Star**：适用于小规模或小数据量

通过合理配置和选择，可以充分发挥昇腾 AI 处理器的通信性能。