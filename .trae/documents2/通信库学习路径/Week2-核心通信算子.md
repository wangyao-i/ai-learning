# Week2 - 核心通信算子

## 1. AllReduce 算子

### 1.1 功能概述

**AllReduce** 是分布式训练中最常用的集合通信操作，执行以下两个步骤：
1. **归约（Reduce）**：将所有进程的数据按指定操作（求和、求积、最大、最小）合并
2. **广播（Broadcast）**：将归约结果发送到所有进程

**应用场景**：梯度聚合、参数同步

### 1.2 算法选择策略

HCCL 根据以下条件选择最优算法：

| 条件 | 选择算法 | 原因 |
|------|----------|------|
| 小数据量（< 64KB） | Star | 单步完成，延迟低 |
| 中等数据量 + Server内 | Ring | 通信关系简单 |
| 大数据量 + Server内 | Mesh | 并行度高 |
| Server间 + 2的幂次节点 | RHD | 对数复杂度 |
| Server间 + 非2幂次节点 | NHR/NB | 层次化优化 |

### 1.3 Ring AllReduce 实现

**阶段1：Reduce-Scatter**
```cpp
void RingAllReduce::ReduceScatterPhase(
    const std::vector<DeviceBuffer>& buffers,
    int rank, int size, HcclComm comm) {
    
    size_t chunkSize = totalSize / size;
    
    for (int step = 0; step < size - 1; step++) {
        // 计算发送和接收目标
        int sendRank = (rank - step + size) % size;
        int recvRank = (rank - step - 1 + size) % size;
        
        // 获取数据块指针
        void* sendChunk = buffers[sendRank].GetData() + sendRank * chunkSize;
        void* recvChunk = buffers[recvRank].GetData() + recvRank * chunkSize;
        
        // 异步发送
        comm->Send(sendChunk, chunkSize, sendRank);
        
        // 异步接收
        comm->Recv(recvChunk, chunkSize, recvRank);
        
        // 等待接收完成
        comm->WaitRecv();
        
        // 归约操作（原地归约到recvChunk）
        ReduceInPlace(recvChunk, sendChunk, chunkSize, reduceOp_);
    }
}
```

**阶段2：AllGather**
```cpp
void RingAllReduce::AllGatherPhase(
    const std::vector<DeviceBuffer>& buffers,
    int rank, int size, HcclComm comm) {
    
    size_t chunkSize = totalSize / size;
    
    for (int step = 0; step < size - 1; step++) {
        // 计算发送和接收目标
        int sendRank = (rank - step + size) % size;
        int recvRank = (rank - step - 1 + size) % size;
        
        // 获取数据块指针
        void* sendChunk = buffers[sendRank].GetData() + sendRank * chunkSize;
        void* recvChunk = buffers[recvRank].GetData() + recvRank * chunkSize;
        
        // 异步发送当前拥有的数据块
        comm->Send(sendChunk, chunkSize, sendRank);
        
        // 异步接收数据块
        comm->Recv(recvChunk, chunkSize, recvRank);
        
        // 等待完成
        comm->WaitAll();
    }
}
```

### 1.4 性能分析

**复杂度分析**：
- **时间复杂度**：O(n) 步通信，每步 O(1)
- **数据传输量**：2*(n-1)*size 字节（Reduce-Scatter + AllGather）
- **适用场景**：中等数据量、节点数较少

**Hockney 模型**：
```
时间 = α*(2n-2) + β*(2*(n-1)*dataSize) + γ*dataSize
```
- α：节点间固定延迟
- β：每字节传输耗时
- γ：每字节归约计算耗时

---

## 2. AllGather 算子

### 2.1 功能概述

**AllGather** 将所有进程的数据收集并分发到每个进程：
- 输入：每个进程有 size/n 数据
- 输出：每个进程有完整的 size 数据

**应用场景**：模型并行中的参数收集、数据并行中的梯度聚合

### 2.2 Mesh AllGather 实现

**Mesh 拓扑特点**：
- 将 n 个节点排列成 √n × √n 的网格
- 每个节点与上下左右邻居通信
- 分 X 方向和 Y 方向两轮完成

```cpp
void MeshAllGather::Execute(
    const std::vector<DeviceBuffer>& inputBuffers,
    std::vector<DeviceBuffer>& outputBuffers,
    HcclComm comm) {
    
    int rank = comm->GetRank();
    int size = comm->GetSize();
    
    // 计算网格维度
    int dimX = static_cast<int>(std::sqrt(static_cast<float>(size)));
    int dimY = size / dimX;
    
    int x = rank % dimX;
    int y = rank / dimX;
    
    // 初始化：每个rank将自己的数据放入outputBuffer
    memcpy(outputBuffers[rank].GetData(), 
           inputBuffers[rank].GetData(), 
           inputBuffers[rank].GetSize());
    
    // X方向通信：同一行内交换数据
    for (int i = 0; i < dimX; i++) {
        if (i == x) continue;
        
        int targetRank = i + y * dimX;
        size_t sendSize = inputBuffers[rank].GetSize();
        
        // 异步交换数据
        comm->SendRecv(
            outputBuffers[rank].GetData(), sendSize, targetRank,
            outputBuffers[targetRank].GetData(), sendSize, targetRank
        );
    }
    
    // Y方向通信：同一列内交换数据
    for (int j = 0; j < dimY; j++) {
        if (j == y) continue;
        
        int targetRank = x + j * dimX;
        size_t sendSize = inputBuffers[rank].GetSize();
        
        // 异步交换数据
        comm->SendRecv(
            outputBuffers[rank].GetData(), sendSize, targetRank,
            outputBuffers[targetRank].GetData(), sendSize, targetRank
        );
    }
    
    // 等待所有通信完成
    comm->WaitAll();
}
```

### 2.3 性能对比

| 算法 | 通信步数 | 数据传输量 | 适用场景 |
|------|----------|------------|----------|
| Ring | n-1 | (n-1)*dataSize | 节点数少 |
| Mesh | 2√n | 2√n*dataSize | 节点数多 |
| RHD | log2(n) | log2(n)*dataSize | 2幂次节点 |

---

## 3. Broadcast 算子

### 3.1 功能概述

**Broadcast** 将根进程的数据广播到所有进程：
- 输入：根进程有完整数据
- 输出：所有进程获得相同数据

**应用场景**：参数初始化、模型分发

### 3.2 Star Broadcast 实现

**Star 算法**（单步完成）：
```cpp
void StarBroadcast::Execute(
    void* sendBuf, void* recvBuf,
    size_t count, HcclDataType dataType,
    int root, HcclComm comm) {
    
    int rank = comm->GetRank();
    int size = comm->GetSize();
    
    if (rank == root) {
        // 根进程：向所有其他进程发送数据
        std::vector<CommRequest> requests;
        
        for (int i = 0; i < size; i++) {
            if (i != root) {
                requests.push_back(
                    comm->SendAsync(sendBuf, count * GetDataTypeSize(dataType), i)
                );
            }
        }
        
        // 根进程自己复制数据
        memcpy(recvBuf, sendBuf, count * GetDataTypeSize(dataType));
        
        // 等待所有发送完成
        comm->WaitAll(requests);
    } else {
        // 非根进程：从根进程接收数据
        comm->Recv(recvBuf, count * GetDataTypeSize(dataType), root);
    }
}
```

### 3.3 Tree Broadcast 实现

**Tree 算法**（对数步数）：
```cpp
void TreeBroadcast::Execute(
    void* sendBuf, void* recvBuf,
    size_t count, HcclDataType dataType,
    int root, HcclComm comm) {
    
    int rank = comm->GetRank();
    int size = comm->GetSize();
    size_t dataSize = count * GetDataTypeSize(dataType);
    
    if (rank == root) {
        memcpy(recvBuf, sendBuf, dataSize);
    }
    
    // 构建二叉树
    int parent = -1;
    std::vector<int> children;
    
    // 计算父节点和子节点
    CalculateTreeTopology(rank, size, root, parent, children);
    
    // 接收阶段：从父节点接收数据
    if (rank != root && parent != -1) {
        comm->Recv(recvBuf, dataSize, parent);
    }
    
    // 发送阶段：向子节点发送数据
    std::vector<CommRequest> requests;
    for (int child : children) {
        requests.push_back(
            comm->SendAsync(recvBuf, dataSize, child)
        );
    }
    
    // 等待所有发送完成
    comm->WaitAll(requests);
}
```

### 3.4 算法对比

| 算法 | 通信步数 | 根进程带宽 | 适用场景 |
|------|----------|------------|----------|
| Star | 1 | n-1路并发 | 小规模集群 |
| Tree | log2(n) | 均衡负载 | 大规模集群 |

---

## 4. ReduceScatter 算子

### 4.1 功能概述

**ReduceScatter** 先归约所有进程数据，再按块分散：
- 输入：每个进程有完整数据
- 输出：每个进程获得归约结果的一部分

**应用场景**：梯度分片、分布式损失计算

### 4.2 实现原理

```cpp
void ReduceScatter::Execute(
    const std::vector<DeviceBuffer>& inputBuffers,
    DeviceBuffer& outputBuffer,
    size_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm) {
    
    int rank = comm->GetRank();
    int size = comm->GetSize();
    
    size_t elementSize = GetDataTypeSize(dataType);
    size_t totalSize = recvCount * elementSize * size;
    size_t chunkSize = recvCount * elementSize;
    
    // 临时缓冲区：存放中间结果
    DeviceBuffer tempBuffer(totalSize);
    
    // Step 1: AllGather 收集所有数据
    std::vector<DeviceBuffer> gatheredBuffers(size);
    for (int i = 0; i < size; i++) {
        gatheredBuffers[i] = inputBuffers[i];
    }
    
    AllGather(gatheredBuffers, tempBuffer, comm);
    
    // Step 2: 归约并提取本rank的数据块
    void* resultChunk = outputBuffer.GetData();
    void* tempPtr = tempBuffer.GetData() + rank * chunkSize;
    
    // 初始化结果为第一个rank对应块
    memcpy(resultChunk, tempPtr, chunkSize);
    
    // 归约其他rank的对应块
    for (int i = 0; i < size; i++) {
        if (i == rank) continue;
        
        void* srcPtr = tempBuffer.GetData() + i * chunkSize;
        ReduceInPlace(resultChunk, srcPtr, chunkSize, op);
    }
}
```

---

## 5. AlltoAll 算子

### 5.1 功能概述

**AlltoAll** 实现全对全的数据交换：
- 输入：每个进程有 n 块数据，每块发往不同目标
- 输出：每个进程从其他进程接收数据块

**应用场景**：数据并行中的特征交换、模型并行中的层间通信

### 5.2 PairWise AlltoAll 实现

```cpp
void PairWiseAlltoAll::Execute(
    const DeviceBuffer& sendBuf,
    DeviceBuffer& recvBuf,
    size_t count, HcclDataType dataType,
    HcclComm comm) {
    
    int rank = comm->GetRank();
    int size = comm->GetSize();
    
    size_t elementSize = GetDataTypeSize(dataType);
    size_t blockSize = count * elementSize;
    
    std::vector<CommRequest> sendRequests;
    std::vector<CommRequest> recvRequests;
    
    // 与每个其他rank交换数据
    for (int i = 0; i < size; i++) {
        if (i == rank) continue;
        
        // 计算发送和接收的偏移
        size_t sendOffset = i * blockSize;
        size_t recvOffset = i * blockSize;
        
        // 异步发送
        sendRequests.push_back(
            comm->SendAsync(
                sendBuf.GetData() + sendOffset,
                blockSize,
                i
            )
        );
        
        // 异步接收
        recvRequests.push_back(
            comm->RecvAsync(
                recvBuf.GetData() + recvOffset,
                blockSize,
                i
            )
        );
    }
    
    // 等待所有通信完成
    comm->WaitAll(sendRequests);
    comm->WaitAll(recvRequests);
}
```

### 5.3 性能特点

| 指标 | PairWise | Ring-based |
|------|----------|------------|
| 通信步数 | n-1 | n-1 |
| 带宽利用 | 充分 | 受限 |
| 网络拥塞 | 易拥塞 | 较平稳 |
| 适用数据量 | 大 | 小-中 |

---

## 6. 算子注册机制

### 6.1 注册架构

```cpp
// 算子注册模板
template<typename OpType>
class OperatorRegistry {
public:
    // 注册算子
    static void Register(const std::string& name) {
        registry_[name] = []() -> std::unique_ptr<Operator> {
            return std::make_unique<OpType>();
        };
    }
    
    // 获取算子实例
    static std::unique_ptr<Operator> Get(const std::string& name) {
        auto it = registry_.find(name);
        if (it != registry_.end()) {
            return it->second();
        }
        return nullptr;
    }
    
private:
    static std::unordered_map<std::string, 
        std::function<std::unique_ptr<Operator>()>> registry_;
};

// 宏定义简化注册
#define REGISTER_OPERATOR(op_name, op_class) \
    static bool op_class##_registered = []() { \
        OperatorRegistry<op_class>::Register(op_name); \
        return true; \
    }();
```

### 6.2 算法选择器

```cpp
class AlgorithmSelector {
public:
    static std::shared_ptr<Algorithm> Select(
        const TaskInfo& taskInfo,
        const TopoInfo& topoInfo,
        const ResourceInfo& resInfo) {
        
        // 1. 根据数据量选择
        size_t dataSize = taskInfo.count * GetDataTypeSize(taskInfo.dataType);
        
        // 2. 根据节点数选择
        int size = topoInfo.rankCount;
        
        // 3. 根据拓扑类型选择
        TopoType topoType = topoInfo.topoType;
        
        // 4. 应用算法选择策略
        if (dataSize < SMALL_DATA_THRESHOLD) {
            return std::make_shared<StarAlgorithm>();
        } else if (topoType == TopoType::SERVER_INTRA && size <= 8) {
            return std::make_shared<RingAlgorithm>();
        } else if (topoType == TopoType::SERVER_INTRA && size > 8) {
            return std::make_shared<MeshAlgorithm>();
        } else if (IsPowerOfTwo(size)) {
            return std::make_shared<RHDAlgorithm>();
        } else {
            return std::make_shared<NHRAlgorithm>();
        }
    }
};
```

---

## 7. 学习要点总结

### 7.1 算子特性对比

| 算子 | 输入 | 输出 | 复杂度 | 典型应用 |
|------|------|------|--------|----------|
| AllReduce | 各有n数据 | 各有归约结果 | O(log n) | 梯度聚合 |
| AllGather | 各有n/k数据 | 各有完整数据 | O(log n) | 参数收集 |
| Broadcast | 根有数据 | 各有相同数据 | O(log n) | 参数分发 |
| ReduceScatter | 各有完整数据 | 各有部分结果 | O(log n) | 梯度分片 |
| AlltoAll | 各有n块数据 | 各有n块数据 | O(n) | 数据交换 |

### 7.2 算法选择原则

1. **数据量**：小数据用Star，大数据用Ring/Mesh
2. **节点数**：少节点用Ring，多节点用Mesh/RHD
3. **节点规模**：2幂次用RHD，非2幂次用NHR/NB
4. **网络状况**：拥塞严重用Ring，带宽充足用Mesh

### 7.3 下周学习计划

- **Week3**：深入学习通信算法的实现细节
- 重点：Ring、Mesh、RHD 等算法的具体实现