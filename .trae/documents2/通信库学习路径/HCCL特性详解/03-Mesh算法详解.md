# Mesh 算法详解

## 1. 算法概述

**Mesh** 算法将通信节点排列成二维网格拓扑，分 X 方向和 Y 方向两轮完成通信：

- 每个节点与四个方向的邻居通信（上下左右）
- 适用于 Server 内多卡通信
- 并行度高，带宽利用率好

**特点**：
- 通信步数：2√n 步
- 复杂度：O(√n)
- 优点：并行度高，适合大规模单机多卡
- 缺点：需要完整的网格拓扑

---

## 2. 算法原理

### 2.1 拓扑结构

```
Mesh 拓扑示例（8节点，2x4网格）：

    X方向 →
  ┌─────┬─────┬─────┬─────┐
Y │  0  │  1  │  2  │  3  │
↑ ├─────┼─────┼─────┼─────┤
│ │  4  │  5  │  6  │  7  │
  └─────┴─────┴─────┴─────┘

节点坐标：
  rank = y * dimX + x
  x = rank % dimX
  y = rank / dimX

邻居关系：
  水平邻居：(x±1, y)
  垂直邻居：(x, y±1)
```

### 2.2 通信过程

**AllGather Mesh 过程**：

```
初始状态：每个rank有自己的数据块
  Rank:  0   1   2   3   4   5   6   7
  Data: [a] [b] [c] [d] [e] [f] [g] [h]

X方向通信（行内交换）：
  Step 1: 0↔1, 2↔3, 4↔5, 6↔7
  Result: [ab] [ab] [cd] [cd] [ef] [ef] [gh] [gh]
  
  Step 2: 0↔2, 1↔3, 4↔6, 5↔7
  Result: [abcd] [abcd] [abcd] [abcd] [efgh] [efgh] [efgh] [efgh]

Y方向通信（列内交换）：
  Step 1: 0↔4, 1↔5, 2↔6, 3↔7
  Result: [abcdefgh] * 8
```

---

## 3. 代码实现

### 3.1 算法类结构

```cpp
class MeshAllGather : public Algorithm {
public:
    void Execute(
        const std::vector<Buffer>& sendBuffers,
        std::vector<Buffer>& recvBuffers,
        const TaskInfo& taskInfo,
        HcclComm comm) override;
    
private:
    void XDirectionComm(
        std::vector<Buffer>& buffers,
        int rank, int dimX, int dimY,
        size_t dataSize, HcclComm comm);
    
    void YDirectionComm(
        std::vector<Buffer>& buffers,
        int rank, int dimX, int dimY,
        size_t dataSize, HcclComm comm);
};
```

### 3.2 主执行函数

```cpp
void MeshAllGather::Execute(
    const std::vector<Buffer>& sendBuffers,
    std::vector<Buffer>& recvBuffers,
    const TaskInfo& taskInfo,
    HcclComm comm) {
    
    int rank = comm->GetRank();
    int size = comm->GetSize();
    
    // 计算网格维度
    int dimX = static_cast<int>(std::sqrt(static_cast<float>(size)));
    int dimY = size / dimX;
    
    size_t totalSize = taskInfo.count * GetDataTypeSize(taskInfo.dataType);
    
    // 初始化：复制自己的数据
    memcpy(recvBuffers[rank].Data(), sendBuffers[rank].Data(), totalSize);
    
    // X方向通信
    XDirectionComm(recvBuffers, rank, dimX, dimY, totalSize, comm);
    
    // Y方向通信
    YDirectionComm(recvBuffers, rank, dimX, dimY, totalSize, comm);
}
```

### 3.3 X 方向通信

```cpp
void MeshAllGather::XDirectionComm(
    std::vector<Buffer>& buffers,
    int rank, int dimX, int dimY,
    size_t dataSize, HcclComm comm) {
    
    int x = rank % dimX;
    int y = rank / dimX;
    
    for (int offset = 1; offset < dimX; offset++) {
        // 计算目标节点
        int targetX = (x + offset) % dimX;
        int targetRank = targetX + y * dimX;
        
        // 交换数据
        Exchange(buffers[rank].Data(),
                 buffers[targetRank].Data(),
                 dataSize,
                 rank, targetRank,
                 comm);
    }
}
```

### 3.4 Y 方向通信

```cpp
void MeshAllGather::YDirectionComm(
    std::vector<Buffer>& buffers,
    int rank, int dimX, int dimY,
    size_t dataSize, HcclComm comm) {
    
    int x = rank % dimX;
    int y = rank / dimX;
    
    for (int offset = 1; offset < dimY; offset++) {
        // 计算目标节点
        int targetY = (y + offset) % dimY;
        int targetRank = x + targetY * dimX;
        
        // 交换数据
        Exchange(buffers[rank].Data(),
                 buffers[targetRank].Data(),
                 dataSize,
                 rank, targetRank,
                 comm);
    }
}
```

### 3.5 数据交换函数

```cpp
void MeshAllGather::Exchange(
    void* buf1, void* buf2, size_t size,
    int rank1, int rank2, HcclComm comm) {
    
    // 异步发送/接收
    CommRequest sendReq = comm->SendAsync(buf1, size, rank2);
    CommRequest recvReq = comm->RecvAsync(buf2, size, rank1);
    
    // 等待完成
    comm->Wait({sendReq, recvReq});
    
    // 合并数据（保留双方数据）
    MergeData(buf1, buf2, size);
}
```

---

## 4. 性能分析

### 4.1 时间复杂度

```cpp
// Mesh AllGather 时间估算
double CalculateMeshTime(int dimX, int dimY, double alpha, double beta, size_t dataSize) {
    // X方向: dimX-1 步
    // Y方向: dimY-1 步
    // 每步时间: α + β * dataSize
    int totalSteps = (dimX - 1) + (dimY - 1);
    return totalSteps * (alpha + beta * dataSize);
}
```

### 4.2 性能对比

```
场景：8节点(2x4), 1MB数据

Ring:
  时间 = 2 * 7 * (α + β * 128KB)
       ≈ 14 * (1μs + 12.8μs)
       ≈ 193μs

Mesh:
  时间 = (2-1 + 4-1) * (α + β * 1MB)
       = 4 * (1μs + 100μs)
       = 404μs

但Mesh可以并行执行多条链路，实际性能更好！
```

### 4.3 并行度分析

```
Mesh 并行度示例（2x4）：

X方向Step 1: 0↔1, 2↔3, 4↔5, 6↔7  (4对并行)
X方向Step 2: 0↔2, 1↔3, 4↔6, 5↔7  (4对并行)

Y方向Step 1: 0↔4, 1↔5, 2↔6, 3↔7  (4对并行)

总并行度：min(dimX, dimY) 对
```

---

## 5. 优化策略

### 5.1 自适应网格

**原理**：根据实际拓扑动态调整网格维度

```cpp
class AdaptiveMesh : public MeshAlgorithm {
public:
    void Execute(...) override {
        // 获取实际拓扑信息
        TopoInfo topoInfo = GetTopologyInfo();
        
        // 根据实际链路带宽选择最优维度
        int optimalDimX = CalculateOptimalDimension(topoInfo);
        
        // 使用最优维度执行
        ExecuteWithDimension(optimalDimX);
    }
};
```

### 5.2 分层 Mesh

**原理**：将大规模集群分成多个子网格

```cpp
class HierarchicalMesh : public Algorithm {
public:
    void Execute(...) override {
        // 第一层：子网格内通信
        for (auto& subMesh : subMeshes_) {
            subMesh->Execute();
        }
        
        // 第二层：子网格间通信
        InterMeshComm();
    }
};
```

---

## 6. 使用场景

### 6.1 适用场景

| 场景 | 说明 |
|------|------|
| 大规模单机多卡 | 并行度高 |
| 高带宽环境 | 充分利用带宽 |
| 规则拓扑 | 网格结构规整 |
| 大数据量 | 传输效率高 |

### 6.2 不适用场景

| 场景 | 原因 |
|------|------|
| 小规模集群 | 开销相对较大 |
| 不规则拓扑 | 需要完整网格 |
| 网络拥塞 | 多条链路竞争 |

---

## 7. 总结

Mesh 算法是一种高并行度的集合通信算法：

**优点**：
- 并行度高
- 带宽利用率好
- 适合大规模单机多卡

**缺点**：
- 需要完整网格拓扑
- 小规模场景开销较大

**适用场景**：
- 大规模单机多卡集群
- 高带宽环境
- 大数据量通信

通过自适应网格和分层技术，可以进一步提升Mesh算法的适用性和性能。