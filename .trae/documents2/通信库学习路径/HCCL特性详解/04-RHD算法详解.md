# RHD 算法详解

## 1. 算法概述

**RHD**（Recursive Halving-Doubling）递归二分倍增算法是一种高效的集合通信算法：

- 使用异或操作寻找配对节点
- 分 Reduce 和 Broadcast 两个阶段
- 适用于节点数为 2 的幂次的场景

**特点**：
- 通信步数：log2(n) 步
- 复杂度：O(log n)
- 优点：通信步数最少，时延低
- 缺点：非 2 幂次节点会引入额外通信量

---

## 2. 算法原理

### 2.1 配对机制

RHD 使用**异或操作**来寻找配对节点：

```cpp
// 配对公式
partner = rank ^ step

// step 取值：1, 2, 4, 8, ..., n/2
```

**配对示例（8节点）**：

```
step=1:
  0 ↔ 1 (0^1=1, 1^1=0)
  2 ↔ 3 (2^1=3, 3^1=2)
  4 ↔ 5 (4^1=5, 5^1=4)
  6 ↔ 7 (6^1=7, 7^1=6)

step=2:
  0 ↔ 2 (0^2=2, 2^2=0)
  1 ↔ 3 (1^2=3, 3^2=1)
  4 ↔ 6 (4^2=6, 6^2=4)
  5 ↔ 7 (5^2=7, 7^2=5)

step=4:
  0 ↔ 4 (0^4=4, 4^4=0)
  1 ↔ 5 (1^4=5, 5^4=1)
  2 ↔ 6 (2^4=6, 6^4=2)
  3 ↔ 7 (3^4=7, 7^4=3)
```

### 2.2 Reduce 阶段

**目标**：将所有节点的数据归约到根节点

```
初始状态（8节点）：
  Rank:  0   1   2   3   4   5   6   7
  Data: [a] [b] [c] [d] [e] [f] [g] [h]

step=1 (0↔1, 2↔3, 4↔5, 6↔7):
  Result: [a+b] [a+b] [c+d] [c+d] [e+f] [e+f] [g+h] [g+h]

step=2 (0↔2, 1↔3, 4↔6, 5↔7):
  Result: [a+b+c+d] [a+b+c+d] [a+b+c+d] [a+b+c+d] [e+f+g+h] [e+f+g+h] [e+f+g+h] [e+f+g+h]

step=4 (0↔4, 1↔5, 2↔6, 3↔7):
  Result: [all] [all] [all] [all] [all] [all] [all] [all]
```

### 2.3 Broadcast 阶段

**目标**：将根节点的数据广播到所有节点

```
初始状态：只有rank 0有完整数据
  Rank:  0      1      2      3      4      5      6      7
  Data: [all] [----] [----] [----] [----] [----] [----] [----]

step=4 (0→4, 1→5, 2→6, 3→7):
  Result: [all] [all] [all] [all] [all] [----] [----] [----]

step=2 (0→2, 1→3, 4→6, 5→7):
  Result: [all] [all] [all] [all] [all] [all] [all] [----]

step=1 (0→1, 2→3, 4→5, 6→7):
  Result: [all] [all] [all] [all] [all] [all] [all] [all]
```

---

## 3. 代码实现

### 3.1 算法类结构

```cpp
class RHDAllReduce : public Algorithm {
public:
    RHDAllReduce(HcclReduceOp op) : reduceOp_(op) {}
    
    void Execute(
        const std::vector<Buffer>& sendBuffers,
        std::vector<Buffer>& recvBuffers,
        const TaskInfo& taskInfo,
        HcclComm comm) override;
    
private:
    void ReducePhase(
        std::vector<Buffer>& buffers,
        int rank, int size, size_t dataSize, HcclComm comm);
    
    void BroadcastPhase(
        std::vector<Buffer>& buffers,
        int rank, int size, size_t dataSize, HcclComm comm);
    
    HcclReduceOp reduceOp_;
};
```

### 3.2 Reduce 阶段实现

```cpp
void RHDAllReduce::ReducePhase(
    std::vector<Buffer>& buffers,
    int rank, int size, size_t dataSize, HcclComm comm) {
    
    for (int step = 1; step < size; step *= 2) {
        int partner = rank ^ step;
        
        if (partner < size) {
            // 确定发送/接收方向
            bool isSender = (rank < partner);
            
            if (isSender) {
                // 发送数据
                comm->Send(buffers[rank].Data(), dataSize, partner);
                
                // 接收数据
                Buffer temp(dataSize);
                comm->Recv(temp.Data(), dataSize, partner);
                
                // 归约：recv OP send
                Reduce(buffers[rank].Data(), temp.Data(), dataSize, reduceOp_);
            } else {
                // 接收数据
                Buffer temp(dataSize);
                comm->Recv(temp.Data(), dataSize, partner);
                
                // 发送数据
                comm->Send(buffers[rank].Data(), dataSize, partner);
                
                // 归约：recv OP send
                Reduce(temp.Data(), buffers[rank].Data(), dataSize, reduceOp_);
                memcpy(buffers[rank].Data(), temp.Data(), dataSize);
            }
        }
    }
}
```

### 3.3 Broadcast 阶段实现

```cpp
void RHDAllReduce::BroadcastPhase(
    std::vector<Buffer>& buffers,
    int rank, int size, size_t dataSize, HcclComm comm) {
    
    for (int step = size / 2; step > 0; step /= 2) {
        int partner = rank ^ step;
        
        if (partner < size) {
            if (rank < partner) {
                // 发送数据给partner
                comm->Send(buffers[rank].Data(), dataSize, partner);
            } else {
                // 从partner接收数据
                comm->Recv(buffers[rank].Data(), dataSize, partner);
            }
        }
    }
}
```

### 3.4 主执行函数

```cpp
void RHDAllReduce::Execute(
    const std::vector<Buffer>& sendBuffers,
    std::vector<Buffer>& recvBuffers,
    const TaskInfo& taskInfo,
    HcclComm comm) {
    
    int rank = comm->GetRank();
    int size = comm->GetSize();
    
    size_t totalSize = taskInfo.count * GetDataTypeSize(taskInfo.dataType);
    
    // 初始化：复制输入数据
    memcpy(recvBuffers[rank].Data(), sendBuffers[rank].Data(), totalSize);
    
    // Step 1: Reduce阶段
    ReducePhase(recvBuffers, rank, size, totalSize, comm);
    
    // Step 2: Broadcast阶段
    BroadcastPhase(recvBuffers, rank, size, totalSize, comm);
}
```

---

## 4. 性能分析

### 4.1 时间复杂度

```cpp
// RHD AllReduce 时间估算
double CalculateRHDTime(int n, double alpha, double beta, size_t dataSize) {
    int steps = static_cast<int>(std::log2(n));
    // Reduce: steps * (α + β * dataSize)
    // Broadcast: steps * (α + β * dataSize)
    return 2.0 * steps * (alpha + beta * dataSize);
}
```

### 4.2 性能对比

```
场景：8节点，1MB数据

Ring:
  时间 = 2 * 7 * (α + β * 128KB)
       ≈ 14 * (1μs + 12.8μs)
       ≈ 193μs

RHD:
  时间 = 2 * 3 * (α + β * 1MB)
       = 6 * (1μs + 100μs)
       = 606μs

结论：RHD在大数据量时有优势
```

```
场景：8节点，10MB数据

Ring:
  时间 = 2 * 7 * (α + β * 1.25MB)
       ≈ 14 * (1μs + 125μs)
       ≈ 1764μs

RHD:
  时间 = 2 * 3 * (α + β * 10MB)
       = 6 * (1μs + 1000μs)
       = 6006μs

结论：Ring在大数据量时反而更快！
```

### 4.3 算法选择边界

```cpp
// RHD vs Ring 选择边界
size_t CalculateBreakEvenPoint(int n, double alpha, double beta) {
    // 解方程：2*(n-1)*(α + β*x/n) = 2*log2(n)*(α + β*x)
    // x = (n-1)*α / (log2(n)*β*n - β*(n-1))
    
    double stepsRing = n - 1;
    double stepsRHD = std::log2(n);
    
    double numerator = stepsRing * alpha;
    double denominator = stepsRHD * beta * n - beta * stepsRing;
    
    return static_cast<size_t>(numerator / denominator);
}
```

---

## 5. 非 2 幂次处理

### 5.1 补齐策略

**原理**：将节点数补齐到最近的 2 幂次

```cpp
class RHDWithPadding : public RHDAllReduce {
public:
    void Execute(...) override {
        int actualSize = comm->GetSize();
        
        // 计算补齐后的大小
        int paddedSize = NextPowerOfTwo(actualSize);
        
        if (paddedSize != actualSize) {
            // 补齐处理
            ExecuteWithPadding(actualSize, paddedSize);
        } else {
            // 正常执行
            RHDAllReduce::Execute(...);
        }
    }
};
```

### 5.2 NHR 算法

**原理**：非均衡层次环算法，专门处理非 2 幂次场景

```cpp
class NHRAllReduce : public Algorithm {
public:
    void Execute(...) override {
        // 分层处理
        // 第一层：形成完整的环
        // 第二层：处理剩余节点
    }
};
```

---

## 6. 使用场景

### 6.1 适用场景

| 场景 | 说明 |
|------|------|
| Server间通信 | 跨节点延迟高，需要减少步数 |
| 2幂次节点 | 算法效率最高 |
| 中等数据量 | 平衡延迟和带宽 |

### 6.2 不适用场景

| 场景 | 原因 |
|------|------|
| 非2幂次节点 | 需要额外处理 |
| 小数据量 | 开销相对较大 |
| 网络拥塞 | 多对节点同时通信 |

---

## 7. 总结

RHD 算法是一种高效的集合通信算法：

**优点**：
- 通信步数最少（log2(n)）
- 时延低
- 适合大规模多机集群

**缺点**：
- 非2幂次节点需要额外处理
- 小数据量时开销较大

**适用场景**：
- Server间通信
- 2幂次节点集群
- 中等数据量通信

通过补齐策略或使用NHR算法，可以处理非2幂次节点的情况。