# Week3 - 通信算法实现

## 1. Ring 算法

### 1.1 算法原理

**Ring** 算法将所有节点组织成一个环形拓扑：
- 每个节点只与相邻节点通信
- 数据沿环单向传递
- 适用于 Server 内和 Server 间通信

**特点**：
- 通信步数：n-1 步
- 复杂度：O(n)
- 优点：通信关系简单，抗拥塞能力强
- 缺点：线性复杂度，大集群时延较高

### 1.2 AllReduce Ring 实现

```cpp
class RingAllReduce : public Algorithm {
public:
    RingAllReduce(HcclReduceOp op) : reduceOp_(op) {}
    
    void Execute(
        const std::vector<Buffer>& sendBuffers,
        std::vector<Buffer>& recvBuffers,
        const TaskInfo& taskInfo,
        HcclComm comm) override {
        
        int rank = comm->GetRank();
        int size = comm->GetSize();
        
        // 数据切分
        size_t totalSize = taskInfo.count * GetDataTypeSize(taskInfo.dataType);
        size_t chunkSize = totalSize / size;
        
        // Step 1: Reduce-Scatter
        ReduceScatter(sendBuffers, recvBuffers, chunkSize, rank, size, comm);
        
        // Step 2: AllGather
        AllGather(recvBuffers, chunkSize, rank, size, comm);
    }
    
private:
    void ReduceScatter(
        const std::vector<Buffer>& sendBuffers,
        std::vector<Buffer>& recvBuffers,
        size_t chunkSize, int rank, int size, HcclComm comm) {
        
        // 初始化：复制自己的数据块
        size_t selfOffset = rank * chunkSize;
        memcpy(recvBuffers[rank].Data() + selfOffset,
               sendBuffers[rank].Data() + selfOffset,
               chunkSize);
        
        for (int step = 0; step < size - 1; step++) {
            // 计算发送和接收目标
            int sendRank = (rank - step + size) % size;
            int recvRank = (rank - step - 1 + size) % size;
            
            // 发送数据块
            size_t sendOffset = sendRank * chunkSize;
            comm->Send(recvBuffers[rank].Data() + sendOffset, chunkSize, sendRank);
            
            // 接收数据块
            size_t recvOffset = recvRank * chunkSize;
            comm->Recv(recvBuffers[rank].Data() + recvOffset, chunkSize, recvRank);
            
            // 归约操作
            Reduce(recvBuffers[rank].Data() + recvOffset,
                   recvBuffers[rank].Data() + sendOffset,
                   chunkSize, reduceOp_);
        }
    }
    
    void AllGather(
        std::vector<Buffer>& recvBuffers,
        size_t chunkSize, int rank, int size, HcclComm comm) {
        
        for (int step = 0; step < size - 1; step++) {
            int sendRank = (rank - step + size) % size;
            int recvRank = (rank - step - 1 + size) % size;
            
            // 发送当前拥有的数据块
            size_t sendOffset = sendRank * chunkSize;
            comm->Send(recvBuffers[rank].Data() + sendOffset, chunkSize, sendRank);
            
            // 接收数据块
            size_t recvOffset = recvRank * chunkSize;
            comm->Recv(recvBuffers[rank].Data() + recvOffset, chunkSize, recvRank);
        }
    }
    
    HcclReduceOp reduceOp_;
};
```

### 1.3 性能分析

```cpp
// Ring AllReduce 时间复杂度分析
// 总时间 = Reduce-Scatter时间 + AllGather时间
//        = 2 * [(n-1) * α + (n-1) * chunkSize * β]
//        = 2(n-1)(α + chunkSize * β)

// 其中：
// α = 节点间固定延迟（约1-2μs）
// β = 每字节传输时间（取决于带宽）
// chunkSize = totalSize / n

// 当 n=8, totalSize=1MB 时：
// chunkSize = 128KB
// 时间 ≈ 2*7*(α + 128KB*β)
```

---

## 2. Mesh 算法

### 2.1 算法原理

**Mesh** 算法将节点排列成二维网格：
- 每个节点与四个方向的邻居通信（上下左右）
- 分 X 方向和 Y 方向两轮完成
- 适用于 Server 内通信

**特点**：
- 通信步数：2√n 步
- 复杂度：O(√n)
- 优点：并行度高，适合大规模集群
- 缺点：需要完整的网格拓扑

### 2.2 AllGather Mesh 实现

```cpp
class MeshAllGather : public Algorithm {
public:
    void Execute(
        const std::vector<Buffer>& sendBuffers,
        std::vector<Buffer>& recvBuffers,
        const TaskInfo& taskInfo,
        HcclComm comm) override {
        
        int rank = comm->GetRank();
        int size = comm->GetSize();
        
        // 计算网格维度
        int dimX = static_cast<int>(std::sqrt(static_cast<float>(size)));
        int dimY = size / dimX;
        
        int x = rank % dimX;
        int y = rank / dimX;
        
        size_t totalSize = taskInfo.count * GetDataTypeSize(taskInfo.dataType);
        
        // 初始化：复制自己的数据
        memcpy(recvBuffers[rank].Data(), sendBuffers[rank].Data(), totalSize);
        
        // X方向通信
        for (int offset = 1; offset < dimX; offset++) {
            int targetX = (x + offset) % dimX;
            int targetRank = targetX + y * dimX;
            
            // 双向交换数据
            Exchange(recvBuffers[rank].Data(),
                     recvBuffers[targetRank].Data(),
                     totalSize, rank, targetRank, comm);
        }
        
        // Y方向通信
        for (int offset = 1; offset < dimY; offset++) {
            int targetY = (y + offset) % dimY;
            int targetRank = x + targetY * dimX;
            
            // 双向交换数据
            Exchange(recvBuffers[rank].Data(),
                     recvBuffers[targetRank].Data(),
                     totalSize, rank, targetRank, comm);
        }
    }
    
private:
    void Exchange(
        void* buf1, void* buf2, size_t size,
        int rank1, int rank2, HcclComm comm) {
        
        // 异步发送/接收
        comm->SendAsync(buf1, size, rank2);
        comm->RecvAsync(buf2, size, rank1);
        
        // 等待完成
        comm->WaitAll();
        
        // 合并数据（AllGather需要保留双方数据）
        std::vector<char> temp(size);
        MergeData(buf1, buf2, temp.data(), size);
        memcpy(buf1, temp.data(), size);
        memcpy(buf2, temp.data(), size);
    }
};
```

### 2.3 数据交换模式

```
Mesh 8节点网格拓扑（2x4）：

    X方向 →
  ┌─────┬─────┬─────┬─────┐
Y │  0  │  1  │  2  │  3  │
↑ ├─────┼─────┼─────┼─────┤
│ │  4  │  5  │  6  │  7  │
  └─────┴─────┴─────┴─────┘

X方向通信步骤（rank=0）：
  Step 1: 0 ↔ 1
  Step 2: 0 ↔ 2 (通过1中转)
  Step 3: 0 ↔ 3 (通过2中转)

Y方向通信步骤（rank=0）：
  Step 1: 0 ↔ 4
```

---

## 3. RHD 算法

### 3.1 算法原理

**RHD**（Recursive Halving-Doubling）递归二分倍增算法：
- 将节点分成两组，组内交换数据
- 每轮合并一半节点
- 适用于节点数为2的幂次的场景

**特点**：
- 通信步数：log2(n) 步
- 复杂度：O(log n)
- 优点：通信步数最少
- 缺点：非2幂次节点会引入额外通信量

### 3.2 AllReduce RHD 实现

```cpp
class RHDAllReduce : public Algorithm {
public:
    RHDAllReduce(HcclReduceOp op) : reduceOp_(op) {}
    
    void Execute(
        const std::vector<Buffer>& sendBuffers,
        std::vector<Buffer>& recvBuffers,
        const TaskInfo& taskInfo,
        HcclComm comm) override {
        
        int rank = comm->GetRank();
        int size = comm->GetSize();
        
        size_t totalSize = taskInfo.count * GetDataTypeSize(taskInfo.dataType);
        
        // 初始化：复制输入数据
        memcpy(recvBuffers[rank].Data(), sendBuffers[rank].Data(), totalSize);
        
        // RHD 主循环
        for (int step = 1; step < size; step *= 2) {
            int partner = rank ^ step;  // 异或操作找配对节点
            
            if (partner < size) {
                // 确定发送/接收方向
                bool isSender = (rank < partner);
                
                if (isSender) {
                    // 发送数据到partner
                    comm->Send(recvBuffers[rank].Data(), totalSize, partner);
                    
                    // 接收partner数据
                    Buffer tempBuffer(totalSize);
                    comm->Recv(tempBuffer.Data(), totalSize, partner);
                    
                    // 归约
                    Reduce(recvBuffers[rank].Data(), tempBuffer.Data(), totalSize, reduceOp_);
                } else {
                    // 接收数据
                    Buffer tempBuffer(totalSize);
                    comm->Recv(tempBuffer.Data(), totalSize, partner);
                    
                    // 发送数据
                    comm->Send(recvBuffers[rank].Data(), totalSize, partner);
                    
                    // 归约
                    Reduce(tempBuffer.Data(), recvBuffers[rank].Data(), totalSize, reduceOp_);
                    memcpy(recvBuffers[rank].Data(), tempBuffer.Data(), totalSize);
                }
            }
        }
        
        // Broadcast阶段（使用RHD广播）
        for (int step = size / 2; step > 0; step /= 2) {
            int partner = rank ^ step;
            
            if (partner < size && rank < partner) {
                comm->Send(recvBuffers[rank].Data(), totalSize, partner);
            } else if (partner < size) {
                comm->Recv(recvBuffers[rank].Data(), totalSize, partner);
            }
        }
    }
    
private:
    HcclReduceOp reduceOp_;
};
```

### 3.3 递归二分过程

```
RHD AllReduce 过程（n=8）：

初始状态：
  Rank:  0   1   2   3   4   5   6   7
  Data: [a] [b] [c] [d] [e] [f] [g] [h]

Step 1 (step=1):
  0↔1: [a+b] [a+b] [c]   [d]   [e]   [f]   [g]   [h]
  2↔3: [a+b] [a+b] [c+d] [c+d] [e]   [f]   [g]   [h]
  4↔5: [a+b] [a+b] [c+d] [c+d] [e+f] [e+f] [g]   [h]
  6↔7: [a+b] [a+b] [c+d] [c+d] [e+f] [e+f] [g+h] [g+h]

Step 2 (step=2):
  0↔2: [a+b+c+d] [a+b] [a+b+c+d] [c+d] [e+f] [e+f] [g+h] [g+h]
  1↔3: [a+b+c+d] [a+b+c+d] [a+b+c+d] [a+b+c+d] [e+f] [e+f] [g+h] [g+h]
  4↔6: [a+b+c+d] [a+b+c+d] [a+b+c+d] [a+b+c+d] [e+f+g+h] [e+f] [e+f+g+h] [g+h]
  5↔7: [a+b+c+d] [a+b+c+d] [a+b+c+d] [a+b+c+d] [e+f+g+h] [e+f+g+h] [e+f+g+h] [e+f+g+h]

Step 3 (step=4):
  0↔4: [all] [all] [all] [all] [all] [all] [all] [all]
```

---

## 4. PairWise 算法

### 4.1 算法原理

**PairWise** 逐对通信算法：
- 每个节点与其他所有节点直接通信
- 适用于 AlltoAll 系列算子
- 避免网络中出现"一打多"现象

**特点**：
- 通信步数：n-1 步
- 复杂度：O(n)
- 优点：避免网络拥塞
- 缺点：通信步数多

### 4.2 AlltoAll PairWise 实现

```cpp
class PairWiseAlltoAll : public Algorithm {
public:
    void Execute(
        const Buffer& sendBuf,
        Buffer& recvBuf,
        const TaskInfo& taskInfo,
        HcclComm comm) override {
        
        int rank = comm->GetRank();
        int size = comm->GetSize();
        
        size_t blockSize = taskInfo.count * GetDataTypeSize(taskInfo.dataType);
        
        // 异步通信请求队列
        std::vector<CommRequest> sendReqs;
        std::vector<CommRequest> recvReqs;
        
        // 与每个其他rank交换数据
        for (int i = 0; i < size; i++) {
            if (i == rank) continue;
            
            // 计算发送和接收偏移
            size_t sendOffset = i * blockSize;  // 发送给i的块
            size_t recvOffset = i * blockSize;  // 从i接收的块
            
            // 异步发送
            sendReqs.push_back(
                comm->SendAsync(
                    sendBuf.Data() + sendOffset,
                    blockSize,
                    i
                )
            );
            
            // 异步接收
            recvReqs.push_back(
                comm->RecvAsync(
                    recvBuf.Data() + recvOffset,
                    blockSize,
                    i
                )
            );
        }
        
        // 等待所有通信完成
        comm->WaitAll(sendReqs);
        comm->WaitAll(recvReqs);
    }
};
```

---

## 5. Star 算法

### 5.1 算法原理

**Star** 星型拓扑算法：
- 根节点与所有其他节点直接通信
- 适用于有根操作（Broadcast、Reduce、Gather、Scatter）
- 单步完成通信

**特点**：
- 通信步数：1 步
- 复杂度：O(1)
- 优点：延迟最低
- 缺点：根节点带宽压力大

### 5.2 Broadcast Star 实现

```cpp
class StarBroadcast : public Algorithm {
public:
    void Execute(
        const Buffer& sendBuf,
        Buffer& recvBuf,
        const TaskInfo& taskInfo,
        int root,
        HcclComm comm) override {
        
        int rank = comm->GetRank();
        int size = comm->GetSize();
        
        size_t totalSize = taskInfo.count * GetDataTypeSize(taskInfo.dataType);
        
        if (rank == root) {
            // 根节点：发送数据到所有其他节点
            std::vector<CommRequest> reqs;
            
            for (int i = 0; i < size; i++) {
                if (i != root) {
                    reqs.push_back(
                        comm->SendAsync(sendBuf.Data(), totalSize, i)
                    );
                }
            }
            
            // 根节点自己复制数据
            memcpy(recvBuf.Data(), sendBuf.Data(), totalSize);
            
            // 等待所有发送完成
            comm->WaitAll(reqs);
        } else {
            // 非根节点：从根节点接收数据
            comm->Recv(recvBuf.Data(), totalSize, root);
        }
    }
};
```

---

## 6. 算法对比与选择

### 6.1 算法特性对比

| 算法 | 通信步数 | 适用场景 | 优点 | 缺点 |
|------|----------|----------|------|------|
| **Ring** | n-1 | Server内/间，中小规模 | 通信关系简单，抗拥塞 | 线性复杂度 |
| **Mesh** | 2√n | Server内，大规模 | 并行度高，带宽利用好 | 需要完整拓扑 |
| **RHD** | log2(n) | Server间，2幂次节点 | 步数最少，时延低 | 非2幂次有额外开销 |
| **PairWise** | n-1 | AlltoAll算子 | 避免一打多 | 步数多 |
| **Star** | 1 | 小规模，有根操作 | 延迟最低 | 根节点压力大 |

### 6.2 算法选择决策树

```
                    数据量 < 64KB?
                   /              \
                  YES              NO
                   │                │
              Star算法       节点数 <= 8?
                              /        \
                            YES        NO
                             │          │
                         Ring算法   Server内?
                                     /     \
                                   YES     NO
                                    │       │
                                Mesh算法  2幂次节点?
                                          /    \
                                        YES    NO
                                         │      │
                                     RHD算法  NHR/NB算法
```

### 6.3 性能模型

**Hockney 模型**：
```cpp
// 单步传输时间
double SingleStepTime(double alpha, double beta, size_t dataSize) {
    return alpha + beta * dataSize;
}

// Ring AllReduce 总时间
double RingAllReduceTime(int n, double alpha, double beta, size_t dataSize) {
    size_t chunkSize = dataSize / n;
    return 2 * (n - 1) * (alpha + beta * chunkSize);
}

// RHD AllReduce 总时间
double RHDAllReduceTime(int n, double alpha, double beta, size_t dataSize) {
    int steps = static_cast<int>(std::log2(n));
    return 2 * steps * (alpha + beta * dataSize);
}
```

---

## 7. 学习要点总结

### 7.1 核心算法实现要点

1. **Ring 算法**：环形拓扑，分 Reduce-Scatter 和 AllGather 两阶段
2. **Mesh 算法**：二维网格，分 X/Y 方向两轮通信
3. **RHD 算法**：递归二分，使用异或操作找配对节点
4. **PairWise 算法**：逐对通信，适用于 AlltoAll
5. **Star 算法**：星型拓扑，单步完成

### 7.2 算法选择策略

| 数据量 | 节点规模 | 推荐算法 |
|--------|----------|----------|
| < 64KB | 任意 | Star |
| 64KB-1MB | <= 8 | Ring |
| 1MB-10MB | > 8 | Mesh |
| > 10MB | 2幂次 | RHD |
| > 10MB | 非2幂次 | NHR/NB |

### 7.3 下周学习计划

- **Week4**：深入学习通信框架与平台层
- 重点：通信域管理、资源分配、算法选择机制