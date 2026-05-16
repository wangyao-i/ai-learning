# Week5 - HCOMM 通信基础库

## 1. HCOMM 架构设计

### 1.1 整体架构

HCOMM（Huawei Communication）是 HCCL 的底层通信基础库，采用分层解耦的设计思路：

```
┌─────────────────────────────────────────────────────────────┐
│                     HCOMM 架构                              │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   控制面（Control Plane）              │  │
│  │  ┌─────────────┬─────────────┬─────────────────────┐  │  │
│  │  │ 通信域管理   │ 拓扑发现    │ 资源管理            │  │  │
│  │  │ Domain Mgmt │ Topo Discovery │ Resource Mgmt    │  │  │
│  │  └─────────────┴─────────────┴─────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                               │
│                            ▼                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   数据面（Data Plane）                 │  │
│  │  ┌─────────────┬─────────────┬─────────────────────┐  │  │
│  │  │ 点对点通信   │ 集合通信    │ 链路管理            │  │  │
│  │  │ Point-to-Point │ Collective │ Link Mgmt        │  │  │
│  │  └─────────────┴─────────────┴─────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                               │
│                            ▼                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   传输层（Transport）                  │  │
│  │  ┌─────────────┬─────────────┬─────────────────────┐  │  │
│  │  │ HCCS        │ RoCE        │ PCIe               │  │  │
│  │  │ 芯片间通信   │ 网络通信    │ 板内通信            │  │  │
│  │  └─────────────┴─────────────┴─────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 设计理念

| 设计原则 | 说明 | 实现方式 |
|----------|------|----------|
| **分层解耦** | 控制面与数据面分离 | 独立模块，清晰接口 |
| **可扩展性** | 支持多种传输链路 | 传输层抽象接口 |
| **高性能** | 零拷贝、异步通信 | 直接内存访问 |
| **可靠性** | 故障检测与恢复 | 心跳机制、重传 |

---

## 2. 控制面设计

### 2.1 通信域管理

```cpp
// 通信域结构
class HcommDomain {
public:
    // 通信域状态
    enum class State {
        UNINITIALIZED,
        INITIALIZING,
        READY,
        DESTROYED
    };
    
    // 节点信息
    struct NodeInfo {
        std::string ip;               // 节点IP
        int port;                     // 端口号
        int deviceId;                 // 设备ID
        std::string deviceType;       // 设备类型
    };
    
    // 初始化通信域
    HcommResult Init(const std::vector<NodeInfo>& nodes, int rank);
    
    // 销毁通信域
    HcommResult Destroy();
    
    // 获取通信域信息
    HcommResult GetInfo(HcommDomainInfo& info);
    
private:
    State state_;                     // 当前状态
    int rank_;                        // 当前rank
    int size_;                        // 通信域大小
    std::vector<NodeInfo> nodes_;     // 节点列表
    std::mutex mutex_;                // 互斥锁
};
```

### 2.2 拓扑发现

```cpp
class TopologyDiscoverer {
public:
    // 发现集群拓扑
    HcommResult Discover(std::vector<TopoNode>& nodes);
    
    // 获取拓扑信息
    HcommResult GetTopology(Topology& topology);
    
    // 计算最优路径
    HcommResult CalculateOptimalPath(
        int srcRank, int dstRank, 
        std::vector<int>& path);
    
private:
    // 探测节点间链路
    HcommResult ProbeLinks(std::vector<TopoNode>& nodes);
    
    // 构建拓扑图
    void BuildTopologyGraph(const std::vector<LinkInfo>& links);
    
    // 最短路径算法
    std::vector<int> ShortestPath(int src, int dst);
};
```

### 2.3 资源管理

```cpp
class ResourceManager {
public:
    // 资源类型
    enum class ResourceType {
        CCL_BUFFER,
        NOTIFY_REGISTER,
        STREAM,
        MEMORY
    };
    
    // 分配资源
    HcommResult Alloc(
        ResourceType type, 
        size_t size, 
        void** handle);
    
    // 释放资源
    HcommResult Free(ResourceType type, void* handle);
    
    // 查询资源状态
    HcommResult Query(ResourceType type, ResourceStatus& status);
    
private:
    std::unordered_map<void*, ResourceDesc> resources_;
    std::mutex mutex_;
};
```

---

## 3. 数据面设计

### 3.1 点对点通信

```cpp
class PointToPoint {
public:
    // 发送模式
    enum class SendMode {
        SYNC,                          // 同步发送
        ASYNC,                         // 异步发送
        BIDIRECT                      // 双向发送
    };
    
    // 同步发送
    HcommResult Send(
        void* buf, 
        size_t size, 
        int dstRank);
    
    // 异步发送
    HcommResult SendAsync(
        void* buf, 
        size_t size, 
        int dstRank, 
        HcommRequest* request);
    
    // 同步接收
    HcommResult Recv(
        void* buf, 
        size_t size, 
        int srcRank);
    
    // 异步接收
    HcommResult RecvAsync(
        void* buf, 
        size_t size, 
        int srcRank, 
        HcommRequest* request);
    
    // 双向通信（同时发送和接收）
    HcommResult SendRecv(
        void* sendBuf, size_t sendSize, int dstRank,
        void* recvBuf, size_t recvSize, int srcRank);
};
```

### 3.2 集合通信

```cpp
class Collective {
public:
    // 集合通信操作
    enum class OpType {
        ALL_REDUCE,
        ALL_GATHER,
        BROADCAST,
        REDUCE_SCATTER,
        ALL_TO_ALL
    };
    
    // 归约操作
    enum class ReduceOp {
        SUM,
        PROD,
        MAX,
        MIN
    };
    
    // 执行集合通信
    HcommResult Execute(
        OpType opType,
        const std::vector<void*>& sendBufs,
        std::vector<void*>& recvBufs,
        size_t count,
        HcommDataType dataType,
        ReduceOp reduceOp = ReduceOp::SUM,
        int root = 0);
    
private:
    // 选择集合通信算法
    std::shared_ptr<CollectiveAlgorithm> SelectAlgorithm(
        OpType opType,
        int size,
        size_t dataSize);
};
```

### 3.3 链路管理

```cpp
class LinkManager {
public:
    // 链路类型
    enum class LinkType {
        HCCS,                          // 芯片间通信
        ROCE,                          // 网络通信
        PCIE                           // 板内通信
    };
    
    // 链路状态
    enum class LinkStatus {
        UP,
        DOWN,
        DEGRADED
    };
    
    // 获取可用链路
    HcommResult GetAvailableLinks(
        int srcRank, 
        int dstRank,
        std::vector<LinkInfo>& links);
    
    // 选择最优链路
    HcommResult SelectOptimalLink(
        int srcRank, 
        int dstRank,
        LinkInfo& link);
    
    // 监控链路状态
    HcommResult MonitorLinks(std::vector<LinkStatus>& status);
    
private:
    // 链路性能评估
    double EvaluateLink(const LinkInfo& link);
};
```

---

## 4. 传输层实现

### 4.1 HCCS 传输

```cpp
class HccsTransport : public Transport {
public:
    // 初始化HCCS传输
    HcommResult Init(int deviceId);
    
    // 发送数据
    HcommResult Send(
        void* buf, 
        size_t size, 
        int dstDeviceId);
    
    // 接收数据
    HcommResult Recv(
        void* buf, 
        size_t size, 
        int srcDeviceId);
    
private:
    int deviceId_;                     // 设备ID
    void* hccsHandle_;                 // HCCS句柄
};
```

### 4.2 RoCE 传输

```cpp
class RoceTransport : public Transport {
public:
    // 初始化RoCE传输
    HcommResult Init(const std::string& ip, int port);
    
    // 发送数据
    HcommResult Send(
        void* buf, 
        size_t size, 
        const std::string& dstIp, 
        int dstPort);
    
    // 接收数据
    HcommResult Recv(
        void* buf, 
        size_t size, 
        const std::string& srcIp, 
        int srcPort);
    
private:
    std::string ip_;                   // 本地IP
    int port_;                         // 端口号
    void* roceHandle_;                 // RoCE句柄
};
```

### 4.3 PCIe 传输

```cpp
class PcieTransport : public Transport {
public:
    // 初始化PCIe传输
    HcommResult Init(int deviceId);
    
    // 发送数据
    HcommResult Send(
        void* buf, 
        size_t size, 
        int dstDeviceId);
    
    // 接收数据
    HcommResult Recv(
        void* buf, 
        size_t size, 
        int srcDeviceId);
    
private:
    int deviceId_;                     // 设备ID
    void* pcieHandle_;                 // PCIe句柄
};
```

---

## 5. 核心接口详解

### 5.1 初始化接口

```cpp
// 初始化HCOMM
HcommResult HcommInit(int* rank, int* size);

// 初始化通信域
HcommResult HcommCommInit(
    HcommComm* comm,
    int rank,
    int size,
    const HcommRootInfo* rootInfo);

// 销毁通信域
HcommResult HcommCommDestroy(HcommComm comm);
```

### 5.2 点对点通信接口

```cpp
// 同步发送
HcommResult HcommSend(
    void* buf,
    size_t size,
    int dstRank,
    HcommComm comm);

// 异步发送
HcommResult HcommSendAsync(
    void* buf,
    size_t size,
    int dstRank,
    HcommComm comm,
    HcommRequest* request);

// 同步接收
HcommResult HcommRecv(
    void* buf,
    size_t size,
    int srcRank,
    HcommComm comm);

// 异步接收
HcommResult HcommRecvAsync(
    void* buf,
    size_t size,
    int srcRank,
    HcommComm comm,
    HcommRequest* request);

// 等待请求完成
HcommResult HcommWait(HcommRequest* request);
```

### 5.3 集合通信接口

```cpp
// AllReduce
HcommResult HcommAllReduce(
    void* sendBuf,
    void* recvBuf,
    size_t count,
    HcommDataType dataType,
    HcommReduceOp reduceOp,
    HcommComm comm);

// AllGather
HcommResult HcommAllGather(
    void* sendBuf,
    void* recvBuf,
    size_t sendCount,
    HcommDataType dataType,
    HcommComm comm);

// Broadcast
HcommResult HcommBroadcast(
    void* sendBuf,
    void* recvBuf,
    size_t count,
    HcommDataType dataType,
    int root,
    HcommComm comm);

// ReduceScatter
HcommResult HcommReduceScatter(
    void* sendBuf,
    void* recvBuf,
    size_t recvCount,
    HcommDataType dataType,
    HcommReduceOp reduceOp,
    HcommComm comm);

// AlltoAll
HcommResult HcommAlltoAll(
    void* sendBuf,
    void* recvBuf,
    size_t count,
    HcommDataType dataType,
    HcommComm comm);
```

---

## 6. 内存管理

### 6.1 内存分配策略

```cpp
class MemoryManager {
public:
    // 内存类型
    enum class MemType {
        HOST,                          // 主机内存
        DEVICE,                        // 设备内存
        PINNED,                        // 锁页内存
        SHARED                         // 共享内存
    };
    
    // 分配内存
    HcommResult Alloc(
        MemType type,
        size_t size,
        void** ptr);
    
    // 释放内存
    HcommResult Free(MemType type, void* ptr);
    
    // 内存拷贝
    HcommResult Copy(
        void* dst,
        void* src,
        size_t size,
        MemType dstType,
        MemType srcType);
    
    // 内存注册（用于DMA）
    HcommResult Register(void* hostPtr, size_t size);
    
private:
    std::unordered_map<void*, MemoryDesc> registeredMem_;
};
```

### 6.2 零拷贝机制

```cpp
// 零拷贝发送
HcommResult ZeroCopySend(
    void* deviceBuf,                  // 设备内存指针（无需拷贝到主机）
    size_t size,
    int dstRank,
    HcommComm comm) {
    
    // 直接使用设备内存地址进行发送
    // 避免Host-Device数据拷贝
    
    // 获取设备内存的物理地址
    void* physicalAddr = GetPhysicalAddress(deviceBuf);
    
    // 通过DMA直接发送
    return DmaSend(physicalAddr, size, dstRank);
}
```

---

## 7. 错误处理与监控

### 7.1 错误码体系

```cpp
typedef enum {
    HCOMM_SUCCESS = 0,                 // 成功
    HCOMM_E_INTERNAL = 1,              // 内部错误
    HCOMM_E_INVALID_ARG = 2,           // 无效参数
    HCOMM_E_MEMORY = 3,                // 内存错误
    HCOMM_E_COMM = 4,                  // 通信错误
    HCOMM_E_TIMEOUT = 5,               // 超时
    HCOMM_E_DEVICE = 6,                // 设备错误
    HCOMM_E_NOT_SUPPORTED = 7,         // 不支持的操作
} HcommResult;
```

### 7.2 监控接口

```cpp
class Monitor {
public:
    // 获取性能指标
    HcommResult GetMetrics(
        HcommComm comm,
        HcommMetrics& metrics);
    
    // 获取链路状态
    HcommResult GetLinkStatus(
        HcommComm comm,
        std::vector<LinkStatus>& status);
    
    // 设置监控回调
    HcommResult SetCallback(
        HcommComm comm,
        HcommCallback callback,
        void* userData);
};
```

---

## 8. 学习要点总结

### 8.1 HCOMM 核心组件

| 组件 | 职责 | 关键功能 |
|------|------|----------|
| **控制面** | 管理通信域和资源 | 拓扑发现、资源分配 |
| **数据面** | 执行实际通信操作 | 点对点、集合通信 |
| **传输层** | 底层传输实现 | HCCS、RoCE、PCIe |
| **内存管理** | 内存分配和优化 | 零拷贝、DMA |

### 8.2 分层设计优势

1. **解耦性**：控制面与数据面独立，便于维护和扩展
2. **灵活性**：支持多种传输链路，按需选择
3. **高性能**：零拷贝、异步通信、DMA优化
4. **可靠性**：完善的错误处理和监控机制

### 8.3 与 HCCL 的关系

```
HCCL（集合通信库）
    ↓ 调用
HCOMM（通信基础库）
    ↓ 调用
硬件层（NPU、网络）
```

HCCL 提供高层集合通信API，HCOMM 提供底层通信能力支撑。

### 8.4 下周学习计划

- **Week6**：性能优化与实践
- 重点：性能测试、调优策略、生产环境部署