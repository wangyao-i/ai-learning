# Week4 - 通信框架与平台

## 1. 通信框架架构

### 1.1 整体架构

HCCL 的通信框架分为三个层次：

```
┌─────────────────────────────────────────────────────────────┐
│                   应用层（AI框架）                          │
│              PyTorch / MindSpore / TensorFlow              │
└───────────────────────────┬─────────────────────────────────┘
                            │ HCCL API
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   通信框架层                                │
│  ┌─────────────┬─────────────┬─────────────┐              │
│  │ 算子接口    │ 算法选择器   │ 通信域管理   │              │
│  │ Interface  │ Selector    │ Domain      │              │
│  └─────────────┴─────────────┴─────────────┘              │
└───────────────────────────┬─────────────────────────────────┘
                            │ 算法调用
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   通信算法层                                │
│  ┌─────────────┬─────────────┬─────────────┐              │
│  │ Ring       │ Mesh        │ RHD         │              │
│  │ PairWise   │ Star        │ NHR/NB      │              │
│  └─────────────┴─────────────┴─────────────┘              │
└───────────────────────────┬─────────────────────────────────┘
                            │ 平台调用
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   通信平台层                                │
│              HCOMM 通信基础库                               │
│  ┌─────────────┬─────────────┬─────────────┐              │
│  │ 控制面      │ 数据面      │ 资源管理    │              │
│  │ Control    │ Data        │ Resource    │              │
│  └─────────────┴─────────────┴─────────────┘              │
└───────────────────────────┬─────────────────────────────────┘
                            │ 硬件调用
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   硬件层                                    │
│              昇腾 AI 处理器（NPU）                          │
│              HCCS / RoCE / PCIe 链路                       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件

| 组件 | 职责 | 关键功能 |
|------|------|----------|
| **算子接口层** | 对外提供统一的通信API | AllReduce、AllGather等接口 |
| **算法选择器** | 根据条件选择最优算法 | 数据量、拓扑、节点数 |
| **通信域管理** | 管理通信上下文和资源 | 通信域创建、销毁、资源分配 |
| **通信算法层** | 实现具体的通信算法 | Ring、Mesh、RHD等 |
| **通信平台层** | 底层通信能力支撑 | 控制面、数据面分离 |

---

## 2. 通信域管理

### 2.1 通信域结构

```cpp
class HcclCommImpl : public HcclComm {
public:
    // 通信域状态
    enum class State {
        UNINITIALIZED,
        INITIALIZING,
        READY,
        DESTROYED
    };
    
    // 通信域信息
    struct DomainInfo {
        int rank;                      // 当前rank
        int size;                      // 通信域大小
        std::string topology;          // 拓扑类型
        std::vector<int> deviceList;   // 设备列表
    };
    
    // 资源信息
    struct ResourceInfo {
        size_t cclBufferSize;          // CCL Buffer大小
        void* cclInAddr;               // CCL_IN地址
        void* cclOutAddr;              // CCL_OUT地址
        std::vector<void*> notifyRegs; // Notify寄存器
    };
    
    // 构造函数
    HcclCommImpl() : state_(State::UNINITIALIZED) {}
    
    // 初始化
    HcclResult Init(const DomainInfo& domainInfo);
    
    // 销毁
    HcclResult Destroy();
    
    // 获取通信域信息
    HcclResult GetDomainInfo(DomainInfo& info);
    
    // 获取资源信息
    HcclResult GetResourceInfo(ResourceInfo& info);
    
private:
    State state_;                      // 当前状态
    DomainInfo domainInfo_;            // 通信域信息
    ResourceInfo resourceInfo_;        // 资源信息
    std::mutex mutex_;                 // 互斥锁
};
```

### 2.2 通信域初始化流程

```cpp
HcclResult HcclCommImpl::Init(const DomainInfo& domainInfo) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (state_ != State::UNINITIALIZED) {
        return HCCL_E_INVALID_ARG;
    }
    
    state_ = State::INITIALIZING;
    
    try {
        // 1. 保存通信域信息
        domainInfo_ = domainInfo;
        
        // 2. 分配CCL Buffer
        resourceInfo_.cclBufferSize = GetEnvBufferSize();
        resourceInfo_.cclInAddr = AllocDeviceMem(resourceInfo_.cclBufferSize);
        resourceInfo_.cclOutAddr = AllocDeviceMem(resourceInfo_.cclBufferSize);
        
        // 3. 初始化Notify寄存器
        resourceInfo_.notifyRegs.resize(domainInfo.size);
        for (int i = 0; i < domainInfo.size; i++) {
            resourceInfo_.notifyRegs[i] = AllocNotifyReg();
        }
        
        // 4. 建立设备间通信通道
        EstablishCommunicationChannels(domainInfo_);
        
        // 5. 同步设备状态
        SyncDeviceStates();
        
        state_ = State::READY;
        
    } catch (const std::exception& e) {
        state_ = State::UNINITIALIZED;
        return HCCL_E_INTERNAL;
    }
    
    return HCCL_SUCCESS;
}
```

### 2.3 CCL Buffer 管理

```cpp
// CCL Buffer 分配策略
size_t GetEnvBufferSize() {
    // 优先从环境变量获取
    const char* envSize = std::getenv("HCCL_BUFFSIZE");
    if (envSize != nullptr) {
        return std::stoull(envSize);
    }
    
    // 默认200MB
    return 200 * 1024 * 1024;
}

// CCL Buffer 使用模式
// CCL_IN: 接收远端数据
// CCL_OUT: 发送本地数据
// 双Buffer设计避免数据冲突
```

---

## 3. 算法选择机制

### 3.1 选择器架构

```cpp
class AlgorithmSelector {
public:
    // 选择算法
    static std::shared_ptr<Algorithm> Select(
        const TaskInfo& taskInfo,
        const TopoInfo& topoInfo,
        const ResourceInfo& resInfo);
    
private:
    // 根据数据量选择
    static AlgorithmType SelectByDataSize(size_t dataSize);
    
    // 根据拓扑选择
    static AlgorithmType SelectByTopology(const TopoInfo& topoInfo);
    
    // 根据节点数选择
    static AlgorithmType SelectByNodeCount(int nodeCount);
    
    // 综合评估
    static std::shared_ptr<Algorithm> Evaluate(
        const std::vector<AlgorithmType>& candidates,
        const TaskInfo& taskInfo);
};
```

### 3.2 选择策略实现

```cpp
std::shared_ptr<Algorithm> AlgorithmSelector::Select(
    const TaskInfo& taskInfo,
    const TopoInfo& topoInfo,
    const ResourceInfo& resInfo) {
    
    size_t dataSize = taskInfo.count * GetDataTypeSize(taskInfo.dataType);
    int nodeCount = topoInfo.rankCount;
    
    // 1. 基于数据量的初步筛选
    AlgorithmType candidate1 = SelectByDataSize(dataSize);
    
    // 2. 基于拓扑的筛选
    AlgorithmType candidate2 = SelectByTopology(topoInfo);
    
    // 3. 基于节点数的筛选
    AlgorithmType candidate3 = SelectByNodeCount(nodeCount);
    
    // 4. 综合评估
    std::vector<AlgorithmType> candidates = {candidate1, candidate2, candidate3};
    return Evaluate(candidates, taskInfo);
}

AlgorithmType AlgorithmSelector::SelectByDataSize(size_t dataSize) {
    if (dataSize < 64 * 1024) {           // < 64KB
        return AlgorithmType::STAR;
    } else if (dataSize < 1 * 1024 * 1024) { // < 1MB
        return AlgorithmType::RING;
    } else {                              // >= 1MB
        return AlgorithmType::MESH;
    }
}

AlgorithmType AlgorithmSelector::SelectByNodeCount(int nodeCount) {
    if (nodeCount <= 8) {
        return AlgorithmType::RING;
    } else if (IsPowerOfTwo(nodeCount)) {
        return AlgorithmType::RHD;
    } else {
        return AlgorithmType::NHR;
    }
}
```

### 3.3 性能预估模型

```cpp
// 算法性能预估
class PerformanceEstimator {
public:
    // 预估执行时间
    static double EstimateTime(
        AlgorithmType type,
        const TaskInfo& taskInfo,
        const TopoInfo& topoInfo);
    
private:
    // 获取算法特性
    static AlgorithmProfile GetProfile(AlgorithmType type);
    
    // Hockney 模型计算
    static double HockneyModel(
        double alpha,      // 固定延迟
        double beta,       // 带宽倒数
        double gamma,      // 计算延迟
        int steps,         // 通信步数
        size_t dataSize    // 数据量
    );
};
```

---

## 4. 任务调度机制

### 4.1 任务结构

```cpp
class CommunicationTask {
public:
    enum class Type {
        ALL_REDUCE,
        ALL_GATHER,
        BROADCAST,
        REDUCE_SCATTER,
        ALL_TO_ALL
    };
    
    struct Config {
        Type type;                      // 任务类型
        void* sendBuf;                  // 发送缓冲区
        void* recvBuf;                  // 接收缓冲区
        uint64_t count;                 // 数据个数
        HcclDataType dataType;          // 数据类型
        HcclReduceOp reduceOp;          // 归约操作（仅AllReduce/ReduceScatter）
        int root;                       // 根节点（仅Broadcast/Reduce/Gather/Scatter）
    };
    
    // 执行任务
    HcclResult Execute(const Config& config, HcclComm comm);
    
private:
    // 选择算法
    std::shared_ptr<Algorithm> SelectAlgorithm(
        const Config& config,
        HcclComm comm);
    
    // 分配临时资源
    HcclResult AllocResources(HcclComm comm);
    
    // 释放临时资源
    void FreeResources();
};
```

### 4.2 执行流程

```cpp
HcclResult CommunicationTask::Execute(const Config& config, HcclComm comm) {
    // 1. 校验参数
    if (!ValidateConfig(config)) {
        return HCCL_E_INVALID_ARG;
    }
    
    // 2. 获取通信域信息
    HcclCommImpl* commImpl = static_cast<HcclCommImpl*>(comm);
    HcclCommImpl::DomainInfo domainInfo;
    commImpl->GetDomainInfo(domainInfo);
    
    // 3. 选择算法
    std::shared_ptr<Algorithm> algorithm = SelectAlgorithm(config, comm);
    
    // 4. 分配临时资源
    if (AllocResources(comm) != HCCL_SUCCESS) {
        return HCCL_E_MEMORY;
    }
    
    // 5. 构建输入输出缓冲区
    std::vector<Buffer> sendBuffers(domainInfo.size);
    std::vector<Buffer> recvBuffers(domainInfo.size);
    
    // 6. 执行算法
    try {
        algorithm->Execute(sendBuffers, recvBuffers, config, comm);
    } catch (const std::exception& e) {
        FreeResources();
        return HCCL_E_INTERNAL;
    }
    
    // 7. 释放资源
    FreeResources();
    
    return HCCL_SUCCESS;
}
```

---

## 5. 流与同步机制

### 5.1 Stream 管理

```cpp
class StreamManager {
public:
    // 获取或创建主流
    aclrtStream GetMainStream(HcclComm comm);
    
    // 创建从流
    aclrtStream CreateSubStream(HcclComm comm);
    
    // 销毁从流
    void DestroySubStream(aclrtStream stream);
    
    // 同步主从流
    void SyncStreams(aclrtStream mainStream, aclrtStream subStream);
    
private:
    std::unordered_map<HcclComm, aclrtStream> mainStreams_;
    std::unordered_map<HcclComm, std::vector<aclrtStream>> subStreams_;
};
```

### 5.2 Notify 同步机制

```cpp
// Notify 操作封装
class NotifyManager {
public:
    // 初始化Notify寄存器
    HcclResult InitNotify(int rankCount);
    
    // Post操作：设置notify
    void Post(int targetRank);
    
    // Wait操作：等待notify
    void Wait(int sourceRank);
    
    // 重置notify
    void Reset(int rank);
    
private:
    std::vector<void*> notifyRegs_;  // Notify寄存器地址
};

// Rank内主从流同步示例
void SyncMainAndSubStream(aclrtStream mainStream, aclrtStream subStream) {
    // 主流通知从流开始
    NotifyManager::Instance()->Post(/*subStreamNotifyId*/);
    
    // 从流执行任务
    ExecuteOnSubStream(subStream);
    
    // 从流通知主流完成
    NotifyManager::Instance()->Post(/*mainStreamNotifyId*/);
    
    // 主流等待从流完成
    NotifyManager::Instance()->Wait(/*mainStreamNotifyId*/);
}
```

---

## 6. 错误处理与日志

### 6.1 错误码体系

```cpp
// 错误码定义
typedef enum {
    HCCL_SUCCESS = 0,                    // 成功
    HCCL_E_INTERNAL = 1,                 // 内部错误
    HCCL_E_INVALID_ARG = 2,              // 无效参数
    HCCL_E_MEMORY = 3,                   // 内存错误
    HCCL_E_NOT_INIT = 4,                 // 未初始化
    HCCL_E_TIMEOUT = 5,                  // 超时
    HCCL_E_DEVICE = 6,                   // 设备错误
    HCCL_E_COMM = 7,                     // 通信错误
    HCCL_E_STREAM = 8,                   // 流错误
} HcclResult;

// 错误信息获取
const char* HcclGetErrorString(HcclResult result) {
    switch (result) {
        case HCCL_SUCCESS: return "Success";
        case HCCL_E_INTERNAL: return "Internal error";
        case HCCL_E_INVALID_ARG: return "Invalid argument";
        case HCCL_E_MEMORY: return "Memory error";
        case HCCL_E_NOT_INIT: return "Not initialized";
        case HCCL_E_TIMEOUT: return "Timeout";
        case HCCL_E_DEVICE: return "Device error";
        case HCCL_E_COMM: return "Communication error";
        case HCCL_E_STREAM: return "Stream error";
        default: return "Unknown error";
    }
}
```

### 6.2 日志系统

```cpp
// 日志级别
enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR,
    FATAL
};

// 日志宏定义
#define HCCL_LOG(level, fmt, ...) \
    Logger::Instance()->Log(LogLevel::level, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define HCCL_DEBUG(fmt, ...) HCCL_LOG(DEBUG, fmt, ##__VA_ARGS__)
#define HCCL_INFO(fmt, ...)  HCCL_LOG(INFO, fmt, ##__VA_ARGS__)
#define HCCL_WARN(fmt, ...)  HCCL_LOG(WARN, fmt, ##__VA_ARGS__)
#define HCCL_ERROR(fmt, ...) HCCL_LOG(ERROR, fmt, ##__VA_ARGS__)
#define HCCL_FATAL(fmt, ...) HCCL_LOG(FATAL, fmt, ##__VA_ARGS__)

// 日志记录示例
HcclResult HcclCommImpl::Init(const DomainInfo& domainInfo) {
    HCCL_DEBUG("Initializing comm with rank=%d, size=%d", 
               domainInfo.rank, domainInfo.size);
    
    try {
        // 初始化逻辑
        HCCL_INFO("Comm initialized successfully");
    } catch (const std::exception& e) {
        HCCL_ERROR("Comm initialization failed: %s", e.what());
        return HCCL_E_INTERNAL;
    }
    
    return HCCL_SUCCESS;
}
```

---

## 7. 学习要点总结

### 7.1 框架核心组件

| 组件 | 职责 | 关键技术 |
|------|------|----------|
| **通信域管理** | 管理通信上下文和资源 | CCL Buffer、Notify寄存器 |
| **算法选择器** | 选择最优算法 | 数据量、拓扑、节点数 |
| **任务调度** | 编排和执行通信任务 | Stream管理、同步机制 |
| **错误处理** | 错误检测和报告 | 错误码体系、日志系统 |

### 7.2 关键设计模式

1. **策略模式**：算法选择器根据条件选择不同算法
2. **工厂模式**：根据类型创建不同的算子实例
3. **单例模式**：日志管理器、Notify管理器
4. **状态模式**：通信域的状态管理

### 7.3 性能优化要点

1. **异步通信**：使用SendAsync/RecvAsync提高并行度
2. **流水线**：主从流并发执行
3. **资源复用**：CCL Buffer在通信域内复用
4. **动态选择**：根据运行时条件选择最优算法

### 7.4 下周学习计划

- **Week5**：深入学习 HCOMM 通信基础库
- 重点：控制面与数据面分离设计、底层通信接口