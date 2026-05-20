# HCCL 框架特性代码学习计划

## 1. 特性总览

HCCL（Huawei Collective Communication Library）是基于昇腾 AI 处理器的高性能集合通信库，为计算集群提供高性能、高可靠的通信方案。

### 1.1 核心特性

| 特性类别 | 特性名称 | 适用场景 | 复杂度 |
|---------|---------|----------|--------|
| **通信原语** | AllReduce | 梯度聚合 | O(log n) |
| | AllGather | 数据收集 | O(log n) |
| | Broadcast | 参数分发 | O(log n) |
| | ReduceScatter | 归约分散 | O(log n) |
| | AlltoAll | 全交换 | O(n) |
| **通信算法** | Ring | Server内/间 | O(n) |
| | Mesh | Server内 | O(√n) |
| | RHD | Server间 | O(log n) |
| | PairWise | AlltoAll算子 | O(n) |
| | Star | 有根操作 | O(1) |
| **性能优化** | 流水线并行 | 大数据量通信 | - |
| | 拓扑感知 | 自动选路 | - |
| | 动态算法选择 | 自适应优化 | - |
| | 内存优化 | Buffer管理 | - |

---

## 2. 学习路径

### 2.1 第一阶段：通信基础（1周）

**学习内容**：
- 集合通信概念与原语
- 通信域与 Rank 概念
- HCCL 项目结构分析
- 核心头文件阅读

**重点文件**：
- `include/hccl/hccl.h` - HCCL 对外接口
- `src/common/` - 通用类型定义
- `src/ops/interface/` - 算子接口

**学习目标**：
- 理解集合通信的基本概念
- 掌握通信域的管理机制
- 熟悉 HCCL 的整体架构

### 2.2 第二阶段：核心算子（2周）

**学习内容**：
- AllReduce 算子实现
- AllGather 算子实现
- Broadcast / ReduceScatter / AlltoAll
- 算子注册机制

**重点文件**：
- `src/ops/scatter/` - Scatter 算子
- `src/ops/registry/` - 算法注册
- `src/ops/interface/` - 算子接口

**学习目标**：
- 掌握各通信算子的实现原理
- 理解算子注册与调度机制

### 2.3 第三阶段：通信算法（2周）

**学习内容**：
- Ring 算法原理与实现
- Mesh 算法原理与实现
- RHD / PairWise / Star 算法
- 算法选择策略

**重点文件**：
- `src/domain/collective_communication/algorithm/`
- `docs/Ring.md` - Ring 算法文档
- `docs/Mesh.md` - Mesh 算法文档

**学习目标**：
- 理解各通信算法的原理
- 掌握算法选择的策略

### 2.4 第四阶段：框架与实践（1周）

**学习内容**：
- 通信框架架构
- HCOMM 基础库
- 性能测试与调优
- 生产环境部署

**重点文件**：
- `src/domain/collective_communication/framework/`
- `test/` - 测试代码
- `docs/` - 文档

**学习目标**：
- 理解通信框架的设计
- 掌握性能调优方法

---

## 3. 项目目录结构

```
hccl/
├── src/                         # HCCL算子源码目录
│   ├── common/                  # 通用逻辑
│   │   ├── types.h              # 类型定义
│   │   └── logging.h            # 日志模块
│   ├── ops/                     # HCCL算子实现
│   │   ├── aicpu/               # Aicpu Kernel流程
│   │   ├── channel/             # 通道资源计算
│   │   ├── inc/                 # 头文件
│   │   ├── interface/           # 算子接口
│   │   ├── registry/            # 算法注册
│   │   ├── scatter/             # Scatter算子
│   │   └── topo/                # 拓扑信息
│   └── domain/                  # 通信域
│       └── collective_communication/
│           ├── algorithm/       # 通信算法
│           └── framework/       # 通信框架
├── include/                     # 对外头文件
│   └── hccl/
│       └── hccl.h              # HCCL API
├── test/                        # 测试代码
│   ├── ut/                      # 单元测试
│   └── st/                      # 系统测试
├── docs/                        # 文档
├── examples/                    # 样例代码
└── build.sh                     # 编译脚本
```

---

## 4. C++ 代码解读重点

### 4.1 HCCL API 接口

```cpp
// AllReduce 接口
HcclResult HcclAllReduce(
    void *sendBuf,           // 发送缓冲区
    void *recvBuf,           // 接收缓冲区
    uint64_t count,          // 数据个数
    HcclDataType dataType,   // 数据类型
    HcclReduceOp op,         // 归约操作
    HcclComm comm,           // 通信域
    aclrtStream stream       // 执行流
);
```

### 4.2 通信域管理

```cpp
class HcclComm {
public:
    // 初始化通信域
    static HcclResult InitAll(HcclComm **comm, int rankCount);
    
    // 获取当前 rank
    HcclResult GetRank(int *rank);
    
    // 获取通信域大小
    HcclResult GetSize(int *size);
    
    // 销毁通信域
    HcclResult Destroy();
};
```

### 4.3 算法选择机制

```cpp
class AlgorithmSelector {
public:
    // 根据条件选择最优算法
    static std::shared_ptr<Algorithm> Select(
        const TaskInfo &taskInfo,     // 任务信息
        const TopoInfo &topoInfo,     // 拓扑信息
        const ResourceInfo &resInfo    // 资源信息
    );
};
```

### 4.4 Ring AllReduce 核心逻辑

```cpp
class RingAllReduce : public Algorithm {
public:
    void Execute(
        const std::vector<DeviceBuffer> &sendBufs,
        const std::vector<DeviceBuffer> &recvBufs,
        const TaskInfo &taskInfo,
        HcclComm comm
    ) override {
        int rank = comm->GetRank();
        int size = comm->GetSize();
        
        // Reduce-Scatter 阶段
        for (int step = 0; step < size - 1; step++) {
            int sendRank = (rank - step + size) % size;
            int recvRank = (rank - step - 1 + size) % size;
            
            // 发送数据到 sendRank
            Send(sendBufs[sendRank], sendRank);
            // 从 recvRank 接收数据
            Recv(recvBufs[recvRank], recvRank);
            // 归约
            Reduce(recvBufs[recvRank], sendBufs[sendRank]);
        }
        
        // AllGather 阶段
        for (int step = 0; step < size - 1; step++) {
            int sendRank = (rank - step + size) % size;
            int recvRank = (rank - step - 1 + size) % size;
            
            // 发送数据
            Send(sendBufs[sendRank], sendRank);
            // 接收数据
            Recv(recvBufs[recvRank], recvRank);
        }
    }
};
```

### 4.5 Mesh 算法核心逻辑

```cpp
class MeshAllGather : public Algorithm {
public:
    void Execute(
        const std::vector<DeviceBuffer> &sendBufs,
        const std::vector<DeviceBuffer> &recvBufs,
        const TaskInfo &taskInfo,
        HcclComm comm
    ) override {
        int rank = comm->GetRank();
        int size = comm->GetSize();
        
        // 计算网格维度
        int dimX = std::sqrt(size);
        int dimY = size / dimX;
        
        int x = rank % dimX;
        int y = rank / dimX;
        
        // X 方向通信
        for (int i = 0; i < dimX; i++) {
            int targetRank = i + y * dimX;
            if (i != x) {
                Exchange(sendBufs[targetRank], recvBufs[targetRank], targetRank);
            }
        }
        
        // Y 方向通信
        for (int j = 0; j < dimY; j++) {
            int targetRank = x + j * dimX;
            if (j != y) {
                Exchange(sendBufs[targetRank], recvBufs[targetRank], targetRank);
            }
        }
    }
};
```

---

## 5. 学习产出

### 5.1 代码分析报告

- **通信算子分析**：详细分析各算子的实现原理
- **算法对比报告**：对比各通信算法的性能特点
- **框架架构分析**：分析通信框架的设计模式

### 5.2 性能测试报告

- **基准测试**：各算法的性能对比
- **参数调优**：不同配置下的性能表现
- **瓶颈分析**：性能瓶颈定位与优化

### 5.3 实践方案

- **部署指南**：HCCL 在集群中的部署方法
- **调试技巧**：通信问题的调试方法
- **优化建议**：性能优化的最佳实践

---

## 6. 学习计划时间线

| 周次 | 学习阶段 | 学习内容 | 产出 |
|------|----------|----------|------|
| 第 1 周 | 通信基础 | 集合通信概念、项目结构 | 基础概念笔记 |
| 第 2 周 | 核心算子 | AllReduce、AllGather | 算子分析报告 |
| 第 3 周 | 核心算子 | Broadcast、ReduceScatter、AlltoAll | 算子分析报告 |
| 第 4 周 | 通信算法 | Ring、Mesh 算法 | 算法分析报告 |
| 第 5 周 | 通信算法 | RHD、PairWise、Star | 算法分析报告 |
| 第 6 周 | 框架实践 | HCOMM、性能调优 | 部署方案文档 |

---

## 7. 总结

HCCL 是昇腾 AI 处理器的核心集合通信库，通过本学习计划，您将全面掌握其核心架构、通信算法和实现原理。学习过程中建议结合源码阅读和实践测试，深入理解分布式通信的技术细节。