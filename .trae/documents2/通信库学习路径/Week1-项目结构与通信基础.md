# Week1 - 项目结构与通信基础

## 1. 项目结构分析

### 1.1 目录结构

```
hccl/
├── src/                         # 核心源码目录
│   ├── common/                  # 通用模块
│   │   ├── types.h              # 数据类型定义
│   │   ├── logging.h            # 日志模块
│   │   └── utils.h              # 工具函数
│   ├── ops/                     # 算子实现
│   │   ├── aicpu/               # AICPU Kernel处理
│   │   ├── channel/             # 通道资源计算
│   │   ├── inc/                 # 头文件
│   │   ├── interface/           # 算子接口定义
│   │   ├── registry/            # 算法注册机制
│   │   ├── scatter/             # Scatter算子实现
│   │   └── topo/                # 拓扑信息处理
│   └── domain/                  # 通信域管理
│       └── collective_communication/
│           ├── algorithm/       # 通信算法实现
│           └── framework/       # 通信框架
├── include/                     # 对外头文件
│   └── hccl/hccl.h             # HCCL API声明
├── test/                        # 测试代码
│   ├── ut/                      # 单元测试
│   └── st/                      # 系统测试
├── docs/                        # 文档
├── examples/                    # 样例代码
└── build.sh                     # 编译脚本
```

### 1.2 核心目录职责

| 目录 | 职责 | 关键文件 |
|------|------|----------|
| `src/common/` | 通用类型和工具函数 | types.h, logging.h |
| `src/ops/` | 通信算子实现 | interface/, registry/, scatter/ |
| `src/domain/` | 通信域管理 | algorithm/, framework/ |
| `include/hccl/` | 对外API | hccl.h |

---

## 2. 集合通信概念

### 2.1 基本概念

**集合通信（Collective Communication）**：多个进程协同完成的数据通信操作。

**通信域（Communicator）**：一组参与通信的进程的集合，是通信操作的上下文。

**Rank**：通信域中的每个进程的唯一标识，从0开始编号。

**Root**：某些操作（如Broadcast、Reduce）需要指定的根进程。

### 2.2 主要通信原语

| 原语 | 功能描述 | 复杂度 |
|------|----------|--------|
| **AllReduce** | 所有进程数据归约，结果广播到所有进程 | O(log n) |
| **AllGather** | 收集所有进程数据，每个进程获得完整数据 | O(log n) |
| **Broadcast** | 从根进程广播数据到所有进程 | O(log n) |
| **Reduce** | 归约所有进程数据，结果存于根进程 | O(log n) |
| **ReduceScatter** | 归约后按块分散到各进程 | O(log n) |
| **AlltoAll** | 每个进程向所有进程发送不同数据 | O(n) |
| **Gather** | 收集所有进程数据到根进程 | O(log n) |
| **Scatter** | 从根进程分散数据到所有进程 | O(log n) |

---

## 3. HCCL API 详解

### 3.1 数据类型定义

```cpp
// 数据类型枚举
typedef enum {
    HCCL_DATA_TYPE_FLOAT32 = 0,      // 32位浮点
    HCCL_DATA_TYPE_FLOAT16 = 1,      // 16位浮点
    HCCL_DATA_TYPE_INT32 = 2,        // 32位整数
    HCCL_DATA_TYPE_INT8 = 3,         // 8位整数
    HCCL_DATA_TYPE_UINT8 = 4,        // 无符号8位整数
    HCCL_DATA_TYPE_BF16 = 5,         // BF16浮点
} HcclDataType;

// 归约操作枚举
typedef enum {
    HCCL_REDUCE_SUM = 0,             // 求和
    HCCL_REDUCE_PROD = 1,            // 求积
    HCCL_REDUCE_MAX = 2,             // 求最大值
    HCCL_REDUCE_MIN = 3,             // 求最小值
} HcclReduceOp;

// HCCL错误码
typedef enum {
    HCCL_SUCCESS = 0,                // 成功
    HCCL_E_INTERNAL = 1,             // 内部错误
    HCCL_E_INVALID_ARG = 2,          // 无效参数
    HCCL_E_MEMORY = 3,               // 内存错误
    HCCL_E_NOT_INIT = 4,             // 未初始化
    HCCL_E_TIMEOUT = 5,              // 超时
} HcclResult;
```

### 3.2 通信域管理接口

```cpp
// 初始化通信域（单机多卡）
HcclResult HcclCommInitAll(
    HcclComm *comm,                  // 输出：通信域句柄
    int rankCount                    // 输入：进程数量
);

// 初始化通信域（自定义）
HcclResult HcclCommInit(
    HcclComm *comm,                  // 输出：通信域句柄
    int rank,                        // 输入：当前进程rank
    int rankCount,                   // 输入：进程总数
    const HcclRootInfo *rootInfo     // 输入：根进程信息
);

// 获取当前rank
HcclResult HcclGetRank(
    HcclComm comm,                   // 输入：通信域句柄
    int *rank                        // 输出：当前rank
);

// 获取通信域大小
HcclResult HcclGetSize(
    HcclComm comm,                   // 输入：通信域句柄
    int *size                        // 输出：通信域大小
);

// 销毁通信域
HcclResult HcclCommDestroy(
    HcclComm comm                    // 输入：通信域句柄
);
```

### 3.3 核心通信接口

```cpp
// AllReduce - 全归约
HcclResult HcclAllReduce(
    void *sendBuf,                   // 输入：发送缓冲区
    void *recvBuf,                   // 输出：接收缓冲区
    uint64_t count,                  // 输入：数据元素个数
    HcclDataType dataType,           // 输入：数据类型
    HcclReduceOp op,                 // 输入：归约操作类型
    HcclComm comm,                   // 输入：通信域
    aclrtStream stream               // 输入：执行流
);

// AllGather - 全收集
HcclResult HcclAllGather(
    void *sendBuf,                   // 输入：发送缓冲区
    void *recvBuf,                   // 输出：接收缓冲区
    uint64_t sendCount,              // 输入：发送数据个数
    HcclDataType dataType,           // 输入：数据类型
    HcclComm comm,                   // 输入：通信域
    aclrtStream stream               // 输入：执行流
);

// Broadcast - 广播
HcclResult HcclBroadcast(
    void *sendBuf,                   // 输入：发送缓冲区（根进程）
    void *recvBuf,                   // 输出：接收缓冲区（非根进程）
    uint64_t count,                  // 输入：数据个数
    HcclDataType dataType,           // 输入：数据类型
    int root,                        // 输入：根进程rank
    HcclComm comm,                   // 输入：通信域
    aclrtStream stream               // 输入：执行流
);

// ReduceScatter - 归约分散
HcclResult HcclReduceScatter(
    void *sendBuf,                   // 输入：发送缓冲区
    void *recvBuf,                   // 输出：接收缓冲区
    uint64_t recvCount,              // 输入：接收数据个数
    HcclDataType dataType,           // 输入：数据类型
    HcclReduceOp op,                 // 输入：归约操作类型
    HcclComm comm,                   // 输入：通信域
    aclrtStream stream               // 输入：执行流
);

// AlltoAll - 全交换
HcclResult HcclAlltoAll(
    void *sendBuf,                   // 输入：发送缓冲区
    void *recvBuf,                   // 输出：接收缓冲区
    uint64_t count,                  // 输入：每个进程发送的数据个数
    HcclDataType dataType,           // 输入：数据类型
    HcclComm comm,                   // 输入：通信域
    aclrtStream stream               // 输入：执行流
);
```

---

## 4. 通信域初始化流程

### 4.1 初始化流程

```
┌─────────────────────────────────────────────────────────────┐
│                    HcclCommInitAll                          │
├─────────────────────────────────────────────────────────────┤
│  1. 获取设备列表                                            │
│     └── aclrtGetDeviceCount()                              │
│                                                             │
│  2. 创建通信域对象                                          │
│     └── new HcclCommImpl()                                 │
│                                                             │
│  3. 初始化通信域                                            │
│     ├── 收集所有设备信息                                    │
│     ├── 建立设备间通信通道                                  │
│     └── 分配CCL Buffer                                     │
│                                                             │
│  4. 同步设备状态                                            │
│     └── 广播通信域配置到所有设备                            │
│                                                             │
│  5. 返回通信域句柄                                          │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 CCL Buffer 管理

CCL（Collective Communication Library）Buffer 是通信域内共享的内存缓冲区，用于跨 Rank 数据交换。

```cpp
class CclBuffer {
public:
    // CCL Buffer类型
    enum class Type {
        CCL_IN,                      // 输入缓冲区
        CCL_OUT                      // 输出缓冲区
    };
    
    // 分配Buffer
    HcclResult Alloc(size_t size);
    
    // 获取Buffer地址
    void* GetAddr(Type type);
    
    // 获取Buffer大小
    size_t GetSize();
    
    // 释放Buffer
    HcclResult Free();
    
private:
    void* cclInAddr_;                // CCL_IN地址
    void* cclOutAddr_;               // CCL_OUT地址
    size_t size_;                    // Buffer大小
};
```

**CCL Buffer 特性**：
- 默认大小：200MB（可通过环境变量 `HCCL_BUFFSIZE` 修改）
- 通信域内所有算子共享同一CCL Buffer
- 采用双Buffer设计（CCL_IN/CCL_OUT）避免数据冲突

---

## 5. 通信原语使用示例

### 5.1 AllReduce 使用示例

```cpp
#include "hccl/hccl.h"

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
    const int size = 1024;
    float* sendBuf = new float[size];
    float* recvBuf = new float[size];
    
    // 初始化数据（每个rank有不同数据）
    for (int i = 0; i < size; i++) {
        sendBuf[i] = static_cast<float>(rank * size + i);
    }
    
    // 创建执行流
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    // 执行AllReduce（求和）
    HcclResult result = HcclAllReduce(
        sendBuf,                      // 输入缓冲区
        recvBuf,                      // 输出缓冲区
        size,                         // 数据个数
        HCCL_DATA_TYPE_FLOAT32,       // 数据类型
        HCCL_REDUCE_SUM,              // 归约操作
        comm,                         // 通信域
        stream                        // 执行流
    );
    
    // 同步等待完成
    aclrtSynchronizeStream(stream);
    
    // 验证结果（所有rank的recvBuf应该相同）
    std::cout << "Rank " << rank << " result[0] = " << recvBuf[0] << std::endl;
    
    // 清理资源
    delete[] sendBuf;
    delete[] recvBuf;
    aclrtDestroyStream(stream);
    HcclCommDestroy(comm);
    
    return 0;
}
```

### 5.2 Broadcast 使用示例

```cpp
// 广播：从root=0广播数据到所有rank
HcclResult result = HcclBroadcast(
    sendBuf,                         // root进程的发送缓冲区
    recvBuf,                         // 非root进程的接收缓冲区
    size,                            // 数据个数
    HCCL_DATA_TYPE_FLOAT32,          // 数据类型
    0,                               // root rank
    comm,                            // 通信域
    stream                           // 执行流
);
```

---

## 6. 学习要点总结

### 6.1 重点概念

1. **通信域**：通信操作的上下文，管理通信资源
2. **Rank**：通信域内进程的唯一标识
3. **CCL Buffer**：跨Rank数据交换的共享内存
4. **Stream**：NPU上的执行流，承载Task序列

### 6.2 核心接口

| 接口 | 功能 | 关键参数 |
|------|------|----------|
| HcclCommInitAll | 初始化通信域 | rankCount |
| HcclAllReduce | 全归约 | sendBuf, recvBuf, count, op |
| HcclAllGather | 全收集 | sendBuf, recvBuf, sendCount |
| HcclBroadcast | 广播 | sendBuf, recvBuf, root |
| HcclReduceScatter | 归约分散 | sendBuf, recvBuf, recvCount |
| HcclAlltoAll | 全交换 | sendBuf, recvBuf, count |

### 6.3 学习建议

1. 理解集合通信的基本概念和原语
2. 熟悉 HCCL API 的使用方式
3. 阅读 `include/hccl/hccl.h` 头文件
4. 尝试编写简单的通信测试程序

---

## 7. 下周学习计划

- **Week2**：深入学习核心通信算子的实现原理
- 重点：AllReduce、AllGather 的算法实现