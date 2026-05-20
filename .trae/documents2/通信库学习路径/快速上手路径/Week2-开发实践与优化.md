# Week2 - 开发实践与优化

## 1. 算法选择策略

### 1.1 自动选择

HCCL 会根据运行时条件自动选择最优算法：

```cpp
// HCCL自动选择，无需手动指定
HcclAllReduce(sendBuf, recvBuf, count, dataType, op, comm, stream);
```

**选择因素**：
- 数据量大小
- 节点数量
- 拓扑结构
- 网络状况

### 1.2 手动选择

通过环境变量或代码手动指定算法：

```bash
# 通过环境变量指定
export HCCL_ALGO_SELECT=ring
export HCCL_ALGO_SELECT=mesh
export HCCL_ALGO_SELECT=rhd
```

```cpp
// 通过代码指定（示例）
SetAlgorithm("mesh");
```

### 1.3 算法选择决策树

```
数据量 < 64KB？
  ├─ 是 → Star算法
  └─ 否 → 节点数 <= 8？
          ├─ 是 → Ring算法
          └─ 否 → Server内？
                  ├─ 是 → Mesh算法
                  └─ 否 → 2幂次节点？
                          ├─ 是 → RHD算法
                          └─ 否 → NHR/NB算法
```

### 1.4 算法性能对比

| 数据量 | 节点数 | Ring | Mesh | RHD | Star |
|--------|--------|------|------|-----|------|
| 64KB | 8 | 0.5ms | 0.8ms | 0.6ms | **0.2ms** |
| 1MB | 8 | 1.2ms | **0.9ms** | 1.0ms | 0.8ms |
| 10MB | 8 | 5.0ms | **3.2ms** | 3.5ms | - |

---

## 2. 性能调优方法

### 2.1 内存优化

#### CCL Buffer 配置

```bash
# 设置CCL Buffer大小（默认200MB）
export HCCL_BUFFSIZE=524288000  # 500MB
```

```cpp
// 根据数据量动态调整
size_t bufferSize = CalculateOptimalBufferSize(dataSize);
SetCCLBufferSize(bufferSize);
```

#### 内存复用

```cpp
// 复用缓冲区，减少分配开销
class BufferPool {
public:
    void* Alloc(size_t size) {
        // 从池中分配
    }
    
    void Free(void* ptr) {
        // 归还到池中
    }
};
```

#### 零拷贝通信

```cpp
// 直接使用设备内存，避免Host-Device拷贝
void* deviceBuf = AllocDeviceMem(dataSize);

// 直接通信
HcclAllReduce(deviceBuf, deviceBuf, count, dataType, op, comm, stream);
```

### 2.2 流水线优化

#### 启用流水线

```bash
# 设置流水线深度（默认4）
export HCCL_PIPELINE_DEPTH=8
```

```cpp
// 流水线并行执行
void ExecutePipelined() {
    for (int i = 0; i < pipelineDepth; i++) {
        // 并发执行多个通信任务
        HcclAllReduceAsync(...);
    }
    WaitAll();
}
```

#### 流管理

```cpp
// 使用多个流提高并行度
aclrtStream streams[4];
for (int i = 0; i < 4; i++) {
    aclrtCreateStream(&streams[i]);
}

// 在不同流上并发执行
HcclAllReduce(..., streams[0]);
HcclAllGather(..., streams[1]);
```

### 2.3 算法优化

#### 选择最优算法

```cpp
// 根据数据量选择算法
std::string SelectAlgorithm(size_t dataSize, int nodeCount) {
    if (dataSize < 64 * 1024) {
        return "star";
    } else if (nodeCount <= 8) {
        return "ring";
    } else if (IsServerIntra()) {
        return "mesh";
    } else if (IsPowerOfTwo(nodeCount)) {
        return "rhd";
    } else {
        return "nhr";
    }
}
```

#### 混合算法

```cpp
// Server内使用Mesh，Server间使用RHD
void HybridExecute() {
    // Server内通信
    MeshAllGather(...);
    
    // Server间通信
    RHDAllReduce(...);
}
```

---

## 3. 性能测试与分析

### 3.1 基准测试

```cpp
#include <chrono>

void BenchmarkAllReduce(size_t dataSize, int iterations) {
    std::vector<double> times;
    
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        HcclAllReduce(sendBuf, recvBuf, count, dataType, op, comm, stream);
        aclrtSynchronizeStream(stream);
        
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(end - start).count();
        times.push_back(time);
    }
    
    // 计算统计信息
    double avgTime = CalculateAverage(times);
    double minTime = *std::min_element(times.begin(), times.end());
    double maxTime = *std::max_element(times.begin(), times.end());
    
    std::cout << "Average: " << avgTime * 1000 << "ms" << std::endl;
    std::cout << "Min: " << minTime * 1000 << "ms" << std::endl;
    std::cout << "Max: " << maxTime * 1000 << "ms" << std::endl;
    
    // 计算带宽
    double bandwidth = dataSize / avgTime / 1e9;  // GB/s
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
}
```

### 3.2 性能分析工具

```cpp
// 使用性能分析工具
class Profiler {
public:
    void Start() {
        startTime_ = GetCurrentTime();
    }
    
    void Stop() {
        endTime_ = GetCurrentTime();
        duration_ = endTime_ - startTime_;
    }
    
    void Print() {
        std::cout << "Duration: " << duration_ * 1000 << "ms" << std::endl;
    }
    
private:
    double startTime_;
    double endTime_;
    double duration_;
};
```

### 3.3 性能瓶颈分析

| 瓶颈类型 | 表现 | 解决方法 |
|----------|------|----------|
| 网络带宽 | 带宽利用率低 | 使用Mesh/RHD算法 |
| 内存带宽 | 内存拷贝慢 | 启用零拷贝 |
| 算法选择 | 不适合当前场景 | 动态算法选择 |
| 同步开销 | 等待时间长 | 异步通信 |

---

## 4. 常见问题解决

### 4.1 通信超时

**问题**：通信操作超时

**原因**：
- 网络故障
- 节点崩溃
- 资源不足

**解决方法**：

```cpp
// 设置超时时间
SetTimeout(30000);  // 30秒

// 错误处理
HcclResult result = HcclAllReduce(...);
if (result == HCCL_E_TIMEOUT) {
    // 重试或降级处理
    RetryOrFallback();
}
```

### 4.2 内存不足

**问题**：内存分配失败

**原因**：
- CCL Buffer过大
- 设备内存不足

**解决方法**：

```bash
# 减小CCL Buffer大小
export HCCL_BUFFSIZE=104857600  # 100MB
```

```cpp
// 分批处理
void ProcessInBatches(size_t totalSize, size_t batchSize) {
    for (size_t offset = 0; offset < totalSize; offset += batchSize) {
        size_t currentSize = std::min(batchSize, totalSize - offset);
        HcclAllReduce(
            sendBuf + offset,
            recvBuf + offset,
            currentSize,
            dataType, op, comm, stream
        );
    }
}
```

### 4.3 性能下降

**问题**：通信性能低于预期

**原因**：
- 算法选择不当
- 网络拥塞
- 配置不合理

**解决方法**：

```cpp
// 诊断性能问题
void DiagnosePerformance() {
    // 1. 检查算法选择
    std::string algo = GetSelectedAlgorithm();
    std::cout << "Algorithm: " << algo << std::endl;
    
    // 2. 检查网络状态
    NetworkStatus status = CheckNetworkStatus();
    std::cout << "Network: " << status << std::endl;
    
    // 3. 检查内存使用
    size_t memUsage = GetMemoryUsage();
    std::cout << "Memory: " << memUsage << " MB" << std::endl;
    
    // 4. 给出优化建议
    GiveOptimizationSuggestions();
}
```

### 4.4 数据不一致

**问题**：不同rank的数据不一致

**原因**：
- 通信顺序错误
- 流同步问题
- 数据初始化错误

**解决方法**：

```cpp
// 确保所有rank执行相同操作
if (rank == 0) {
    InitData(data);
}

// 广播到所有rank
HcclBroadcast(data, size, dataType, 0, comm, stream);

// 同步等待
aclrtSynchronizeStream(stream);
```

---

## 5. 实战案例分析

### 5.1 案例1：分布式训练中的梯度聚合

**场景**：8卡分布式训练，每卡梯度1MB

**实现**：

```cpp
class DistributedTrainer {
public:
    void AggregateGradients(std::vector<float>& gradients) {
        // 1. AllReduce聚合梯度
        HcclAllReduce(
            gradients.data(),
            gradients.data(),
            gradients.size(),
            HCCL_DATA_TYPE_FLOAT32,
            HCCL_REDUCE_SUM,
            comm_,
            stream_
        );
        
        // 2. 同步等待
        aclrtSynchronizeStream(stream_);
        
        // 3. 平均梯度
        float scale = 1.0f / size_;
        ScaleGradients(gradients, scale);
    }
    
private:
    HcclComm comm_;
    aclrtStream stream_;
    int size_;
};
```

### 5.2 案例2：模型并行中的参数收集

**场景**：4卡模型并行，每卡持有部分参数

**实现**：

```cpp
class ModelParallel {
public:
    void CollectParameters(std::vector<float>& params) {
        // 1. AllGather收集所有参数
        HcclAllGather(
            localParams_.data(),
            params.data(),
            localParams_.size(),
            HCCL_DATA_TYPE_FLOAT32,
            comm_,
            stream_
        );
        
        // 2. 同步等待
        aclrtSynchronizeStream(stream_);
    }
    
private:
    std::vector<float> localParams_;
    HcclComm comm_;
    aclrtStream stream_;
};
```

### 5.3 案例3：混合精度训练

**场景**：使用FP16进行训练，FP32进行梯度聚合

**实现**：

```cpp
class MixedPrecisionTrainer {
public:
    void AggregateGradientsFP32(std::vector<half>& gradientsFP16) {
        // 1. 转换为FP32
        std::vector<float> gradientsFP32(gradientsFP16.size());
        CastFP16ToFP32(gradientsFP16, gradientsFP32);
        
        // 2. FP32 AllReduce
        HcclAllReduce(
            gradientsFP32.data(),
            gradientsFP32.data(),
            gradientsFP32.size(),
            HCCL_DATA_TYPE_FLOAT32,
            HCCL_REDUCE_SUM,
            comm_,
            stream_
        );
        
        // 3. 转换回FP16
        CastFP32ToFP16(gradientsFP32, gradientsFP16);
    }
};
```

---

## 6. 最佳实践

### 6.1 开发规范

```cpp
// 1. 错误处理
HcclResult result = HcclAllReduce(...);
if (result != HCCL_SUCCESS) {
    HandleError(result);
}

// 2. 资源管理
class ScopedComm {
public:
    ScopedComm(int deviceCount) {
        HcclCommInitAll(&comm_, deviceCount);
    }
    
    ~ScopedComm() {
        HcclCommDestroy(comm_);
    }
    
    HcclComm Get() { return comm_; }
    
private:
    HcclComm comm_;
};

// 3. 流同步
void ExecuteWithSync() {
    HcclAllReduce(...);
    aclrtSynchronizeStream(stream);
}
```

### 6.2 性能优化清单

| 优化项 | 说明 | 验证方法 |
|--------|------|----------|
| 算法选择 | 根据场景选择最优算法 | 性能测试 |
| 内存配置 | 合理配置CCL Buffer | 内存监控 |
| 流水线深度 | 根据数据量调整 | 带宽测试 |
| 异步通信 | 使用异步API | 延迟测试 |

### 6.3 调试技巧

```cpp
// 1. 启用调试日志
export HCCL_DEBUG_LEVEL=3

// 2. 打印rank信息
std::cout << "Rank " << rank << ": Starting..." << std::endl;

// 3. 验证数据
void VerifyData(void* data, size_t size, int rank) {
    for (size_t i = 0; i < size; i++) {
        if (data[i] != expectedValue) {
            std::cerr << "Data mismatch at rank " << rank << std::endl;
            break;
        }
    }
}

// 4. 性能分析
ProfileOperation([&]() {
    HcclAllReduce(...);
});
```

---

## 7. 学习总结

### 7.1 核心知识点

1. **算法选择**：根据数据量和节点数选择最优算法
2. **性能优化**：内存优化、流水线优化、算法优化
3. **问题解决**：超时、内存不足、性能下降、数据不一致
4. **最佳实践**：开发规范、性能优化清单、调试技巧

### 7.2 开发流程

```
需求分析 → 算法选择 → 代码实现 → 性能测试 → 优化调优 → 部署上线
```

### 7.3 后续学习

1. 深入学习源码实现
2. 参与实际项目开发
3. 研究通信算法优化
4. 学习其他通信库（NCCL、MPI）

---

## 8. 快速参考

### 8.1 常用API速查

```cpp
// 初始化
HcclCommInitAll(&comm, deviceCount);
HcclGetRank(comm, &rank);
HcclGetSize(comm, &size);

// 通信算子
HcclAllReduce(sendBuf, recvBuf, count, dataType, op, comm, stream);
HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream);
HcclBroadcast(sendBuf, recvBuf, count, dataType, root, comm, stream);
HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);

// 清理
HcclCommDestroy(comm);
```

### 8.2 环境变量速查

| 变量名 | 作用 | 默认值 |
|--------|------|--------|
| HCCL_BUFFSIZE | CCL Buffer大小 | 200MB |
| HCCL_ALGO_SELECT | 算法选择 | auto |
| HCCL_PIPELINE_DEPTH | 流水线深度 | 4 |
| HCCL_DEBUG_LEVEL | 调试级别 | 0 |

### 8.3 常见错误码

| 错误码 | 含义 | 解决方法 |
|--------|------|----------|
| HCCL_SUCCESS | 成功 | - |
| HCCL_E_INVALID_ARG | 无效参数 | 检查参数 |
| HCCL_E_MEMORY | 内存错误 | 增加内存 |
| HCCL_E_TIMEOUT | 超时 | 检查网络 |

---

**恭喜完成 HCCL 快速上手路径！** 🎉

您现在已经掌握了 HCCL 的核心概念和开发方法，可以开始在昇腾 NPU 集群中进行分布式通信开发了！