# Week 9: 推理服务化与部署

## 本周目标
- 深入理解推理服务化的架构设计
- 掌握vLLM Serving的实现原理
- 理解OpenAI API兼容接口
- 能够部署和优化推理服务

---

## Day 1: 推理服务化架构总览

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 分析vLLM服务架构 | 了解服务入口 |
| 晚上 1.5h | 服务化架构设计 | 能画出架构图 |

### 学习材料

#### 源码位置
- `vllm/entrypoints/openai/api_server.py` - OpenAI API服务
- `vllm/engine/async_llm_engine.py` - 异步引擎
- `vllm/api_server.py` - 基础API服务

#### 服务化架构图
```
┌─────────────────────────────────────────────────────────────┐
│                    推理服务化架构                           │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Client Layer                       │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │   │
│  │  │ Web App │  │ CLI     │  │ SDK     │             │   │
│  │  └─────────┘  └─────────┘  └─────────┘             │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │ HTTP/gRPC                         │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   API Gateway                        │   │
│  │  ┌──────────────┐  ┌──────────────┐                 │   │
│  │  │ Load Balancer│  │ Rate Limiter │                 │   │
│  │  └──────────────┘  └──────────────┘                 │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Inference Engine                      │   │
│  │  ┌──────────────┐  ┌──────────────┐                 │   │
│  │  │ Async Engine │  │  Scheduler   │                 │   │
│  │  └──────────────┘  └──────────────┘                 │   │
│  │  ┌──────────────┐  ┌──────────────┐                 │   │
│  │  │   Workers    │  │ KV Cache     │                 │   │
│  │  └──────────────┘  └──────────────┘                 │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Model Storage                         │   │
│  │  ┌──────────────┐  ┌──────────────┐                 │   │
│  │  │ Model Weights│  │ Tokenizer    │                 │   │
│  │  └──────────────┘  └──────────────┘                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 核心知识点

#### 推理服务化的核心组件
| 组件 | 功能 | 关键技术 |
|------|------|----------|
| **API Gateway** | 请求路由、负载均衡 | FastAPI, Nginx |
| **Async Engine** | 异步推理执行 | asyncio, Ray |
| **Scheduler** | 请求调度、资源管理 | PagedAttention |
| **Workers** | 模型推理执行 | PyTorch, CUDA |
| **Model Storage** | 模型存储与加载 | HuggingFace, safetensors |

#### 服务化 vs 离线推理
| 特性 | 离线推理 | 在线服务 |
|------|----------|----------|
| **请求模式** | 批量处理 | 实时请求 |
| **延迟要求** | 宽松 | 严格 |
| **资源利用** | 高 | 需权衡 |
| **并发处理** | 简单 | 复杂 |

### 自测题

#### 问题
1. 推理服务化的核心组件有哪些？各自的职责是什么？
2. 在线服务与离线推理的主要区别是什么？
3. 为什么推理服务需要异步引擎？

#### 答案
1. **核心组件与职责**：
   - **API Gateway**：请求路由、负载均衡、限流熔断
   - **Async Engine**：异步推理执行，支持并发处理
   - **Scheduler**：请求调度、资源分配、优先级管理
   - **Workers**：模型推理执行，GPU计算
   - **Model Storage**：模型存储、版本管理、热加载
   - **KV Cache**：缓存管理、内存优化
   - **监控系统**：指标收集、告警通知、性能分析

2. **在线vs离线区别**：
   - **请求模式**：在线实时请求，离线批量处理
   - **延迟要求**：在线要求低延迟，离线延迟宽松
   - **并发处理**：在线需要高并发，离线顺序处理
   - **资源利用**：在线动态分配，离线固定资源
   - **容错要求**：在线高可用，离线可重试
   - **用户体验**：在线直接影响用户体验

3. **异步引擎的必要性**：
   - **高并发**：支持同时处理多个请求
   - **非阻塞**：避免单个请求阻塞整个服务
   - **流式输出**：支持实时流式响应
   - **资源利用**：更高效地利用GPU资源
   - **扩展性**：便于水平扩展和负载均衡

---

## Day 2: OpenAI API兼容接口

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 分析OpenAI API实现 | 理解接口设计 |
| 晚上 1.5h | API接口详解 | 能使用OpenAI SDK调用 |

### 学习材料

#### 源码位置
- `vllm/entrypoints/openai/api_server.py` - API服务实现
- `vllm/entrypoints/openai/protocol.py` - 协议定义

#### 核心API接口

##### 1. Completions API
```python
# 请求格式
POST /v1/completions
{
    "model": "llama-2-7b",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "n": 1,
    "stream": false
}

# 响应格式
{
    "id": "cmpl-123",
    "object": "text_completion",
    "created": 1234567890,
    "model": "llama-2-7b",
    "choices": [
        {
            "text": "...",
            "index": 0,
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 100,
        "total_tokens": 105
    }
}
```

##### 2. Chat Completions API
```python
# 请求格式
POST /v1/chat/completions
{
    "model": "llama-2-7b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": true
}

# 响应格式（非流式）
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "llama-2-7b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30
    }
}
```

### 核心知识点

#### 流式响应实现
```python
async def generate_stream(request: ChatCompletionRequest):
    """流式生成响应"""
    async for output in engine.generate(request):
        # 构造SSE格式响应
        chunk = {
            "id": output.id,
            "object": "chat.completion.chunk",
            "created": output.created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": output.text
                },
                "finish_reason": output.finish_reason
            }]
        }
        
        # 返回SSE格式
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # 结束标记
    yield "data: [DONE]\n\n"
```

#### OpenAI SDK调用示例
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# 非流式调用
response = client.chat.completions.create(
    model="llama-2-7b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)

# 流式调用
stream = client.chat.completions.create(
    model="llama-2-7b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 自测题

#### 问题
1. Completions API和Chat Completions API的区别是什么？
2. 流式响应如何实现？SSE格式是什么？
3. 如何使用OpenAI SDK调用vLLM服务？

#### 答案
1. **API区别**：
   - **Completions API**：
     - 输入：prompt字符串
     - 输出：直接文本补全
     - 用途：文本生成、续写
   - **Chat Completions API**：
     - 输入：messages列表（包含role和content）
     - 输出：对话式响应
     - 用途：对话系统、问答
   - **格式差异**：Chat API支持多轮对话，Completions API单轮生成

2. **流式响应实现**：
   - **SSE格式**：Server-Sent Events，基于HTTP的流式协议
   - **实现方式**：
     ```
     data: {"id":"chatcmpl-123","object":"chat.completion.chunk",...}
     data: {"choices":[{"delta":{"content":"Hello"}}]}
     data: [DONE]
     ```
   - **逐token推送**：每个token生成后立即发送
   - **客户端处理**：实时接收并显示

3. **OpenAI SDK调用**：
   ```python
   from openai import OpenAI
   
   client = OpenAI(
       base_url="http://localhost:8000/v1",
       api_key="dummy"  # vLLM不需要真实API key
   )
   
   # 非流式调用
   response = client.chat.completions.create(
       model="llama-2-7b",
       messages=[{"role": "user", "content": "Hello!"}]
   )
   
   # 流式调用
   stream = client.chat.completions.create(
       model="llama-2-7b",
       messages=[{"role": "user", "content": "Hello!"}],
       stream=True
   )
   for chunk in stream:
       if chunk.choices[0].delta.content:
           print(chunk.choices[0].delta.content, end="")
   ```

---

## Day 3: Async LLM Engine

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 分析Async Engine代码 | 理解异步机制 |
| 晚上 1.5h | 异步推理流程 | 能描述完整流程 |

### 学习材料

#### 源码位置
- `vllm/engine/async_llm_engine.py` - 异步引擎实现
- `vllm/engine/llm_engine.py` - 同步引擎基类

#### 核心类与数据结构
```python
class AsyncLLMEngine:
    """异步LLM引擎"""
    def __init__(self, engine_config):
        self.engine = LLMEngine(engine_config)
        self.request_tracker = RequestTracker()
        self.background_loop = None
        
    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str
    ) -> AsyncIterator[RequestOutput]:
        """异步生成接口"""
        # 1. 将请求加入队列
        self.request_tracker.add_request(
            request_id, prompt, sampling_params
        )
        
        # 2. 异步等待结果
        async for output in self.request_tracker.get_output(request_id):
            yield output
            
    async def run_engine_loop(self):
        """后台引擎循环"""
        while True:
            # 1. 调度请求
            scheduler_output = self.engine.scheduler.schedule()
            
            # 2. 执行推理
            output = await self.engine.step_async(scheduler_output)
            
            # 3. 更新请求状态
            self.request_tracker.update_outputs(output)
            
            # 4. 异步等待
            await asyncio.sleep(0)
```

### 核心知识点

#### 异步引擎架构
```
┌─────────────────────────────────────────────────────────────┐
│                  Async LLM Engine                           │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Request Tracker                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │ Request 1│  │ Request 2│  │ Request 3│          │   │
│  │  └──────────┘  └──────────┘  └──────────┘          │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Background Loop                         │   │
│  │  while True:                                         │   │
│  │    scheduler_output = scheduler.schedule()           │   │
│  │    output = await engine.step_async()                │   │
│  │    request_tracker.update(output)                    │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 LLM Engine                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │Scheduler │  │ Workers  │  │ KV Cache │          │   │
│  │  └──────────┘  └──────────┘  └──────────┘          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### 同步 vs 异步引擎对比
| 特性 | 同步引擎 | 异步引擎 |
|------|----------|----------|
| **调用方式** | 同步阻塞 | 异步非阻塞 |
| **并发处理** | 单请求 | 多请求并发 |
| **流式输出** | 不支持 | 原生支持 |
| **适用场景** | 离线批处理 | 在线服务 |

### 自测题

#### 问题
1. Async LLM Engine的核心组件有哪些？
2. 异步引擎如何实现并发处理多个请求？
3. Request Tracker的作用是什么？

#### 答案
1. **Async LLM Engine核心组件**：
   - **LLM Engine**：底层的同步推理引擎
   - **Request Tracker**：请求队列管理和状态跟踪
   - **Background Loop**：后台调度循环
   - **Async Generator**：异步生成器接口
   - **Event Loop**：事件循环机制

2. **并发处理机制**：
   - **事件驱动**：基于asyncio的事件循环
   - **请求队列**：所有请求进入Request Tracker队列
   - **后台调度**：Background Loop持续处理请求
   - **异步生成**：使用AsyncIterator逐token返回
   - **非阻塞**：不会阻塞主线程，支持高并发

3. **Request Tracker作用**：
   - **请求管理**：维护所有待处理请求
   - **状态跟踪**：跟踪每个请求的执行状态
   - **优先级管理**：支持请求优先级排序
   - **结果缓存**：缓存请求的中间结果
   - **超时处理**：处理请求超时和取消

---

## Day 4: 推理服务性能优化

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 分析性能瓶颈 | 理解优化方向 |
| 晚上 1.5h | 优化策略实践 | 能实施基础优化 |

### 学习材料

#### 性能优化维度

##### 1. 吞吐量优化
```python
# Continuous Batching
# 动态批处理，提高GPU利用率

# 配置示例
engine_config = EngineConfig(
    max_num_batched_tokens=8192,  # 单批次最大token数
    max_num_seqs=256,             # 最大并发序列数
    max_model_len=4096,           # 最大序列长度
)
```

##### 2. 延迟优化
```python
# Speculative Decoding
# 使用小模型预测，大模型验证

# 配置示例
speculative_config = SpeculativeConfig(
    num_speculative_tokens=4,  # 预测token数
    speculative_model="small-model",
)
```

##### 3. 内存优化
```python
# KV Cache优化
cache_config = CacheConfig(
    block_size=16,              # Block大小
    gpu_memory_utilization=0.9, # GPU内存利用率
    swap_space=4,               # CPU swap空间(GB)
)
```

### 核心知识点

#### 性能指标分析
```
┌─────────────────────────────────────────────────────────────┐
│                  推理服务性能指标                           │
│                                                              │
│  吞吐量指标：                                               │
│  ├── Requests Per Second (RPS)                             │
│  ├── Tokens Per Second (TPS)                               │
│  └── Batch Efficiency                                      │
│                                                              │
│  延迟指标：                                                 │
│  ├── Time to First Token (TTFT)                            │
│  ├── Inter-Token Latency (ITL)                             │
│  ├── End-to-End Latency (E2E)                              │
│  └── P50/P90/P99 Latency                                   │
│                                                              │
│  资源指标：                                                 │
│  ├── GPU Memory Usage                                      │
│  ├── GPU Utilization                                       │
│  ├── CPU Memory Usage                                      │
│  └── Network Bandwidth                                     │
└─────────────────────────────────────────────────────────────┘
```

#### 性能优化策略
| 策略 | 目标 | 方法 |
|------|------|------|
| **Continuous Batching** | 提高吞吐量 | 动态批处理 |
| **Paged Attention** | 降低内存 | 分页KV Cache |
| **Speculative Decoding** | 降低延迟 | 小模型预测 |
| **Quantization** | 降低内存/计算 | INT8/INT4量化 |
| **Prefix Caching** | 降低延迟 | 缓存公共前缀 |

### 自测题

#### 问题
1. 推理服务的关键性能指标有哪些？
2. Continuous Batching如何提高吞吐量？
3. Speculative Decoding的原理是什么？

#### 答案
1. **关键性能指标**：
   - **吞吐量指标**：
     - RPS (Requests Per Second)：每秒请求数
     - TPS (Tokens Per Second)：每秒token数
     - Batch Efficiency：批处理效率
   - **延迟指标**：
     - TTFT (Time To First Token)：首token延迟
     - ITL (Inter-Token Latency)：token间延迟
     - E2E Latency：端到端延迟
   - **资源指标**：
     - GPU Memory Usage：GPU内存使用率
     - GPU Utilization：GPU利用率
     - KV Cache Usage：KV缓存使用率

2. **Continuous Batching原理**：
   - **动态批处理**：新请求可随时加入当前批次
   - **序列完成即释放**：完成的序列立即释放资源
   - **持续调度**：调度器持续检查新请求
   - **资源复用**：释放的资源立即被新请求使用
   - **提高利用率**：减少GPU空闲时间，提高吞吐量

3. **Speculative Decoding原理**：
   - **双模型架构**：小模型预测 + 大模型验证
   - **并行生成**：小模型一次生成多个token
   - **批量验证**：大模型批量验证所有预测
   - **接受机制**：接受匹配的token，拒绝重新生成
   - **加速效果**：减少大模型调用次数，降低延迟

---

## Day 5: 推理服务监控与运维

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 工作中 | 分析监控指标 | 理解监控体系 |
| 晚上 1.5h | 部署监控方案 | 能搭建监控系统 |

### 学习材料

#### 监控指标体系

##### 1. Prometheus指标
```python
# vLLM内置Prometheus指标
from prometheus_client import Counter, Histogram, Gauge

# 请求计数
REQUEST_COUNT = Counter(
    'vllm_request_count',
    'Total number of requests',
    ['model', 'status']
)

# 请求延迟
REQUEST_LATENCY = Histogram(
    'vllm_request_latency_seconds',
    'Request latency in seconds',
    ['model', 'endpoint']
)

# GPU内存使用
GPU_MEMORY_USAGE = Gauge(
    'vllm_gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['gpu_id']
)

# KV Cache使用率
KV_CACHE_USAGE = Gauge(
    'vllm_kv_cache_usage_ratio',
    'KV cache usage ratio',
    ['model']
)
```

##### 2. Grafana Dashboard
```yaml
# Grafana Dashboard配置示例
dashboard:
  title: "vLLM Inference Service"
  panels:
    - title: "Request Rate"
      type: graph
      targets:
        - expr: rate(vllm_request_count[5m])
    
    - title: "Latency P99"
      type: graph
      targets:
        - expr: histogram_quantile(0.99, rate(vllm_request_latency_seconds_bucket[5m]))
    
    - title: "GPU Memory"
      type: gauge
      targets:
        - expr: vllm_gpu_memory_usage_bytes
    
    - title: "KV Cache Usage"
      type: gauge
      targets:
        - expr: vllm_kv_cache_usage_ratio
```

### 核心知识点

#### 监控架构
```
┌─────────────────────────────────────────────────────────────┐
│                    监控架构                                 │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 vLLM Service                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │ Metrics  │  │  Logs    │  │  Traces  │          │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘          │   │
│  └───────┼─────────────┼─────────────┼─────────────────┘   │
│          │             │             │                      │
│          ▼             ▼             ▼                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ Prometheus  │ │   Loki      │ │   Jaeger    │           │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘           │
│         │               │               │                   │
│         └───────────────┼───────────────┘                   │
│                         ▼                                   │
│              ┌─────────────────┐                            │
│              │    Grafana      │                            │
│              └─────────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

#### 告警规则示例
```yaml
# Prometheus告警规则
groups:
  - name: vllm_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(vllm_request_latency_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High request latency"
          
      - alert: HighGPUMemory
        expr: vllm_gpu_memory_usage_bytes / gpu_total_memory_bytes > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory usage too high"
          
      - alert: KVCacheFull
        expr: vllm_kv_cache_usage_ratio > 0.95
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "KV cache almost full"
```

### 自测题

#### 问题
1. 推理服务需要监控哪些关键指标？
2. 如何搭建完整的监控体系？
3. 常见的告警规则有哪些？

---

## Day 6: 实战 - 部署完整推理服务

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 晚上 2h | 部署完整推理服务 | 能独立部署服务 |

### 实战要求

#### 任务描述
部署一个完整的vLLM推理服务，包含：

1. **服务部署**：
```bash
# 启动vLLM服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```

2. **监控部署**：
```yaml
# docker-compose.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

3. **负载测试**：
```python
# 使用Locust进行负载测试
from locust import HttpUser, task, between

class VLLMUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def chat_completion(self):
        self.client.post("/v1/chat/completions", json={
            "model": "llama-2-7b",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 100
        })
```

#### 验证标准
- [ ] 服务成功启动并可访问
- [ ] 监控系统正常工作
- [ ] 负载测试通过

---

## Day 7: 本周复盘 + 第二个月阶段测试

### 学习任务
| 时间 | 任务 | 目标 |
|------|------|------|
| 晚上 1h | 本周复盘 + 阶段测试 | 检验学习效果 |

### 本周知识点回顾

#### 知识点清单
- [ ] 推理服务化架构设计
- [ ] OpenAI API兼容接口
- [ ] Async LLM Engine实现
- [ ] 推理服务性能优化
- [ ] 监控与运维体系

### 第二个月阶段测试 (满分100, 及格65)

#### 问题
1. **vLLM源码深度** (40分)
   - 描述vLLM调度器的完整工作流程 (10分)
   - Block Manager如何实现高效的内存管理？ (10分)
   - Tensor Parallelism的实现原理是什么？ (10分)
   - Flash Attention如何优化内存访问？ (10分)

2. **推理优化** (30分)
   - 对比四种量化方法(LLM.int8()/SmoothQuant/AWQ/GPTQ)的优缺点 (15分)
   - 如何分析推理性能瓶颈？优化策略有哪些？ (15分)

3. **推理服务化** (30分)
   - Async LLM Engine如何实现并发处理？ (10分)
   - OpenAI API的流式响应如何实现？ (10分)
   - 推理服务需要监控哪些关键指标？如何告警？ (10分)

#### 答案要点

**1. vLLM源码深度**：

**调度器工作流程**：
- 请求到达 → 进入Waiting队列
- 调度器检查资源 → 分配Block → 移到Running队列
- Running序列生成token → 更新KV Cache
- 内存不足时 → Preemption → 移到Swapped队列
- 序列完成 → 移除并释放资源

**Block Manager优势**：
- 分页管理，减少内存碎片
- 按需分配，提高内存利用率
- 支持变长序列，无需预分配固定大小
- Block可跨序列复用

**Tensor Parallelism原理**：
- 层内切分，权重按列/行切分到不同GPU
- 每个GPU计算部分结果
- 通过All-Reduce汇聚结果
- 适合单机多卡场景

**Flash Attention优化**：
- 分块计算，不存储完整注意力矩阵
- 在线Softmax，逐块更新
- 内存占用从O(N²)降到O(N)
- 利用GPU内存层次结构

**2. 推理优化**：

**四种量化方法对比**：
| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| LLM.int8() | 精度高，通用 | 需检测异常值 | 通用LLM量化 |
| SmoothQuant | 无需混合精度 | 需校准数据 | 激活范围大的模型 |
| AWQ | INT4高精度 | 需校准数据 | 追求精度的场景 |
| GPTQ | 高压缩率 | 计算复杂 | 追求压缩的场景 |

**性能瓶颈分析与优化**：
- 瓶颈类型：计算密集、内存密集、IO密集
- 分析工具：Profiling、监控指标
- 优化策略：量化、算子融合、批处理、缓存

**3. 推理服务化**：

**Async Engine并发处理**：
- Request Tracker管理请求队列
- Background Loop持续调度
- 异步等待结果，非阻塞
- 支持流式输出

**流式响应实现**：
- SSE (Server-Sent Events) 格式
- 逐token生成并推送
- 客户端实时接收
- 结束标记[DONE]

**监控指标与告警**：
- 关键指标：RPS、TPS、TTFT、ITL、GPU内存
- 告警规则：高延迟、高内存、KV Cache满
- 监控栈：Prometheus + Grafana