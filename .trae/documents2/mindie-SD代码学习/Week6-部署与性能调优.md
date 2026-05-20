# Week6 - 部署与性能调优

## 学习目标

深入理解MindIE-SD的部署策略和性能调优方法，掌握如何构建高可用、高性能的推理服务。

---

## 1. 服务部署

### 1.1 HTTP/JSON接口

```python
class MindIEServer:
    """MindIE-SD推理服务器"""
    
    def __init__(self, config):
        self.config = config
        self.backend = MindIEBackend(config)
        self.app = self._create_app()
    
    def _create_app(self):
        """创建Flask应用"""
        app = Flask(__name__)
        
        @app.route("/v1/images/generations", methods=["POST"])
        def generate_image():
            """生成图像"""
            # 解析请求
            data = request.json
            prompt = data.get("prompt")
            size = data.get("size", "1024x1024")
            steps = data.get("steps", 50)
            
            # 调用后端
            result = self.backend.generate(prompt, size, steps)
            
            # 返回结果
            return jsonify(result)
        
        return app
    
    def run(self, host="0.0.0.0", port=8000):
        """运行服务器"""
        self.app.run(host=host, port=port)
```

### 1.2 流式输出

```python
class StreamServer:
    """流式输出服务器"""
    
    def __init__(self, config):
        self.config = config
        self.backend = MindIEBackend(config)
        self.app = self._create_app()
    
    def _create_app(self):
        """创建Flask应用"""
        app = Flask(__name__)
        
        @app.route("/v1/images/generations/stream", methods=["POST"])
        def stream_generate():
            """流式生成图像"""
            # 解析请求
            data = request.json
            prompt = data.get("prompt")
            size = data.get("size", "1024x1024")
            steps = data.get("steps", 50)
            
            def generate():
                """生成器函数"""
                # 流式生成
                for step, image in enumerate(self.backend.stream_generate(prompt, size, steps)):
                    # 转换为base64
                    image_base64 = base64.b64encode(image).decode("utf-8")
                    # 发送进度
                    yield f"data: {json.dumps({"progress": (step+1)/steps, "image": image_base64})}\n\n"
            
            # 返回SSE响应
            return Response(generate(), mimetype="text/event-stream")
        
        return app
```

---

## 2. 性能测试与分析

### 2.1 性能测试

```python
class PerformanceTester:
    """性能测试器"""
    
    def __init__(self, backend):
        self.backend = backend
    
    def test_latency(self, prompt, size="1024x1024", steps=50, iterations=10):
        """测试延迟"""
        latencies = []
        
        for i in range(iterations):
            start = time.time()
            self.backend.generate(prompt, size, steps)
            end = time.time()
            latencies.append(end - start)
        
        return {
            "average": sum(latencies) / len(latencies),
            "min": min(latencies),
            "max": max(latencies)
        }
    
    def test_throughput(self, prompts, size="1024x1024", steps=50):
        """测试吞吐量"""
        start = time.time()
        
        for prompt in prompts:
            self.backend.generate(prompt, size, steps)
        
        end = time.time()
        throughput = len(prompts) / (end - start)
        
        return throughput
```

### 2.2 性能分析

```python
class Profiler:
    """性能分析器"""
    
    def __init__(self, backend):
        self.backend = backend
    
    def profile(self, prompt, size="1024x1024", steps=50):
        """分析性能"""
        # 启用CUDA分析
        with torch.profiler.profile(
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
            record_shapes=True,
            with_stack=True
        ) as prof:
            # 执行推理
            self.backend.generate(prompt, size, steps)
        
        # 分析结果
        print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## 3. 监控与告警

### 3.1 监控系统

```python
class Monitor:
    """监控系统"""
    
    def __init__(self):
        self.metrics = {
            "latency": [],
            "throughput": [],
            "memory_usage": [],
            "gpu_utilization": []
        }
    
    def record_latency(self, latency):
        """记录延迟"""
        self.metrics["latency"].append(latency)
    
    def record_throughput(self, throughput):
        """记录吞吐量"""
        self.metrics["throughput"].append(throughput)
    
    def record_memory_usage(self, usage):
        """记录内存使用"""
        self.metrics["memory_usage"].append(usage)
    
    def record_gpu_utilization(self, utilization):
        """记录GPU利用率"""
        self.metrics["gpu_utilization"].append(utilization)
    
    def get_metrics(self):
        """获取指标"""
        return self.metrics
```

### 3.2 告警系统

```python
class AlertSystem:
    """告警系统"""
    
    def __init__(self, thresholds):
        self.thresholds = thresholds
    
    def check_thresholds(self, metrics):
        """检查阈值"""
        alerts = []
        
        # 检查延迟
        if metrics.get("latency") and metrics["latency"][-1] > self.thresholds.get("latency", 5.0):
            alerts.append({"type": "latency", "message": "Latency exceeded threshold"})
        
        # 检查内存使用
        if metrics.get("memory_usage") and metrics["memory_usage"][-1] > self.thresholds.get("memory_usage", 80):
            alerts.append({"type": "memory", "message": "Memory usage exceeded threshold"})
        
        # 检查GPU利用率
        if metrics.get("gpu_utilization") and metrics["gpu_utilization"][-1] > self.thresholds.get("gpu_utilization", 90):
            alerts.append({"type": "gpu", "message": "GPU utilization exceeded threshold"})
        
        return alerts
```

---

## 4. 生产环境最佳实践

### 4.1 高可用性

```python
class HighAvailability:
    """高可用性"""
    
    def __init__(self, servers):
        self.servers = servers
        self.current_server = 0
    
    def get_server(self):
        """获取服务器"""
        server = self.servers[self.current_server]
        self.current_server = (self.current_server + 1) % len(self.servers)
        return server
    
    def health_check(self):
        """健康检查"""
        healthy_servers = []
        for server in self.servers:
            if self._check_server_health(server):
                healthy_servers.append(server)
        self.servers = healthy_servers
    
    def _check_server_health(self, server):
        """检查服务器健康状态"""
        try:
            response = requests.get(f"http://{server}/health")
            return response.status_code == 200
        except:
            return False
```

### 4.2 自动扩缩容

```python
class AutoScaler:
    """自动扩缩容"""
    
    def __init__(self, min_instances=1, max_instances=10):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.instances = []
    
    def scale(self, load):
        """根据负载扩缩容"""
        # 计算需要的实例数
        required_instances = max(self.min_instances, min(self.max_instances, int(load / 10)))
        
        # 扩缩容
        if len(self.instances) < required_instances:
            # 扩容
            for i in range(len(self.instances), required_instances):
                instance = self._create_instance()
                self.instances.append(instance)
        elif len(self.instances) > required_instances:
            # 缩容
            for i in range(len(self.instances) - required_instances):
                instance = self.instances.pop()
                self._destroy_instance(instance)
    
    def _create_instance(self):
        """创建实例"""
        # 创建实例
        # ...
        return instance
    
    def _destroy_instance(self, instance):
        """销毁实例"""
        # 销毁实例
        # ...
```

### 4.3 故障处理

```python
class FaultHandler:
    """故障处理器"""
    
    def __init__(self):
        pass
    
    def handle_fault(self, error):
        """处理故障"""
        # 记录错误
        self._log_error(error)
        
        # 重试
        if self._should_retry(error):
            return self._retry()
        
        # 降级
        if self._should_degrade(error):
            return self._degrade()
        
        # 抛出异常
        raise error
    
    def _log_error(self, error):
        """记录错误"""
        print(f"Error: {error}")
    
    def _should_retry(self, error):
        """判断是否应该重试"""
        return isinstance(error, (TimeoutError, ConnectionError))
    
    def _retry(self):
        """重试"""
        # 重试逻辑
        # ...
    
    def _should_degrade(self, error):
        """判断是否应该降级"""
        return isinstance(error, OutOfMemoryError)
    
    def _degrade(self):
        """降级"""
        # 降级逻辑
        # ...
```

---

## 5. 代码阅读重点

### 5.1 service/server.py

**核心类**：
- `MindIEServer`：推理服务器
- `StreamServer`：流式输出服务器

**关键方法**：
- `run()`：运行服务器
- `_create_app()`：创建Flask应用
- `generate_image()`：生成图像

### 5.2 benchmarks/performance_tester.py

**核心类**：
- `PerformanceTester`：性能测试器
- `Profiler`：性能分析器

**关键方法**：
- `test_latency()`：测试延迟
- `test_throughput()`：测试吞吐量
- `profile()`：分析性能

### 5.3 service/monitoring.py

**核心类**：
- `Monitor`：监控系统
- `AlertSystem`：告警系统
- `HighAvailability`：高可用性
- `AutoScaler`：自动扩缩容
- `FaultHandler`：故障处理器

**关键方法**：
- `record_latency()`：记录延迟
- `check_thresholds()`：检查阈值
- `health_check()`：健康检查
- `scale()`：扩缩容
- `handle_fault()`：处理故障

---

## 6. 学习笔记

### 6.1 部署策略

部署策略：
1. **服务化部署**：提供HTTP/JSON接口，支持标准调用
2. **流式输出**：支持SSE/WebSocket，实时展示生成过程
3. **多模型部署**：支持多个模型同时部署，动态加载/卸载
4. **分布式部署**：支持多卡并行推理，扩展模型容量和吞吐量

### 6.2 性能调优

性能调优：
1. **算子融合**：减少kernel launch开销，提高计算密度
2. **内存优化**：减少内存占用，提高内存利用率
3. **量化技术**：降低计算精度，提高计算速度
4. **多流并行**：利用多流特性，并行处理多个任务

### 6.3 监控与维护

监控与维护：
1. **监控系统**：实时监控系统状态，收集性能指标
2. **告警系统**：设置阈值，及时发现并处理问题
3. **高可用性**：实现负载均衡，确保服务稳定
4. **自动扩缩容**：根据负载自动调整实例数
5. **故障处理**：实现重试、降级等故障处理机制

---

## 7. 性能对比

| 指标 | 标准部署 | 优化部署 | 提升比例 |
|------|----------|----------|----------|
| 延迟 | 3.0s | 1.0s | 200% |
| 吞吐量 | 1.0 | 5.0 | 400% |
| 可用性 | 99.0% | 99.9% | 0.9% |
| 资源利用率 | 50% | 80% | 60% |

---

## 8. 自测问题

1. 如何部署MindIE-SD服务？
2. 如何测试和分析性能？
3. 如何实现高可用性？
4. 如何处理故障？

---

## 9. 总结

完成Week6学习后，我们已经全面掌握了MindIE-SD的部署与性能调优技术。通过合理的部署策略和性能优化，可以构建高可用、高性能的推理服务，满足各种应用场景的需求。