# vLLM-Ascend 框架代码学习 - Week 1

> 学习主题：项目结构与平台抽象层
> 学习目标：理解vLLM-Ascend如何注册为vLLM插件，掌握Worker初始化流程
> 官方仓库：<https://github.com/vllm-project/vllm-ascend>

***

## 一、学习进度

| 日期      | 内容        | 状态  |
| ------- | --------- | --- |
| Day 1-2 | 项目入口与初始化  | 进行中 |
| Day 3-4 | Worker架构  | 待开始 |
| Day 5-6 | 模型加载与权重管理 | 待开始 |
| Day 7   | 本周复盘      | 待开始 |

***

## 二、vLLM-Ascend 架构总览

### 2.1 项目定位

vLLM-Ascend 是 vLLM 项目的一个**社区维护的硬件插件**，专为华为昇腾（Ascend）NPU 设计。它遵循 vLLM 社区的 **Hardware Pluggable 架构**，实现了硬件后端的解耦。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        vLLM 架构层次                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    用户层 (User Layer)                          │   │
│  │  LLM类 / SamplingParams / Offline Inference / Online Serve     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    核心层 (Core Layer)                          │   │
│  │  Scheduler / BlockManager / CacheEngine / ModelExecutor         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              平台抽象层 (Platform Abstraction)                   │   │
│  │  Platform / Worker / ModelRunner / Attention / Communicator     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│          ┌─────────────────────┼─────────────────────┐                 │
│          ▼                     ▼                     ▼                 │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐          │
│  │ CUDA Plugin │       │Ascend Plugin│       │ Other Plugin│          │
│  │ (vLLM内置)  │       │(vllm-ascend)│       │  (扩展中)   │          │
│  └─────────────┘       └─────────────┘       └─────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计理念：控制反转（IoC）

vLLM-Ascend 的核心在于它如何让 vLLM "看见" 自己。这一机制依赖于 **Python 的 entry\_points**：

```python
# setup.py 中的关键定义
setup(
    name="vllm-ascend",
    entry_points={
        'vllm.platform_plugins': [
            "ascend = vllm_ascend:register"
        ]
    }
)
```

**关键解读**：

* `vllm.platform_plugins`：这是 vLLM 主库预留的扩展接口组

* `ascend = vllm_ascend:register`：告诉 vLLM，当扫描 platform\_plugins 时，如果看到 `ascend` 标签，请执行 `vllm_ascend` 包下的 `register` 函数

这种设计实现了**控制反转（IoC）**：

* vLLM 不需要硬编码 `import vllm_ascend`

* 而是在运行时动态发现并加载插件

* 如果环境中安装了该包，vLLM 启动时会自动尝试初始化 NPU 环境

***

## 三、项目目录结构详解

### 3.1 顶层目录结构

```
vllm-ascend/
├── vllm_ascend/                  # 核心代码目录
│   ├── __init__.py              # 包入口，包含register函数
│   ├── platform.py              # 平台抽象层实现
│   ├── worker/                  # Worker实现
│   │   ├── worker.py           # 主Worker类
│   │   ├── model_runner.py     # 模型运行器
│   │   └── ...
│   ├── attention/               # Attention实现
│   │   ├── attention.py        # 基础Attention
│   │   ├── mla_v1.py           # Multi-Latent Attention
│   │   ├── sfa_v1.py           # Sparse Flash Attention
│   │   └── ...
│   ├── ops/                     # 算子实现
│   │   ├── attention.py        # Attention算子
│   │   ├── layernorm.py        # LayerNorm算子
│   │   └── ...
│   ├── torchair/                # TorchAir图编译
│   │   ├── torchair_mla.py     # MLA图模式
│   │   └── ...
│   ├── eplb/                    # 专家并行负载均衡
│   │   ├── core/
│   │   │   └── policy/         # 负载均衡策略
│   │   └── ...
│   ├── quantization/            # 量化实现
│   │   ├── awq.py
│   │   ├── smoothquant.py
│   │   └── ...
│   ├── patch/                   # 动态补丁系统
│   │   ├── patch_0_9_2.py      # 针对vLLM 0.9.2的补丁
│   │   └── ...
│   └── utils/                   # 工具函数
├── csrc/                        # C++扩展
│   ├── tiling_base.h           # 算子分块
│   └── ...
├── benchmarks/                  # 性能测试
├── tests/                       # 测试用例
├── examples/                    # 示例代码
├── setup.py                     # 构建配置
└── pyproject.toml              # 项目配置
```

### 3.2 核心模块职责

| 目录              | 职责          | 关键文件                                     |
| --------------- | ----------- | ---------------------------------------- |
| `vllm_ascend/`  | 核心适配逻辑      | `__init__.py`, `platform.py`             |
| `worker/`       | Worker和模型运行 | `worker.py`, `model_runner.py`           |
| `attention/`    | 注意力机制实现     | `attention.py`, `mla_v1.py`, `sfa_v1.py` |
| `ops/`          | 底层算子        | `attention.py`, `layernorm.py`           |
| `torchair/`     | 图编译优化       | `torchair_mla.py`                        |
| `eplb/`         | 专家并行负载均衡    | `core/policy/`                           |
| `quantization/` | 量化实现        | `awq.py`, `smoothquant.py`               |
| `patch/`        | 动态补丁        | `patch_*.py`                             |
| `csrc/`         | C++算子扩展     | `tiling_base.h`                          |

***

## 四、代码阅读：项目入口与初始化

### 4.1 `vllm_ascend/__init__.py` - 包入口

**核心功能**：

1. 定义版本信息
2. 导出核心接口
3. 提供 `register` 函数供 vLLM 调用

**关键代码解读**：

```python
# vllm_ascend/__init__.py

__version__ = "0.7.3"  # 版本号

def register():
    """
    注册函数 - vLLM 通过 entry_points 调用此函数
    
    主要职责：
    1. 检测 NPU 环境
    2. 注册 Ascend 平台
    3. 应用补丁
    4. 初始化必要组件
    """
    # 1. 检测 torch_npu 是否可用
    try:
        import torch_npu
    except ImportError:
        raise ImportError(
            "torch_npu is required for vLLM-Ascend. "
            "Please install it first."
        )
    
    # 2. 注册平台
    from vllm_ascend.platform import AscendPlatform
    from vllm import platform
    
    # 将 AscendPlatform 注册到 vLLM 的平台注册表
    platform.register_platform("ascend", AscendPlatform)
    
    # 3. 应用补丁（根据 vLLM 版本）
    from vllm_ascend.patch import apply_patches
    apply_patches()
    
    # 4. 初始化日志
    from vllm_ascend.utils import init_logger
    init_logger()

# 导出核心接口
from vllm_ascend.platform import AscendPlatform
from vllm_ascend.worker.worker import AscendWorker
```

**关键问题解答**：

**Q: 如何注册为vLLM插件？**

A: 通过 Python 的 entry\_points 机制：

1. 在 `setup.py` 中声明 `vllm.platform_plugins` 入口点
2. 提供 `register` 函数作为入口
3. vLLM 启动时会扫描已安装的插件并调用 `register` 函数

***

### 4.2 `vllm_ascend/platform.py` - 平台抽象层

**核心功能**：
平台抽象层是 NPU 与 vLLM 交互的桥梁。vLLM 的上层逻辑（如 Scheduler、Worker）是设备无关的，当它们需要查询显存、设置设备或同步流时，会调用 Platform 接口。

**关键代码解读**：

```python
# vllm_ascend/platform.py

import torch
import torch_npu
from vllm.platforms import Platform

class AscendPlatform(Platform):
    """
    昇腾平台实现
    
    继承自 vLLM 的 Platform 基类，实现 NPU 特定的操作
    """
    
    @classmethod
    def get_device_name(cls) -> str:
        """返回设备名称"""
        return "ascend"
    
    @classmethod
    def is_async_output_supported(cls) -> bool:
        """是否支持异步输出"""
        return True
    
    @classmethod
    def set_device(cls, device_id: int):
        """
        设置当前设备
        
        对应 CUDA: torch.cuda.set_device(device_id)
        对应 NPU: torch_npu.npu.set_device(device_id)
        """
        torch_npu.npu.set_device(device_id)
    
    @classmethod
    def current_device(cls) -> int:
        """获取当前设备ID"""
        return torch_npu.npu.current_device()
    
    @classmethod
    def synchronize(cls):
        """
        同步设备流
        
        确保所有 NPU 操作完成
        """
        torch_npu.npu.synchronize()
    
    @classmethod
    def get_device_total_memory(cls, device_id: int) -> int:
        """
        获取设备总内存（HBM）
        
        用于计算 KV Cache 可用空间
        """
        # torch_npu.npu.mem_get_info() 返回 (free, total)
        _, total = torch_npu.npu.mem_get_info(device_id)
        return total
    
    @classmethod
    def get_device_available_memory(cls, device_id: int) -> int:
        """
        获取设备可用内存
        
        用于调度器判断是否能容纳新请求
        """
        free, _ = torch_npu.npu.mem_get_info(device_id)
        return free
    
    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """是否支持锁页内存"""
        # NPU 暂不支持 CUDA 的 pin_memory
        return False
    
    @classmethod
    def get_punica_wrapper(cls):
        """获取 Punica 包装器（用于 LoRA）"""
        from vllm_ascend.lora.punica import AscendPunicaWrapper
        return AscendPunicaWrapper()
```

**关键问题解答**：

**Q: 如何抽象NPU硬件差异？**

A: 通过继承 vLLM 的 `Platform` 基类并实现所有抽象方法：

1. **设备管理**：封装 `torch_npu.npu.set_device` 和 `torch_npu.npu.synchronize`
2. **内存透视**：通过 `torch_npu.npu.mem_get_info()` 获取 NPU 内存信息
3. **特性检测**：返回 NPU 支持的特性（如异步输出）

**Q: 与 CUDA Platform 的差异？**

| 功能     | CUDA Platform             | Ascend Platform              |
| ------ | ------------------------- | ---------------------------- |
| 设备设置   | `torch.cuda.set_device`   | `torch_npu.npu.set_device`   |
| 内存查询   | `torch.cuda.mem_get_info` | `torch_npu.npu.mem_get_info` |
| 同步     | `torch.cuda.synchronize`  | `torch_npu.npu.synchronize`  |
| 锁页内存   | 支持                        | 不支持                          |
| Stream | CUDA Stream               | NPU Stream                   |

***

### 4.3 `vllm_ascend/patch/` - 动态补丁系统

**为什么需要补丁系统？**

这是 vLLM-Ascend 最具工程智慧也最"无奈"的设计。原因如下：

1. vLLM 主干代码迭代极快
2. 部分 CUDA 语义（如 CUDA Graph）无法直接映射到 NPU 的 ACL Graph
3. 插件层必须在运行时对 vLLM 的核心代码进行"热修补"（Monkey Patching）

**工作原理**：

```python
# vllm_ascend/patch/__init__.py

import vllm

def apply_patches():
    """
    根据 vLLM 版本应用对应的补丁
    """
    version = vllm.__version__
    
    if version.startswith("0.6."):
        from .patch_0_6_x import apply_patch
        apply_patch()
    elif version.startswith("0.7."):
        from .patch_0_7_x import apply_patch
        apply_patch()
    else:
        # 使用最新补丁
        from .patch_latest import apply_patch
        apply_patch()
```

**典型修补点**：

```python
# vllm_ascend/patch/patch_latest.py

def apply_patch():
    # 1. 修补 ModelRunner - 替换 CUDA Graph 捕获逻辑
    from vllm.worker.model_runner import ModelRunner
    original_capture = ModelRunner.capture_model
    
    def ascend_capture(self, *args, **kwargs):
        # 使用 Torch-NPU 的 Graph 模式
        # 或禁用 Graph 模式
        pass
    
    ModelRunner.capture_model = ascend_capture
    
    # 2. 修补 Worker - 修改权重加载方式
    from vllm.worker.worker import Worker
    original_load = Worker.load_model
    
    def ascend_load(self, *args, **kwargs):
        # 确保 Tensor 被正确放置在 npu 设备
        pass
    
    Worker.load_model = ascend_load
```

***

## 五、代码阅读：Worker架构

### 5.1 `vllm_ascend/worker/worker.py` - Worker主类

**核心功能**：
Worker 是 vLLM 中负责模型推理的核心组件。每个 GPU/NPU 对应一个 Worker。

**关键代码解读**：

```python
# vllm_ascend/worker/worker.py

from vllm.worker.worker_base import WorkerBase
from vllm_ascend.worker.model_runner import AscendModelRunner

class AscendWorker(WorkerBase):
    """
    昇腾 Worker 实现
    
    负责：
    1. 初始化模型
    2. 加载权重
    3. 执行推理
    4. 管理 KV Cache
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_runner: Optional[AscendModelRunner] = None
    
    def init_model(self):
        """
        初始化模型
        
        主要步骤：
        1. 设置 NPU 设备
        2. 创建 ModelRunner
        3. 分配 KV Cache
        """
        # 1. 设置设备
        import torch_npu
        torch_npu.npu.set_device(self.device)
        
        # 2. 创建 ModelRunner
        self.model_runner = AscendModelRunner(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device=self.device,
        )
        
        # 3. 加载模型
        self.model_runner.load_model()
    
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "ModelOutput":
        """
        执行模型推理
        
        这是 Worker 的核心方法，由 Scheduler 调用
        
        Args:
            scheduler_output: 调度器输出，包含要处理的请求信息
            
        Returns:
            ModelOutput: 模型输出，包含生成的 token
        """
        # 1. 准备输入
        input_tokens = scheduler_output.scheduled_token_ids
        input_positions = scheduler_output.scheduled_position_ids
        block_tables = scheduler_output.block_tables
        
        # 2. 执行前向传播
        output = self.model_runner.execute_model(
            input_tokens=input_tokens,
            input_positions=input_positions,
            block_tables=block_tables,
        )
        
        return output
    
    def profile_run(self) -> int:
        """
        预热运行，计算 KV Cache 可用空间
        
        Returns:
            可用的 KV Cache Block 数量
        """
        # 运行一次推理以确定内存占用
        # 然后计算剩余内存可分配多少 Block
        pass
```

### 5.2 `vllm_ascend/worker/model_runner.py` - 模型运行器

**核心功能**：
ModelRunner 负责实际的模型执行，包括前向传播、采样等。

**关键代码解读**：

```python
# vllm_ascend/worker/model_runner.py

class AscendModelRunner:
    """
    昇腾模型运行器
    
    负责：
    1. 加载模型权重
    2. 执行前向传播
    3. 管理 KV Cache
    """
    
    def __init__(
        self,
        model_config,
        parallel_config,
        scheduler_config,
        device,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device = device
        
        self.model = None
        self.block_size = model_config.block_size
    
    def load_model(self):
        """
        加载模型
        
        主要步骤：
        1. 获取模型类
        2. 实例化模型
        3. 加载权重
        4. 移动到 NPU
        """
        # 1. 获取模型类（根据架构名称）
        from vllm.model_executor.models import get_model_architecture
        model_class = get_model_architecture(self.model_config.architecture)
        
        # 2. 实例化模型
        self.model = model_class(self.model_config)
        
        # 3. 加载权重
        self.model.load_weights(self.model_config.model_path)
        
        # 4. 移动到 NPU
        self.model = self.model.to(self.device)
        
        # 5. 设置为评估模式
        self.model.eval()
    
    def execute_model(
        self,
        input_tokens: torch.Tensor,
        input_positions: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行模型前向传播
        
        Args:
            input_tokens: [num_tokens] 输入 token ID
            input_positions: [num_tokens] 位置 ID
            block_tables: [num_seqs, max_blocks] Block 表
            
        Returns:
            logits: [num_tokens, vocab_size] 输出 logits
        """
        # 1. 准备输入
        input_ids = input_tokens.to(self.device)
        positions = input_positions.to(self.device)
        
        # 2. 执行前向传播
        with torch.no_grad():
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                block_tables=block_tables,
            )
        
        # 3. 计算 logits
        logits = self.model.compute_logits(hidden_states)
        
        return logits
```

***

## 六、架构图

### 6.1 模块依赖关系

```
                    ┌─────────────────┐
                    │   vLLM Core     │
                    │  (设备无关层)    │
                    └────────┬────────┘
                             │
                             │ Platform 接口
                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        vllm_ascend 插件层                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ platform.py │  │  worker/    │  │ attention/  │  │    ops/     │   │
│  │ 平台抽象层  │──│   Worker    │──│  Attention  │──│   算子层    │   │
│  │             │  │ ModelRunner │  │   Backend   │  │             │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│         │                │                │                │          │
│         │                │                │                │          │
│         └────────────────┴────────────────┴────────────────┘          │
│                                   │                                    │
│                                   │ torch_npu API                      │
│                                   ▼                                    │
│                        ┌─────────────────┐                             │
│                        │    torch_npu    │                             │
│                        │  (华为 PyTorch  │                             │
│                        │    适配层)      │                             │
│                        └────────┬────────┘                             │
│                                 │                                      │
│                                 │ CANN API                             │
│                                 ▼                                      │
│                        ┌─────────────────┐                             │
│                        │      CANN       │                             │
│                        │  (昇腾计算架构)  │                             │
│                        └────────┬────────┘                             │
│                                 │                                      │
│                                 │ Driver API                           │
│                                 ▼                                      │
│                        ┌─────────────────┐                             │
│                        │   Ascend NPU    │                             │
│                        │    (硬件层)      │                             │
│                        └─────────────────┘                             │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Worker 初始化流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Worker 初始化流程                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. vLLM 启动                                                           │
│     │                                                                   │
│     ▼                                                                   │
│  2. 扫描 entry_points，发现 vllm-ascend                                 │
│     │                                                                   │
│     ▼                                                                   │
│  3. 调用 vllm_ascend.register()                                        │
│     ├── 检测 torch_npu                                                  │
│     ├── 注册 AscendPlatform                                             │
│     └── 应用补丁                                                        │
│     │                                                                   │
│     ▼                                                                   │
│  4. 创建 Worker                                                         │
│     │                                                                   │
│     ▼                                                                   │
│  5. Worker.init_model()                                                 │
│     ├── torch_npu.npu.set_device(device_id)                            │
│     ├── 创建 AscendModelRunner                                          │
│     │   ├── 获取模型类                                                  │
│     │   ├── 实例化模型                                                  │
│     │   ├── 加载权重                                                    │
│     │   └── model.to(device)                                           │
│     └── 分配 KV Cache                                                   │
│     │                                                                   │
│     ▼                                                                   │
│  6. Worker 就绪，等待调度                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

***

## 七、与vLLM原版差异

| 模块          | vLLM原版 (CUDA)  | vLLM-Ascend (NPU)       | 差异说明     |
| ----------- | -------------- | ----------------------- | -------- |
| Platform    | CudaPlatform   | AscendPlatform          | 设备API不同  |
| Worker      | Worker         | AscendWorker            | 权重加载适配   |
| ModelRunner | ModelRunner    | AscendModelRunner       | 图模式适配    |
| Attention   | FlashAttention | AscendAttention         | 算子实现不同   |
| Memory      | CUDA Memory    | NPU Memory              | 内存管理策略不同 |
| Graph       | CUDA Graph     | ACL Graph / Turbo Graph | 图编译方式不同  |

***

## 八、实践练习

### 8.1 验证环境

```bash
# 检查 NPU 是否可用
python -c "import torch_npu; print(torch_npu.npu.is_available())"

# 检查 NPU 数量
python -c "import torch_npu; print(torch_npu.npu.device_count())"

# 查看 NPU 信息
npu-smi info
```

### 8.2 验证插件加载

```python
# test_plugin.py
import vllm

# 检查是否加载了 ascend 插件
from vllm import platform
print(f"Current platform: {platform.current_platform()}")

# 应该输出: Current platform: AscendPlatform
```

### 8.3 简单推理测试

```python
# test_inference.py
from vllm import LLM, SamplingParams

# 创建 LLM（会自动使用 Ascend 后端）
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")

# 创建采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)

# 执行推理
prompts = ["Hello, my name is", "The future of AI is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

***

## 九、疑问与待深入

* [ ] torch\_npu 与 torch.cuda 的 API 差异有哪些？

* [ ] CANN 软件栈的层次结构是什么？

* [ ] 如何调试 NPU 上的性能问题？

* [ ] 补丁系统的具体实现细节？

***

## 十、本周复盘

### 收获

1. 理解了 vLLM-Ascend 的插件注册机制（entry\_points）
2. 掌握了 Platform 抽象层的设计
3. 了解了 Worker 和 ModelRunner 的初始化流程
4. 理解了补丁系统的必要性

### 待深入

1. 深入阅读 Worker 的分布式初始化代码
2. 研究 ModelRunner 的图模式实现
3. 了解 CANN 软件栈

### 下周计划

1. Attention 与 KV Cache 核心
2. 理解 PagedAttention 在 NPU 上的实现

