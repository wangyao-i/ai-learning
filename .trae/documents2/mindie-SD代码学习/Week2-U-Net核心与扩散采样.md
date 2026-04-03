# Week2 - U-Net核心与扩散采样

## 学习目标

深入理解U-Net的核心结构和扩散采样算法，掌握Stable Diffusion的生成原理。

---

## 1. U-Net核心结构

### 1.1 U-Net基本架构

U-Net是Stable Diffusion的核心组件，负责从噪声中生成图像。其基本结构包括：

- **编码器（Encoder）**：逐步下采样，提取特征
- **瓶颈（Bottleneck）**：最底层，包含最抽象的特征
- **解码器（Decoder）**：逐步上采样，恢复图像
- **跳跃连接（Skip Connections）**：保留细节信息

### 1.2 U-Net代码实现

```python
class UNet(nn.Module):
    """U-Net模型"""
    
    def __init__(self, in_channels=4, out_channels=4, 
                 time_emb_dim=1280, context_dim=1024):
        super().__init__()
        
        # 时间嵌入
        self.time_embed = TimeEmbedding(time_emb_dim)
        
        # 编码器
        self.encoder = Encoder(in_channels, time_emb_dim, context_dim)
        
        # 瓶颈
        self.bottleneck = Bottleneck(time_emb_dim, context_dim)
        
        # 解码器
        self.decoder = Decoder(out_channels, time_emb_dim, context_dim)
    
    def forward(self, x, timesteps, context):
        # 时间嵌入
        t_emb = self.time_embed(timesteps)
        
        # 编码
        encoder_outputs = self.encoder(x, t_emb, context)
        
        # 瓶颈
        bottleneck_output = self.bottleneck(
            encoder_outputs[-1], t_emb, context
        )
        
        # 解码
        output = self.decoder(bottleneck_output, encoder_outputs, t_emb, context)
        
        return output
```

### 1.3 融合U-Net实现

```python
class FusedUNet(UNet):
    """融合U-Net实现"""
    
    def __init__(self, in_channels=4, out_channels=4,
                 time_emb_dim=1280, context_dim=1024):
        super().__init__(in_channels, out_channels, time_emb_dim, context_dim)
        # 使用融合算子
        self.fused_encoder = FusedEncoder(in_channels, time_emb_dim, context_dim)
        self.fused_bottleneck = FusedBottleneck(time_emb_dim, context_dim)
        self.fused_decoder = FusedDecoder(out_channels, time_emb_dim, context_dim)
    
    def forward(self, x, timesteps, context):
        # 使用融合算子进行计算
        t_emb = self.time_embed(timesteps)
        encoder_outputs = self.fused_encoder(x, t_emb, context)
        bottleneck_output = self.fused_bottleneck(
            encoder_outputs[-1], t_emb, context
        )
        output = self.fused_decoder(bottleneck_output, encoder_outputs, t_emb, context)
        return output
```

---

## 2. 扩散采样算法

### 2.1 扩散模型原理

扩散模型通过逐步去噪的过程生成图像：

1. **前向过程**：逐步向图像添加高斯噪声
2. **反向过程**：逐步从噪声中恢复图像
3. **采样过程**：从随机噪声开始，逐步去噪生成图像

### 2.2 Euler采样器

```python
class EulerSampler:
    """Euler采样器"""
    
    def __init__(self, num_steps=50):
        self.num_steps = num_steps
    
    def sample(self, model, latents, timesteps, context):
        """Euler采样"""
        for i, t in enumerate(timesteps):
            # 预测噪声
            noise_pred = model(latents, t, context)
            
            # 计算去噪步骤
            alpha_prod_t = self._get_alpha_prod(t)
            alpha_prod_t_prev = self._get_alpha_prod(timesteps[i+1] if i+1 < len(timesteps) else 0)
            beta_prod_t = 1 - alpha_prod_t
            
            # Euler更新
            latents = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
            latents = alpha_prod_t_prev ** 0.5 * latents + (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
        
        return latents
```

### 2.3 DPM++采样器

```python
class DPMPPSampler:
    """DPM++采样器"""
    
    def __init__(self, num_steps=50):
        self.num_steps = num_steps
    
    def sample(self, model, latents, timesteps, context):
        """DPM++采样"""
        for i, t in enumerate(timesteps):
            # 预测噪声
            noise_pred = model(latents, t, context)
            
            # DPM++更新
            # ... 具体实现
            
        return latents
```

### 2.4 量化采样器

```python
class QuantizedSampler:
    """量化采样器"""
    
    def __init__(self, sampler, quantization_bits=8):
        self.sampler = sampler
        self.quantization_bits = quantization_bits
        self.quantizer = Quantizer(bits=quantization_bits)
    
    def sample(self, model, latents, timesteps, context):
        # 量化模型
        quantized_model = self.quantizer.quantize(model)
        
        # 采样过程
        for i, t in enumerate(timesteps):
            # 量化输入
            quantized_latents = self.quantizer.quantize_tensor(latents)
            quantized_context = self.quantizer.quantize_tensor(context)
            
            # 前向传播
            noise_pred = quantized_model(quantized_latents, t, quantized_context)
            
            # 反量化输出
            noise_pred = self.quantizer.dequantize_tensor(noise_pred)
            
            # 更新潜在表示
            latents = self.sampler.step(latents, noise_pred, t)
        
        return latents
```

---

## 3. 代码阅读重点

### 3.1 backend/unet/unet.py

**核心类**：
- `UNet`：基础U-Net实现
- `FusedUNet`：融合U-Net实现
- `UNetBlock`：U-Net块

**关键方法**：
- `__init__()`：初始化U-Net
- `forward()`：前向传播
- `get_output()`：获取输出

### 3.2 backend/sampler/euler.py

**核心类**：
- `EulerSampler`：Euler采样器
- `EulerAncestralSampler`：Euler祖先采样器

**关键方法**：
- `sample()`：采样
- `step()`：单步更新
- `_get_alpha_prod()`：获取alpha参数

### 3.3 backend/sampler/quantized_sampler.py

**核心类**：
- `QuantizedSampler`：量化采样器
- `Quantizer`：量化器

**关键方法**：
- `sample()`：采样
- `quantize()`：量化
- `dequantize()`：反量化

---

## 4. 学习笔记

### 4.1 U-Net工作原理

U-Net的工作原理：
1. **编码**：逐步下采样，提取多尺度特征
2. **瓶颈**：在最底层处理最抽象的特征
3. **解码**：逐步上采样，恢复图像细节
4. **跳跃连接**：将编码器的特征直接传递到解码器

### 4.2 扩散采样过程

扩散采样的过程：
1. **初始化**：从随机噪声开始
2. **迭代去噪**：逐步去除噪声
3. **条件引导**：使用文本条件引导生成
4. **图像生成**：最终生成目标图像

### 4.3 量化采样的优势

量化采样的优势：
1. **性能提升**：计算速度提升30-50%
2. **内存优化**：显存占用降低25-40%
3. **质量保证**：精度损失控制在1%以内

---

## 5. 性能对比

| 指标 | 标准U-Net | 融合U-Net | 提升比例 |
|------|----------|-----------|----------|
| 单块前向时间 | 1.2ms | 0.4ms | 200% |
| Kernel调用次数 | 5 | 1 | 80%减少 |
| 内存访问次数 | 10 | 3 | 70%减少 |

| 指标 | 标准采样 | 量化采样 | 提升比例 |
|------|----------|----------|----------|
| 采样速度 | 2.5s | 1.2s | 108% |
| 显存占用 | 10GB | 6GB | -40% |
| FID分数 | 12.5 | 12.8 | +0.3 |

---

## 6. 自测问题

1. U-Net的基本结构是什么？
2. 扩散采样的基本原理是什么？
3. Euler采样器和DPM++采样器的区别是什么？
4. 量化采样如何提高性能？

---

## 7. 下一步学习

完成Week2学习后，将进入Week3，学习VAE与图像编码解码。