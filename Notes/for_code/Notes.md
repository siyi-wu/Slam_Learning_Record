# MobileStereoNet

## 理论背景

在双目立体匹配（Stereo Matching）任务中，以 PSMNet 为代表的经典网络证明了：**拼接式 Cost Volume + 3D CNN 代价聚合** 能够取得极高的精度。因为 3D 卷积可以同时在**空间维度（H, W）**和**视差维度（D）**上进行上下文信息的正则化（Regularization）。

**痛点（Bottleneck）：**

标准的 3D 卷积极其耗时且占用显存。假设 3D 卷积核大小为 $K \times K \times K$，输入通道数为 $C_{in}$，输出通道数为 $C_{out}$。对于生成的每一个特征体素，其乘加操作（MACs）的计算复杂度为：

$$\text{Cost}_{standard} = K^3 \cdot C_{in} \cdot C_{out}$$

这种立方的计算量使得 3D CNN 根本无法在移动端（无人机、自动驾驶边缘计算节点）实时运行。MobileStereoNet (WACV 2021) 的提出就是为了打破这个瓶颈。

### 核心创新：3D 深度可分离卷积 (3D Depthwise Separable Convolution)

MobileStereoNet 的核心灵感来自于 2D 图像领域的 MobileNet。它将 3D 卷积解耦为两步操作，从而在几乎不损失精度的前提下，大幅降低参数量和计算量。

- **Step 1: 3D Depthwise Convolution (逐通道 3D 卷积)**
  - **原理：** 对输入的每一个通道独立应用一个 $K \times K \times K$ 的卷积核。此时通道之间不发生信息交换，仅在空间（H, W）和视差（D）维度聚合信息。
  - **复杂度：** $K^3 \cdot C_{in}$
- **Step 2: 3D Pointwise Convolution (逐点 3D 卷积)**
  - **原理：** 使用 $1 \times 1 \times 1$ 的卷积核。它的作用是在空间和视差维度保持不变的情况下，将不同通道的特征进行线性组合（融合通道信息）。
  - **复杂度：** $1^3 \cdot C_{in} \cdot C_{out}$

**性能对比（理论提速）：**

通过这种解耦，计算量之比为：

$$\frac{\text{Cost}_{separable}}{\text{Cost}_{standard}} = \frac{K^3 \cdot C_{in} + C_{in} \cdot C_{out}}{K^3 \cdot C_{in} \cdot C_{out}} = \frac{1}{C_{out}} + \frac{1}{K^3}$$

由于通常 $K=3$，且通道数 $C_{out}$ 较大（如 32 或 64），这个比值大约是 $\frac{1}{27}$。**这意味着理论计算量下降了近 96%**

### 网络架构流水线 (MSNet3D Pipeline)

MobileStereoNet 提供了一个极其精简的 Pipeline：

1. **2D 特征提取：** 使用 **MobileNetV2** 提取左右图像的特征（利用了倒残差结构 Inverted Residuals）。
2. **Cost Volume 构建：** 采用 Concatenation 方式构建 4D 体积。
3. **3D 代价聚合：** 使用基于 **3D深度可分离卷积** 构建的 3D Hourglass（沙漏/编码器-解码器）网络。
4. **视差计算：** 使用 Soft-Argmin 回归出连续的亚像素视差图。

------

### 4. 核心代码实现 (PyTorch)

在 PyTorch 中，当 `groups = in_channels` 时，`nn.Conv3d` 就会执行 Depthwise 卷积。

```python
import torch
import torch.nn as nn

class DepthwiseSeparableConv3d(nn.Module):
    """
    MobileStereoNet 的核心组件：3D 深度可分离卷积块
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        
        # 1. Depthwise Conv: 负责空间与视差维度的特征提取
        # 核心技巧：设置 groups = in_channels，让每个输入通道独立卷积
        self.depthwise = nn.Conv3d(
            in_channels=in_channels, 
            out_channels=in_channels, # 输出通道数必须等于输入通道数
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding, 
            groups=in_channels,       # 关键参数！
            bias=False                # 后面接BN，所以不需要bias
        )
        self.bn1 = nn.BatchNorm3d(in_channels)
        
        # 2. Pointwise Conv: 负责通道维度的特征融合
        # 使用 1x1x1 卷积，groups 恢复为 1 (默认)
        self.pointwise = nn.Conv3d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: [Batch, Channels, Disparity, Height, Width]
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out

# --- 测试与验证 ---
if __name__ == "__main__":
    # 模拟一个经过拼接的 Cost Volume
    # [Batch=2, Channels=32, Disparity=48, Height=64, Width=128]
    dummy_cost_volume = torch.randn(2, 32, 48, 64, 128)
    
    # 初始化 MobileStereoNet 的核心卷积块
    msnet_conv_block = DepthwiseSeparableConv3d(in_channels=32, out_channels=64)
    
    # 前向传播
    output_volume = msnet_conv_block(dummy_cost_volume)
    print(f"输入形状: {dummy_cost_volume.shape}")
    print(f"输出形状: {output_volume.shape}") 
    # 预期输出: torch.Size([2, 64, 48, 64, 128])
```

### 工程与调试

1. **MACs vs. Latency（理论计算量与实际延迟）：** 虽然理论上 3D Depthwise Conv 减少了 90%+ 的计算量，但在实际 GPU（如 NVIDIA RTX 系列）上，它的**显存访存密集（Memory-bound）**特性会导致其提速比例远达不到理论值。这是因为大量的轻量级算子增加了 Kernel 调度的开销（CUDA Kernel Launch Overhead）。

2. **边缘端部署优化：**

   如果要部署到 NVIDIA Jetson 或者高通 DSP 平台，请务必检查推理引擎（TensorRT / SNPE）对 `Conv3d` 的 `groups` 参数的支持程度。早期版本的推理引擎可能会把 Depthwise Conv3d 退化为极其缓慢的常规循环计算。

3. **正则化作用：** 即便是在双目视觉中加入简单的语义信息，3D 卷积依然是极其强大的平滑先验（Smoothness Prior）。它能有效地解决弱纹理区域（Textureless regions，比如白墙）、反光区域的视差估计问题，这是传统 SLAM 极难处理的。