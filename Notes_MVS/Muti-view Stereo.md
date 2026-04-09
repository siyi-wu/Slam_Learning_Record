# MVSNet Depth Inference for  Unstructured Multi-view Stereo

2018

## 关键词

- **Multi-view Stereo (MVS)**：多视图立体视觉，即通过多张不同角度的照片恢复场景的 3D 结构。
- **Cost Volume (代价体)**：一种 3D 数据结构，用于存储不同深度假设下的匹配误差或相似度。
- **Differentiable Homography (可微单应性)**：连接 2D 特征与 3D 空间的数学桥梁，允许梯度反向传播。
- **Depth Map (深度图)**：输出结果，图像中每个像素到相机的距离。
- **End-to-End Learning (端到端学习)**：直接从输入图像到输出深度图的训练过程。

## 解决的问题

**为什么要通过学习来做 MVS？** 在 MVSNet 之前，传统的 MVS 方法（如 COLMAP 中的 PatchMatch）依赖于手工设计的特征匹配（如 NCC 或 SAD）。

- **痛点 1：弱纹理与反光。** 传统方法在白墙、玻璃或重复纹理区域容易匹配失败。
- **痛点 2：计算复杂度。** 之前的深度学习方法（如 SurfaceNet）试图将整个空间体素化（Voxelization），这极其消耗显存，限制了重建的分辨率。

**MVSNet 的目标：** 设计一个网络，既能利用深度学习提取强大的特征（解决弱纹理问题），又能高效地处理非结构化（任意视角、任意数量）的图片输入，生成高精度的深度图。

## 核心思想

MVSNet 的核心在于**“将几何约束融入深度学习网络”**。

它没有像“黑盒”一样直接预测深度，而是模拟了传统几何中的 **Plane Sweep（平面扫描）** 算法：

1. **特征提取**：用 CNN 提取图像的深度特征。
2. **构建视锥代价体 (Camera Frustum Cost Volume)**：不像 SurfaceNet 那样在世界坐标系建方块，MVSNet 在**参考相机的视锥体**内构建代价体。
   - *直观理解*：想象从参考相机出发，在前方切出很多平行的“深度平面”。
3. **单应性变换 (Homography Warping)**：把其他视角的特征图，根据假设的深度，通过投影变换“扭”到参考相机的视角下。
4. **方差聚合**：看这些扭过来的特征图和参考图像长得像不像（计算方差）。

## 方法

### 特征提取

对一张参考图像 (Reference Image) 和 $N$ 张源图像 (Source Images) 使用 2D CNN 提取特征。

- **输出**：$N+1$ 个 32 通道的特征图（Feature Maps）。

### 可微单应性变换 (Differentiable Homography)

这是几何与深度学习结合的关键。

对于参考图像的每一个像素 $(u, v)$ 和假设的深度 $d$，我们想知道它在源图像 $i$ 中的位置 $(u_i, v_i)$。

公式如下：

$\mathbf{x}_i \sim \mathbf{K}_i (\mathbf{R}_i \mathbf{R}_{ref}^{-1} (\mathbf{K}_{ref}^{-1} \mathbf{x}_{ref} \cdot d) + \mathbf{t}_{i, ref})$

或者用单应性矩阵描述：

$\mathbf{H}_i(d) = \mathbf{K}_i \mathbf{R}_i (\mathbf{I} - \frac{(\mathbf{C}_{ref} - \mathbf{C}_i) \mathbf{n}^T}{d}) \mathbf{R}_{ref}^{-1} \mathbf{K}_{ref}^{-1}$

- **直觉解释**：这实际上是在问：“如果参考图中的这个点深度是 $d$，那它投射到源图片上应该在哪？”然后我们把源图片的特征在这个位置取出来（通过双线性插值）。

### 代价体构建

现在我们有了参考特征 $F_{ref}$ 和多个扭曲过来的源特征 $F_{warp, i}$。如何把它们合并成一个体？

MVSNet 提出了基于 **方差 (Variance)** 的度量：

$C = \frac{\sum_{i=1}^{N} (F_i - \bar{F})^2}{N}$

- **为什么用方差？**
  - **能够处理任意数量的输入**：无论输入 2 张还是 10 张图，方差输出的维度不变。
  - **光照鲁棒性**：它比较的是特征的一致性，而非绝对强度。
  - **几何意义**：方差越小，说明大家在这一点“意见越一致”，即该深度假设越可能是真的。

### 深度估计

现在的 Cost Volume 是一个 4D 张量（长、宽、深度假设层数、特征通道）。

1. **3D CNN 正则化**：使用类似 U-Net 的 3D 网络对 Cost Volume 进行平滑，消除噪声，引入上下文信息。

2. **Soft Argmin (期望深度)**：

   通常我们会选概率最大的那个深度（Argmax），但这不可导且是离散的。MVSNet 计算深度的**数学期望**：

   $D = \sum_{d=d_{min}}^{d_{max}} d \cdot P(d)$

   其中 $P(d)$ 是通过 Softmax 得到的概率。这样做使得输出是连续的（Sub-pixel accuracy），且完全可微。

## 创新点

**基于方差的代价度量 (Variance-based Cost Metric)**：

- 解决了多视角输入数量不固定的问题。
- 将多视图匹配问题转化为一个通用的特征聚合问题。

**在参考相机视锥体内构建 Cost Volume**：

- 相比于世界坐标系的体素网格（SurfaceNet），这种方法更高效，且能聚焦于参考视角可见的部分。

**端到端的深度回归**：

- 引入 Soft Argmin 操作，使得网络可以直接回归出连续的深度值，而非简单的分类任务。

## 实验与结果

**数据集**：DTU 数据集（室内物体，光照变化大，有精确 Ground Truth）。

**指标**：

- **Accuracy (准确度)**：重建点云与真实点云的距离。
- **Completeness (完整度)**：真实点云被重建出来的比例。

**结果**：

- 在 DTU 数据集上大幅超越了传统方法（如 COLMAP, Tola）和之前的深度学习方法（SurfaceNet）。
- **泛化能力**：直接将在 DTU 上训练的模型拿到 Tanks and Temples（室外大场景）数据集上测试，依然表现出色，证明网络学到了几何原理，而不仅仅是死记硬背数据。

## 结论

MVSNet 证明了深度学习在多视图立体视觉中的巨大潜力。它不仅仅是用 CNN 提取特征，而是**将“极线几何”和“平面扫描”的物理约束显式地编码进了网络结构中**。这使得模型既具有深度学习的特征鲁棒性，又具有几何方法的数学严谨性。

## 可学习知识点

1. **Feature Warping (特征扭曲)**：在深度学习中，如何利用已知几何参数（如相机位姿）对特征图进行空间变换（Spatial Transformer Network 的思想）。
2. **Cost Volume (代价体)**：理解这是立体匹配（Stereo Matching）和 MVS 的核心数据结构。它是 $W \times H \times D$ 的网格，每个格子存储“这个像素在这个深度下的匹配代价”。
3. **Regression vs. Classification**：在预测连续值（如深度、坐标）时，使用 Soft Argmin（回归期望）通常比直接分类（Argmax）效果更好，精度更高。

## 伪代码

### 总流程

```python
class MVSNet(nn.Module):
    def __init__(self, refine=False):
        super(MVSNet, self).__init__()
        self.feature_net = FeatureNet() # 2D CNN
        self.cost_reg_net = CostRegNet() # 3D CNN

    def forward(self, imgs, proj_matrices, depth_values):
        # imgs: [Batch, N_views, 3, H, W]
        # proj_matrices: [Batch, N_views, 4, 4] (投影矩阵)
        # depth_values: [Batch, D] (比如 192 个假设深度)

        # 1. 特征提取 (Feature Extraction)
        # 既然是 Siamese Network，我们就把 Batch 和 View 维度合并一起处理
        imgs = torch.unbind(imgs, 1) # 把 N 个视角拆开
        ref_feature = self.feature_net(imgs[0]) # 参考图特征
        src_features = [self.feature_net(img) for img in imgs[1:]] # 源图特征

        # 2. & 3. 构建代价体 (Homography Warping + Cost Volume)
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, n_depth, 1, 1) # 参考图直接复制 D 份
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2

        for src_feat, proj_mat in zip(src_features, proj_matrices[1:]):
            # === 核心几何变换 ===
            # 将源图特征 warp 到参考图的视角和每一个深度平面上
            warped_volume = homo_warping(src_feat, proj_matrices[0], proj_mat, depth_values)
            
            # 累加用于计算方差
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2
        
        # 计算方差 (Variance) 作为 Cost Volume
        # Var = E[X^2] - (E[X])^2
        cost_volume = volume_sq_sum / n_views - (volume_sum / n_views) ** 2

        # 4. 代价体正则化 (3D CNN)
        # 输入: [B, C, D, H/4, W/4] -> 输出: [B, 1, D, H/4, W/4]
        prob_volume = self.cost_reg_net(cost_volume) 
        prob_volume = F.softmax(prob_volume, dim=1) # 沿着 D 维度做 Softmax

        # 5. 深度回归 (Depth Regression)
        depth = depth_regression(prob_volume, depth_values)
        
        return depth
```

### 可微单应性变换

```python
def homo_warping(src_feature, ref_proj, src_proj, depth_values):
    # src_feature: [Batch, C, H, W]
    # ref_proj, src_proj: [Batch, 4, 4] (包含了 K, R, t)
    # depth_values: [Batch, D]

    batch, channels, height, width = src_feature.shape
    num_depth = depth_values.shape[1]

    # 步骤 A: 计算单应性矩阵 (Homography)
    # H(d) = K_src * (R * K_ref^-1 + t/d * n^T) ... 
    # 为了简化计算，通常直接对 4x4 投影矩阵操作
    # 这里省略繁琐的矩阵乘法代码，直接展示思路：
    # 这一步会算出一个变换矩阵，把 Ref 的像素坐标映射到 Src 的像素坐标
    
    # 步骤 B: 构建参考图的像素网格 (Meshgrid)
    y, x = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
    # 扩展维度以匹配 Batch 和 Depth
    # ... (坐标归一化等操作)

    # 步骤 C: 核心变换 logic
    # 假设我们得到了旋转后的坐标 rot_coords 和平移量 trans_vec
    
    # 公式: uv_src = K_src * (Rot * inv(K_ref) * uv_ref * depth + trans)
    # 注意：这里实际上是用矩阵乘法一次性完成所有点的变换
    
    # 最终我们需要得到 grid: [Batch, D, H, W, 2]
    # 最后一维的 '2' 存的是 (x, y) 采样坐标，范围是 [-1, 1]
    
    # 步骤 D: 可微采样 (Differentiable Sampling)
    # 这是深度学习能做 MVS 的关键函数！
    # 它根据 grid 里的坐标，在 src_feature 上进行双线性插值
    warped_feature = F.grid_sample(
        src_feature, 
        grid.view(batch, num_depth * height, width, 2), 
        mode='bilinear', 
        padding_mode='zeros'
    )

    # Reshape 回 5D 张量: [Batch, C, D, H, W]
    return warped_feature.view(batch, channels, num_depth, height, width)
```

### 3D CNN

```python
class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        # 这是一个类似 3D U-Net 的结构
        
        # 下采样 (Downsampling)
        self.conv0 = ConvBnReLU3D(32, 8)
        self.conv1 = ConvBnReLU3D(8, 16, stride=2) 
        self.conv2 = ConvBnReLU3D(16, 32, stride=2) # 空间尺寸变小，深度 D 没变或也变小

        # 上采样 (Upsampling) + Skip Connection
        self.conv3 = ConvBnReLU3D(32, 16, stride=1) # 这里的反卷积把尺寸变大
        # ... (省略中间层)

        # 输出层：把特征通道压扁成 1
        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        # x: [Batch, 32, D, H, W] (32是特征提取后的通道数)
        out = self.conv0(x)
        out = self.conv1(out)
        # ... U-Net 流程 ...
        out = self.prob(out) 
        return out.squeeze(1) # [Batch, D, H, W]
```

### 深度回归(Soft Argmin)

```python
def depth_regression(p, depth_values):
    # p: 概率体 [Batch, D, H, W] (已经做过 Softmax)
    # depth_values: [Batch, D] (假设的深度平面值，如 400mm, 405mm, ...)
    
    depth = torch.sum(p * depth_values.view(p.shape[0], p.shape[1], 1, 1), dim=1)
    
    return depth # 输出 [Batch, H, W] 的深度图
```

# Cascade Cost Volume for High-Resolution Multi-View Stereo  and Stereo Matching

2020

## 关键词

- **Multi-View Stereo (MVS, 多视图立体视觉)**
- **3D Cost Volume (三维代价体)**
- **Cascade / Coarse-to-Fine Strategy (级联/由粗到精策略)**
- **Hypothesis Pruning (假设空间剪枝)**
- **Residual Depth Regression (残差深度回归)**

## 解决的问题

在 Deep MVS（如 MVSNet ）和 Stereo Matching（如 PSMNet ）中，核心操作是构建 **3D Cost Volume**。

- **计算复杂性：** Cost Volume 的尺寸为 $W \times H \times D \times F$（宽×高×深度假设数×特征通道）。其显存占用和计算量随着体积分辨率呈**立方级 (Cubically)** 增长 。
- **工程瓶颈：** 受限于 GPU 显存（例如 Tesla P100 16GB），标准 MVSNet 只能处理降采样后的 Cost Volume（如原图的 1/4 或 1/8）。
- **后果：** 低分辨率 Cost Volume 导致无法恢复高频几何细节；若要提高分辨率，则必须大幅减少深度假设平面数量 $D$，导致深度量化误差增大，出现分层伪影（Stratification artifacts）。

## 核心思想

论文提出了一种 **Cascade Cost Volume（级联代价体）** 结构，核心在于**“自适应假设采样” (Adaptive Hypothesis Sampling)**。

与其构建一个覆盖全深度范围的高分辨率 Cost Volume，不如将其分解为多个阶段（Stages）：

- **Stage 1 (Coarse):** 在低分辨率特征图上，覆盖**完整**的深度范围，进行稀疏采样，通过 3D CNN 回归出一个粗糙的初始深度图。
- **Stage k (Fine):** 基于上一阶段的预测深度，**动态缩小**深度假设范围（只在预测值附近搜索），并**加密**采样间隔，同时使用更高分辨率的特征图进行细化 。

**本质：** 这是一种在 3D 空间中的**不确定性缩减（Uncertainty Reduction）**过程，将计算资源集中在物体表面附近的“有效区域”。

## 方法

该方法是一个通用模块，可嵌入 MVSNet 或 GwcNet 等骨干网络。

### 特征金字塔 (Feature Pyramid)

利用 FPN 提取 $N$ 个尺度的特征图（例如 1/16, 1/4, 1 尺度），对应级联的 $N$ 个阶段 。

### 级联代价体构建 (Cascade Formulation)

对于第 $k$ 个阶段，深度假设平面的生成遵循以下规则：

- **深度范围 (Hypothesis Range) $R_k$:**
  - $R_1$ 覆盖场景的全局深度范围（基于稀疏点云确定）。
  - 后续阶段 $R_{k+1} = R_k \cdot w_k$，其中 $w_k < 1$ 是缩减因子。这意味着搜索范围逐级收窄 。
- **深度间隔 (Interval) $I_k$:**
  - 后续阶段 $I_{k+1} = I_k \cdot p_k$，其中 $p_k < 1$。这意味着深度分辨率逐级提高 。
- **平面数量 (Plane Number) $D_k$:**
  - 由 $D_k=R_k / I_k$ 决定。由于$R_k$大幅减小，即使$I_k$变小，所需的平面数$D_k$依然可以保持很小（如 48, 32, 8），从而节省显存。

### 可微单应性变换 (Differentiable Warping)

在 MVS 任务中，第 $k+1$ 阶段的单应性变换 $H_i$ 不再基于全局深度，而是基于**上一阶段预测深度 $d_k^m$ + 当前阶段残差 $\Delta_{k+1}^m$** ：

$T_i(d_k^m + \Delta_{k+1}^m) = K_i \cdot R_i \cdot (I - \frac{(t_1 - t_i) \cdot n_1^T}{d_k^m + \Delta_{k+1}^m}) \cdot R_1^T \cdot K_1^{-1}$

这意味着网络学习的是**残差深度 (Residual Depth)**，这大大降低了学习难度。

### 损失函数 (Loss Function)

采用多尺度监督，总 Loss 为各阶段 Loss 的加权和：

$Loss = \sum_{k=1}^{N} \lambda^k \cdot L^k$ 。

## 创新点

- **空间-深度解耦：** 首次在 Deep MVS 中引入级联结构，打破了 Cost Volume 分辨率与显存占用的强耦合关系 。
- **残差学习范式：** 证明了通过学习深度残差（而非直接回归绝对深度）在由粗到精框架下的有效性 。
- **SOTA 性能与效率：** 在保持高精度的同时，显著降低了推理时间和显存消耗，使得在普通 GPU 上输出原图分辨率的深度图成为可能 。

## 实验与结果

**DTU Benchmark (MVS):**

- **精度：** 排名第 1（论文发表时）。总体误差 (Overall Error) 从 MVSNet 的 0.551mm 降至 0.355mm 。
- **效率：** 显存减少 **50.6%** (10823MB $\to$ 5345MB)，运行时间减少 **59.3%** 。

**Tanks and Temples:** 在 Intermediate 集合上排名第 1，证明了其在非受控环境下的泛化能力 。

**Stereo Matching (Scene Flow / KITTI):** 将 Cascade 模块应用于 GwcNet，EPE (End-Point-Error) 降低约 15.2%，显存降低 36.9% 。

## 结论

Cascade Cost Volume 通过将单一的高昂计算任务分解为多个轻量级、递进式的子任务，利用**由粗到精**的预测来剪枝无效的深度假设空间。这种方法不仅大幅提升了计算效率，还通过在高分辨率特征图上进行精细匹配，显著提高了重建的几何细节 。

## 可学习内容

**Plane-Sweep Stereo (平面扫描立体视觉) 的现代演进：**

- 传统 Plane-Sweep 算法需要遍历所有深度平面。本文展示了如何利用深度学习的**先验知识 (Prior)** 来“智能”地选择扫描平面，这与 SLAM 中利用上一帧位姿和地图点投影来限制搜索区域（Search Region）的思想异曲同工。

**3D 正则化 (3D Regularization) 的代价：**

- 理解为什么 3D CNN (如 3D U-Net) 处理 Cost Volume 效果好但昂贵。它利用了上下文信息（Context）解决了弱纹理区域的匹配歧义，但引入了维度灾难。Cascade 方法是缓解这一问题的标准解法。

**不确定性传播 (Uncertainty Propagation)：**

- 第一阶段的深度间隔 $I_1$ 必须足够大以覆盖误差，而后续阶段的搜索范围 $R_k$ 取决于上一阶段预测的**置信度**。虽然论文中使用固定的缩减因子，但在更高级的 SLAM 系统中，这通常对应于协方差矩阵（Covariance Matrix）的收敛过程。

**残差网络 (ResNet) 思想在几何中的应用：**

- 通常我们认为 ResNet 用于图像分类，但在这里，**学习 $\Delta d$ (深度修正量)** 比直接学习 $d$ 更容易收敛。这对于设计任何迭代优化的几何网络（如光流网络 Raft）都是核心直觉。

# IterMVS Iterative Probability Estimation for Efficient Multi-View Stereo

2021

## 关键词

- **Multi-View Stereo (MVS)**：多视图立体视觉
- **GRU (Gated Recurrent Unit)**：门控循环单元
- **Iterative Refinement**：迭代优化
- **Probability Distribution**：概率分布
- **Efficiency**：高效性（低显存、低延迟）
- **Generalization**：泛化能力

## 解决的问题

这篇论文主要致力于解决现有基于深度学习的 MVS 方法中存在的两个核心矛盾 ：

- **资源消耗过大**：以 MVSNet  为代表的方法需要构建巨大的 3D 代价体（Cost Volume）并使用 3D CNN 进行正则化，导致**显存占用极高**，难以处理高分辨率图像。
- **由粗到细策略的缺陷**：为了降低显存，后续方法（如 Cascade-MVSNet）采用了由粗到细（Coarse-to-Fine）的级联策略。虽然节省了显存，但**如果粗糙层级的深度估计错误，这种错误会传播到精细层级且无法恢复**（例如细小的物体在低分辨率下消失了） 。
- **泛化能力与效率的平衡**：PatchmatchNet  虽然极快且省显存，但在新场景（如 Tanks & Temples）上的泛化能力较弱。

**一句话总结**：IterMVS 旨在实现 PatchmatchNet 级别的**高效率**，同时保持甚至超越级联方法的**重建质量和泛化能力**。

## 核心思想

IterMVS 的核心灵感来源于光流领域的 **RAFT** 。它不再一次性计算所有深度的概率，也不通过缩放图像尺寸来逐级细化，而是：

- **在固定分辨率下迭代**：始终在 1/4 分辨率下工作，避免了粗糙层级丢失细节的问题。
- **隐状态编码概率**：利用 GRU 的**隐状态 (Hidden State)** 来记忆和编码像素的深度**概率分布** 。
- **动态采样**：每次迭代根据当前的深度估计，在附近重新采样深度假设（Depth Hypotheses），计算匹配代价，然后更新隐状态。

这种方式模拟了传统优化算法的梯度下降过程，但所有步骤都是可微分且由神经网络学习得到的。

## 方法

**多尺度特征提取 (Feature Extractor)**：

- 使用 FPN 提取多尺度特征，但这些特征主要用于计算匹配代价，而不是用于构建级联的代价体结构 。

**基于 GRU 的概率估计器 (GRU-based Probability Estimator)** ：

- **初始化**：首先在全深度范围内进行一次全局搜索，得到初始的隐状态 $h_0$。
- **迭代更新 (Iterative Update)**：
  - **输入**：当前的隐状态、当前估计的深度图、通过单应性变换（Homography Warping）计算得到的匹配代价（Matching Cost）。
  - **处理**：GRU 单元接收这些输入，更新隐状态。
  - **输出**：从更新后的隐状态中解码出新的深度图和置信度图。
- **深度采样**：随着迭代进行，搜索范围（Search Range）逐渐缩小，模拟了“由粗到细”的搜索，但这是在**深度域**上进行的，而不是在图像分辨率上进行的 。

**深度图预测 (Depth Prediction Strategy)** ：

- 这是论文的一个微小但精妙的设计。
- **混合模式**：结合了**分类（Classification）**和**回归（Regression）**。
- 首先通过 `argmax` 找到概率最高的深度索引（分类，抗干扰能力强）。
- 然后在该索引附近进行 `soft-argmax`（回归，获取亚像素精度）。这避免了直接使用 `soft-argmax` 处理多峰分布时产生的平均值偏差问题 。

**空间上采样 (Spatial Upsampling)** ：

- 使用类似 RAFT 的 Convex Upsampling，通过卷积网络预测权重，将 1/4 分辨率的深度图加权插值回原分辨率，这比双线性插值更能保持边缘锐利 。

## 创新点

- **隐式概率分布编码**：提出使用 RNN (GRU) 的隐状态来编码像素级的深度概率分布，而不是显式地存储一个巨大的 Probability Volume 。这极大地节省了显存。
- **混合深度预测机制**：解决了多模态分布（Multimodal Distribution，即一个像素可能对应多个深度峰值，常见于物体边缘或纹理重复区域）下的深度估计偏差问题。分类确定峰值位置，回归确定精确值 。
- **高效的迭代架构**：证明了在 MVS 中，固定分辨率下的迭代优化（Iterative Refinement）比图像金字塔式的级联结构（Coarse-to-Fine）具有更好的泛化性和鲁棒性。

## 实验与结果

**数据集**：DTU, Tanks & Temples, ETH3D。

**效率对比**：

- 显存占用和运行时间与图像分辨率呈线性关系，远优于 MVSNet 的立方级增长 。
- 比 PatchmatchNet 慢一点点，但比 Cascade-MVSNet 快且省显存 。

**性能对比**：

- **DTU**：性能与 Cascade-MVSNet 持平，优于 PatchmatchNet 。
- **Tanks & Temples / ETH3D**：这是重点。IterMVS 展现了极强的**泛化能力**，在这些从没见过的真实场景数据集上，性能显著优于 PatchmatchNet，甚至击败了许多更复杂的模型 。

**可视化**：能够清晰地看到随着迭代次数增加，概率分布从杂乱变得尖锐（确信） 。

## 结论

IterMVS 提出了一种新的数据驱动 MVS 范式。通过迭代式地估计和细化像素深度概率，它成功打破了 MVS 重建中“高质量”与“高效率”不可兼得的魔咒。它证明了**时序上的迭代（RNN）可以替代空间上的堆叠（3D CNN）**，在保持极低资源消耗的同时，获得了最先进的泛化性能 。

## 可学习知识点

这篇论文非常适合用来串联 MVS 的关键概念：

1. **Cost Volume 的本质**：理解为什么 MVSNet 需要构建 Cost Volume（为了把多图匹配问题变成一个分类/回归问题），以及为什么它那么占显存。
2. **Soft-argmax vs. Hard-argmax**：
   - *Hard-argmax (分类)*：取概率最大的那个深度。优点：鲁棒，不受噪声峰值影响。缺点：由离散采样决定，精度低（阶梯状）。
   - *Soft-argmax (回归/期望)*：对深度值求期望。优点：可微，亚像素精度。缺点：如果概率分布有两个峰（例如物体边缘），求期望会得到两个峰中间的错误深度。
   - *IterMVS 的结合*：学习如何取长补短。
3. **RNN 在几何中的应用**：通常我们认为 RNN 处理语音或文本。但在几何视觉中（如 RAFT, IterMVS），RNN 被视为一个**优化器（Optimizer）**。它一步步地“看”当前的误差，然后“推”着深度值往正确的方向走。
4. **Epipolar Geometry (对极几何) 的应用**：论文中的 `Differentiable Warping`  步骤，本质上就是利用单应性矩阵（Homography）和对极约束，将源图像的特征投影到参考图像上。这是 MVS 的物理基础。

# MVS2D Efficient Multi-view Stereo via Attention-Driven 2D Convolutions

2021

## 关键词

- **Multi-view Stereo (MVS)**：多视图立体视觉
- **Epipolar Attention**：极线注意力机制
- **2D Convolution**：2D 卷积
- **Efficiency**：高效性/实时性
- **Depth Estimation**：深度估计

## 解决的问题

**背景：** 传统的基于深度学习的 MVS 方法（如 MVSNet, DPSNet）虽然精度高，但通常采用“构建代价体 (Cost Volume) + 3D 卷积正则化”的范式 。 **痛点：**

- **计算量大，速度慢**：3D 卷积的计算复杂度非常高，导致推理速度慢，难以满足实时应用需求 。
- **内存消耗大**：3D 代价体占用的显存随深度假设数量线性增加，限制了高分辨率重建 。
- **单视图方法的局限**：纯单视图（Single-view）深度估计虽然快，但缺乏多视图几何约束，存在尺度模糊且精度较低 。

**核心目标：** 如何在保持单视图网络（2D CNN）的高效率的同时，引入多视图的几何约束以达到 MVS 的高精度？

## 核心思想

MVS2D 的核心思想是**“将多视图几何约束通过注意力机制融入到单视图 2D 网络中”**。

它不再显式地构建一个巨大的 3D 代价体，而是利用**极线几何（Epipolar Geometry）**作为引导。对于参考图像中的每一个像素，网络只在其对应的“极线”上搜索匹配点，并通过注意力机制（Attention）聚合这些匹配特征，增强原始图像的特征表示 。

## 方法

### 网络架构

- **骨干网络**：采用标准的 2D UNet 架构（Encoder-Decoder），类似于单视图深度估计网络 。
- **纯 2D 卷积**：整个网络只包含 2D 卷积操作，完全去除了昂贵的 3D 卷积 。

### 极线注意力模块 (Epipolar Attention Module)

这是论文的核心创新，插入在 UNet 的特定层中。其工作流程如下：

1. **几何投影**：给定源图像 $I_0$ 上的像素 $p_0$ 和深度假设范围，根据相机位姿 $T_i$ 和内参 $\mathcal{K}$，将其投影到参考图像 $I_i$ 的极线上，采样 $K$ 个点 $p_i^k$ 。
   - 公式：$p_i(d_0) = R_i p_0(d_0) + t_i$ 。
2. **特征提取与匹配**：提取源像素和参考像素的特征，计算匹配分数（相似度）。
   - 利用点积注意力机制（Scaled-dot product attention）计算权重 $w_{ik}$ 。
3. **特征聚合**：根据计算出的注意力权重，对参考图像沿极线的特征进行加权求和，并融合回源图像的特征图中 。
   - 直观理解：如果极线上的某个点与源像素匹配度高（权重于大），则该点的特征被强烈激活并补充给源像素。

### 鲁棒性设计 (Robustness)

- **针对位姿噪声**：当相机位姿不准确时，对应的像素可能不在极线上。MVS2D 采用**多尺度（Multi-scale）**注意力策略。在网络的低分辨率层（高层特征）应用注意力模块，因为在低分辨率下，像素的感受野更大，能够容忍更大的极线偏差 。

## 创新点

- **去 3D 化**：证明了不需要构建显式的 3D 代价体和使用 3D 卷积，仅靠 2D 卷积也能实现 SOTA 级别的 MVS 效果 。
- **几何引导的注意力机制**：提出了 Epipolar Attention Module，将几何约束（极线搜索）与深度学习中的注意力机制完美结合，实现了稀疏且高效的特征聚合 。
- **单/多视图无缝融合**：架构本质上是一个增强版的单视图网络，能够灵活地在纯单视图模式和多视图增强模式之间切换，且计算开销增加很少 。

## 实验与结果

**速度 (Efficiency)**：

- 比 MVSNet 快 **10倍** 。
- 比 PatchmatchNet（当时最快的之一）快 **2倍** 。
- 在 ScanNet 数据集上，推理速度达到 **42.9 FPS** 。

**精度 (Accuracy)**：

- **ScanNet**：AbsRel 误差从 MVSNet 的 0.094 降低到 **0.059**，优于 NAS 和 DPSNet 。
- **DTU**：在点云重建质量上，与 PatchmatchNet 相当甚至更优（特别是完整性），且速度更快 。
- **鲁棒性**：在输入位姿由噪声干扰时，多尺度策略（Ours-robust）表现出显著的稳定性 。

**消融实验**：

- 证明了端到端学习深度编码（Learned codes）比固定的编码方式（如 One-hot 或 Cosine）效果更好 。

## 结论

MVS2D 提出了一种简单而高效的 MVS 方法。通过将单视图特征与通过极线注意力机制提取的多视图线索相结合，该方法在保持极高推理速度（2D 卷积）的同时，实现了最先进的深度估计精度。这为未来在资源受限设备（如移动端 AR/VR）上进行高精度 3D 重建指明了方向 。

## 可学习知识点

1. **极线约束的本质 (Epipolar Constraint)**：
   - **知识点**：在多视图几何中，给定一张图的一个点，其在另一张图的对应点一定分布在对应的极线上。
   - **论文应用**：MVS2D 并没有全图搜索，而是严格沿着极线采样。这不仅符合几何原理，还极大地减少了搜索空间（从 2D 全图搜索降维到 1D 线搜索），这是它“快”的几何根源。
2. **代价体 vs. 注意力 (Cost Volume vs. Attention)**：
   - **代价体**：类似于暴力的“试错法”，把所有可能的深度都算一遍相似度存起来，再用 3D CNN 去“雕刻”出正确的深度面。
   - **注意力**：类似于“查询机制”，源像素作为 Query，极线上的采样点作为 Key/Value。网络自动学习“关注”哪个深度假设，直接聚合特征。这是一种更稀疏、更灵活的信息融合方式。
3. **2D CNN 与 3D CNN 的权衡**：
   - 3D CNN 处理的是 $(C, D, H, W)$ 数据，计算量是 $O(D \cdot H \cdot W)$。
   - 2D CNN 处理的是 $(C, H, W)$ 数据。
   - MVS2D 证明了只要几何先验给得准，强大的 2D 特征提取器配合几何引导，可以替代昂贵的 3D 上下文正则化。

# Multi-View Depth Estimation by Fusing Single-View Depth Probability with Multi-View Geometry

2022

## 关键词

- **Multi-View Stereo (MVS, 多视图立体视觉)**：利用多张图片和相机位姿恢复深度的技术。
- **Single-View Depth Estimation (单目深度估计)**：仅凭一张图推断深度（利用纹理、透视等线索）。
- **Probabilistic Depth Sampling (概率深度采样)**：基于不确定性来选择搜索范围，而非盲目搜索。
- **Cost Volume (代价体)**：MVS 中用于存储不同深度假设下匹配代价的三维张量。
- **Uncertainty (不确定性)**：量化模型对预测结果的“确信度”（这里指方差 $\sigma^2$）。

## 解决的问题

传统的 **多视图立体视觉 (MVS)** 方法虽然精度高，但存在两大痛点 ：

1. **计算与内存消耗大**：通常需要在整个深度范围内（如 0.5m 到 10m）均匀采样大量深度假设（hypothesis），导致构建的 Cost Volume 巨大。
2. **特殊场景失效**：在**弱纹理区域**（如白墙）、**反光表面**（如镜子）、**移动物体**上，几何匹配容易失效，导致错误的深度估计。

另一方面，**单目深度估计**虽然能处理弱纹理（因为它“认识”墙壁），但由于尺度模糊，几何精度较差 。

**核心痛点**：如何让 MVS “变聪明”，不在错误的深度范围内浪费计算资源，并在几何匹配失效时利用单目的先验知识来兜底？

## 核心思想

**MaGNet** (Monocular and Geometric Network) 的核心哲学是：**用单目概率分布来“指导”多视图几何匹配**。

- **直觉理解**：
  - 如果单目网络看了一眼图，说“这面墙大概在 3米远，我很确定（方差小）”，那么 MVS 只需要在 3米附近搜索，不需要从 0米搜到 10米。
  - 如果单目网络说“这是个反光物体，我不确定深度”，那么 MVS 就需要扩大搜索范围。
  - 如果在进行多视图匹配时，邻帧的单目预测认为某个深度“不可能”，那么即便颜色匹配上了（可能是反光造成的假象），我们也应该忽略它。

## 方法

MaGNet 的 pipeline 分为三个主要步骤，形成一个迭代优化的闭环 ：

### Step 1: 单视图概率估计 (D-Net)

- **输入**：单张 RGB 图像。
- **输出**：每个像素的深度分布，建模为高斯分布 $\mathcal{N}(\mu, \sigma^2)$ 。
  - $\mu$：预测的深度。
  - $\sigma$：不确定性（Aleatoric Uncertainty）。
- **作用**：提供初始的深度猜测和搜索范围。

### Step 2: 概率深度采样与一致性加权 (Probabilistic Sampling & Consistency Weighting)

1. **概率采样**：不再均匀采样，而是根据 D-Net 预测的 $\mu$ 和 $\sigma$，在置信区间 $[\mu - \beta\sigma, \mu + \beta\sigma]$ 内采样 。
   - **优势**：在确信的地方少采样（高效），在不确信的地方多采样（鲁棒）。作者只用了 5 个采样点就达到了竞品 64 个采样点的效果 。
2. **一致性加权**：在计算多视图匹配代价（Matching Score）时，引入了一个权重 $w^{dc}$ 。
   - 如果投影到邻帧的点，其深度在邻帧的单目预测中概率很低（说明邻帧认为这不可能是对的），则权重置为 0。
   - **几何意义**：这有效剔除了遮挡（Occlusion）和反光/弱纹理导致的错误匹配 。

### Step 3: 多视图概率分布更新 (G-Net)

- **输入**：由 Step 2 得到的“稀疏” Cost Volume。
- **输出**：更新后的深度分布 $\mu^{new}$ 和 $\sigma^{new}$ 。
- **机制**：G-Net 观察匹配代价，如果发现某个采样点的匹配分数很高，就将 $\mu$ 移向该点，并减小 $\sigma$。

**迭代 (Iterative Refinement)**：将 G-Net 的输出反馈回 Step 2，进行更精细的采样和匹配，通常迭代 3 次 。

## 创新点

- **概率深度采样 (Probabilistic Depth Sampling)**：利用单目不确定性来动态调整 MVS 的搜索空间，极大地压缩了 Cost Volume（体积减少 92%）。
- **深度一致性加权 (Depth Consistency Weighting)**：利用单目先验来验证多视图几何的一致性，解决了 MVS 在反光和弱纹理区域的顽疾 。
- **迭代式融合框架**：设计了一个从 Single-View 到 Multi-View 再反馈回 Single-View 的闭环系统，使得两者互相修正 。

## 实验与结果

**数据集**：ScanNet (室内), 7-Scenes, KITTI (室外) 。

**定量结果**：

- 在 ScanNet 和 7-Scenes 上达到 SOTA（State-of-the-Art）水平 。
- 在 KITTI 上，击败了包括 NeuralRGBD 在内的多视图方法 。
- **效率**：仅使用 15 个深度假设（5个点 $\times$ 3次迭代）就超过了使用 64 个均匀采样点的方法 。

**定性结果**：

- **弱纹理**：白墙重建平滑，没有 MVS 常见的噪声。
- **反光表面**：镜子没有被错误地重建为“镜子里的虚像”，因为一致性加权抑制了错误的几何匹配 。
- **移动物体**：在 KITTI 上，移动车辆的深度估计依然准确（传统 MVS 会因为移动破坏极线约束而失效）。

## 结论

MaGNet 证明了融合单目线索（语义理解、纹理梯度）和多视图几何（物理约束）是解决深度估计难题的最佳路径。通过概率化的方式（均值+方差），模型能够自适应地平衡两者的贡献，既保留了 MVS 的精度，又获得了单目方法的鲁棒性和效率 。

## 可学习知识点

**Cost Volume 的本质**：它本质上是在问“如果深度是 $d$，这几个相机看到的像素长得像不像？”。理解这一点是学习 MVSNet 等深度学习 MVS 方法的基础。

**不确定性 (Uncertainty) 的作用**：在 SLAM 后端优化中，我们用协方差矩阵（Covariance Matrix）来加权误差。这篇论文展示了在前端深度估计中，方差（$\sigma$）如何指导我们“该去哪里搜索匹配点”。

**贝叶斯滤波的思想**：

- **Prior (先验)** = 单目深度估计 (D-Net)。
- **Likelihood (似然)** = 多视图几何匹配 (Matching Score)。
- **Posterior (后验)** = 融合后的深度 (G-Net)。
- 这是一个典型的贝叶斯更新过程的深度学习实现。

**极线搜索 (Epipolar Search)**：论文中提到的将点投影到邻帧进行匹配 ，正是 SLAM 中极线搜索的神经网络版本。

# MVSFormer++ Revealing the Devil in Transformer's Details for Multi-View Stereo

2022

## 关键词

- **Multi-View Stereo (MVS)**: 多视图立体几何，从多张已知位姿的图片中恢复 3D 结构 。
- **Transformer**: 基于自注意力机制的深度学习架构。
- **DINOv2**: Facebook (Meta) 发布的视觉大模型，具有极强的特征提取和泛化能力。
- **Attention Dilution (注意力稀释)**: 当序列长度增加（如图像分辨率变大）时，注意力机制失效的现象 。
- **Cost Volume (代价体)**: MVS 中的核心概念，用于存储不同深度假设下的特征匹配相似度。

## 解决的问题

这篇论文主要解决了将 Transformer 应用于 MVS 时遇到的**“细节魔鬼”**，具体包括：

1. **不同模块需求不同**：MVS 包含“特征提取”和“代价体正则化”两个主要步骤。以前的方法盲目套用 Transformer，忽略了这两个模块对特征聚合的需求截然不同 。
2. **预训练模型缺乏跨视图信息**：强大的预训练模型（如 ViT/DINO）通常是单目训练的，直接用在 MVS 中缺乏多视角几何（Epipolar geometry）的交互能力 。
3. **长度外推（Length Extrapolation）难题**：这是工程落地的痛点。训练时图片通常较小（如 $640 \times 512$），但测试（如 SLAM 重建）时图片很大（如 2K, 4K）。Transformer 在处理这种序列长度剧烈变化时，性能会大幅下降 。

## 核心思想

**“因地制宜”与“几何感知”**。 作者认为不能用同一套 Attention 机制打通全场。

- **特征编码端**：利用 **Side View Attention (SVA)** 将跨视图信息注入到冻结的 DINOv2 中，并使用线性注意力（Linear Attention）来处理高分辨率特征。
- **代价体端**：将代价体视为一个全局序列，使用标准注意力（Vanilla Attention）进行去噪，但引入 **Frustoconical Positional Encoding (FPE)** 和 **Adaptive Attention Scaling (AAS)** 来解决高分辨率下的几何定位和注意力稀释问题。

## 方法

论文提出了 **MVSFormer++**，主要包含两个改进部分：

### 特征编码器：SVA (Side View Attention)

- **利用 DINOv2**：使用冻结参数的 DINOv2 提取强大的语义特征 。
- **旁路注入 (Side-Tuning)**：设计了一个轻量级的 SVA 模块，与 DINOv2 并行。它不改变 DINOv2 权重，而是通过 Cross-Attention 学习参考图像（Reference）和源图像（Source）之间的几何关联 。
- **线性注意力 (Linear Attention)**：作者发现对于特征提取，基于特征聚合的线性注意力效果最好，且计算量低，适合处理大图 。
- **标准化 2D 位置编码 (Normalized 2D-PE)**：为了适应分辨率变化，将像素坐标归一化到统一尺度，保证训练和测试的位置感知一致性 。

### 代价体正则化：CVT (Cost Volume Transformer)

- **纯Transformer正则化**：放弃了传统的3DCNN，将代价体展平为序列，使用 FlashAttention 加速的标准注意力机制进行全局去噪 。
- **FPE (3D 视锥位置编码)**：
  - MVS 的深度搜索通常是在视锥（Frustum）内进行的（基于逆深度）。
  - 作者设计了一种符合视锥几何的 3D 位置编码，将空间坐标归一化到视锥的近平面和远平面之间 。这让模型理解“相对深度”，而非绝对坐标，极大提升了泛化能力。
- **AAS (自适应注意力缩放)**：
  - **问题**：当图像分辨率变大，序列长度 $N$ 暴增。Softmax 操作会导致注意力分数变小（分布变平），有效信息被“稀释” 。
  - **解法**：根据序列长度动态调整 Attention 中的缩放因子 $\lambda$。公式为 $Attention = Softmax(\frac{\kappa \log n}{\sqrt{d}} QK^T)V$ 。这保证了无论图片多大，注意力的熵保持不变。

## 创新点

- **差异化注意力设计**：首次明确指出特征编码适合 Linear Attention（关注特征聚合），而代价体正则化适合 Vanilla Attention（关注空间去噪和全局关联） 。
- **SVA 架构**：提出了一种高效将多视角信息注入预训练 ViT 的方法，无需微调大模型本身 。
- **FPE (Frustoconical PE)**：将位置编码与 MVS 的相机视锥几何紧密结合，解决了 Transformer 在 3D 空间中的位置感知问题
- **AAS (Adaptive Attention Scaling)**：从数学上解决了 Transformer 在 MVS 高分辨率测试时的“水土不服”问题 。

## 实验与结果

**数据集**：DTU (室内物体), Tanks-and-Temples (室外大场景), ETH3D 。

**性能**：

- 在 **DTU** 和 **Tanks-and-Temples** 榜单上达到了 **SOTA (State-of-the-Art)** 。
- 在 T&T 的 Intermediate 集上 F-score 达到 67.03，Advanced 集达到 41.70，显著优于 MVSFormer 和 GeoMVSNet 。

**消融实验 (Ablation Study)**：

- 证明了 Linear Attention 在代价体上表现极差（因为它丢失了 dot product 后的相关性信息），但在特征提取上表现最优 。
- 证明了没有 AAS 和 FPE，模型在高分辨率图片上的性能会显著下降 。

## 结论

MVSFormer++ 通过深入分析 Transformer 在 MVS 各个环节的细节，成功将强大的预训练模型（DINOv2）与几何感知的 Transformer 组件结合。特别是提出的 FPE 和 AAS，有效解决了 Transformer 在不同分辨率下的泛化难题，为高精度稠密重建树立了新标杆 。

## 可学习知识点

**从 2D 到 3D 的特征鸿沟**：

- 仅仅用 ImageNet 预训练的模型是不够的，SLAM/MVS 需要理解“匹配”和“极线几何”。SVA 模块就是为了弥补这个鸿沟。

**分辨率泛化 (Resolution Generalization)**：

- 在 SLAM 中，相机分辨率各异。学习 **Normalized PE** 和 **AAS** 的思想，明白为什么深度学习模型训练完不能直接换个分辨率跑。
- **直觉**：Softmax 在处理 100 个点和 10000 个点时，输出的概率分布尖锐程度完全不同。AAS 就是通过数学手段强制让它们保持一致。

**视锥几何 (Frustum Geometry)**：

- 理解为什么 MVS 和 SLAM 中常用 Inverse Depth（逆深度）。FPE 的设计正是基于这种几何结构（近大远小）。

**Transformer 变体选择**：

- 并不是所有 Attention 都是一样的。Linear Attention 牺牲了部分表达能力换取 $O(N)$ 复杂度，适合特征提取；Standard Attention 是 $O(N^2)$，但在需要精细去噪的 Cost Volume 阶段是必须的。

# Adaptive Fusion of Single-View and Multi-View Depth for Autonomous Driving

2024

## 关键词

- **Depth Estimation (深度估计):** 也就是从图像恢复场景的 $Z$ 轴信息。
- **Multi-View Stereo (MVS, 多视图立体视觉):** 利用几何三角测量恢复深度，依赖准确的位姿。
- **Single-View Depth (单目深度估计):** 利用深度神经网络对场景的语义理解恢复深度，不依赖位姿，但有尺度模糊。
- **Adaptive Fusion (自适应融合):** 动态选择权重的融合策略。
- **Robustness (鲁棒性):** 系统在噪声（如位姿误差）下的稳定性。
- **Autonomous Driving (自动驾驶):** 应用场景。

## 解决的问题

这篇论文主要解决了现有 **多视图深度估计系统在“非理想”条件下的脆弱性问题**。

- **痛点 1：位姿噪声 (Noisy Poses):** 传统的 MVS 方法（如 MVSNet）高度依赖相机位姿（Pose）的准确性来构建极线几何（Epipolar Geometry）。然而，在自动驾驶的 SLAM 系统中，位姿往往含有噪声（漂移、未校准、闭环延迟等）。一旦位姿不准，几何约束就会将像素匹配到错误的位置，导致深度估计完全崩坏 。
- **痛点 2：动态物体与弱纹理:** MVS 假设场景是静态的。对于移动的车辆或无纹理的路面，基于几何匹配的方法会失效 。
- **现有融合方法的缺陷:** 之前的一些融合单目和多视图的方法（如 MaGNet, MVS2D），在位姿有噪声时，往往会被错误的多视图几何信息“带偏”，导致性能甚至不如单纯的单目网络 。

## 核心思想

**“几何不够，语义来凑；谁准信谁。”**

核心思想是构建一个双分支网络（单目分支 + 多视图分支），并设计一个 **自适应融合模块 (Adaptive Fusion Module, AF Module)**。这个模块充当“裁判”，通过检查**几何投影的一致性**（Warping Confidence），动态地决定在图像的每一个像素点上，是信任多视图分支（几何精度高但脆弱）还是单目分支（鲁棒性强但精度一般）。

## 方法

**特征提取 (Feature Extraction):**

- 两个分支共享同一个 Backbone (如 ConvNeXt)，提取多尺度特征 。这保证了效率。

**双分支预测 (Two-Branch Prediction):**

- **单目分支 (Single-View Branch):** 类似于传统的单目深度估计网络，利用 Decoder 恢复深度 $d_s$ 和单目置信度 $M_s$ 。它主要依赖语义线索（比如：看到车轮知道车的大小，看到地平线推测距离）。
- **多视图分支 (Multi-View Branch):** 基于 Cost Volume（代价体）的方法。它利用给定的位姿构建 Plane Sweep Volume，通过比较特征相似度来寻找深度 $d_m$ 和多视图置信度 $M_m$ 。
- *特征融合 (Feature Fusion):* 作者还将单目的特征注入到多视图分支中，弥补 MVS 在弱纹理区域的特征缺失 。

**自适应融合模块 (Adaptive Fusion Module - 关键!):**

- **裁判员机制 ($M_w$):** 为了判断多视图分支是否靠谱，作者引入了 **Warping Confidence Map ($M_w$)**。
- **计算方法:** 利用多视图分支预测的深度 $d_m$ 和输入的位姿，将相邻帧图像 Warp（变换）到当前帧。如果位姿准、深度准且物体没动，Warp 过来的图像应该和当前帧很像（光度一致性）。如果差异大，说明几何约束失效了（可能是位姿错了，也可能是动目标）。
- **最终融合:** 网络接收 ($d_s, M_s, d_m, M_m, M_w$)，通过卷积层输出最终的深度 $d_{fuse}$ 。这意味着网络学会了：**当 $M_w$ 低时（几何不可靠），自动通过权重向 $d_s$ 倾斜。**

## 创新点

- **AFNet 架构:** 首次提出专门针对位姿噪声鲁棒性的单目与多视图深度自适应融合网络 。
- **鲁棒性基准测试 (Robustness Benchmark):** 论文发现现有领域缺乏对位姿噪声的测试标准，因此建立了一套新的基准：在数据集中人为注入不同程度的位姿噪声，量化评估算法的抗干扰能力 。
- **Warping Confidence ($M_w$):** 利用重投影误差（Photometric Consistency）作为显式的信号来指导融合，巧妙地解决了何时该信几何、何时该信语义的问题。

## 实验与结果

**数据集:** DDAD 和 KITTI 。

**标准性能:** 在位姿准确（Ground Truth Pose）的情况下，AFNet 在 DDAD 和 KITTI 上均达到了 State-of-the-Art (SOTA)，AbsRel 误差显著低于 MaGNet 等方法 。

**抗噪性能 (关键结果):**

- 当引入位姿噪声时，传统的 MVS 方法（如 IterMVS）和普通融合方法（如 MVS2D）性能急剧下降，甚至不如纯单目方法。
- AFNet 在各种噪声水平下均保持了最高的精度，且非常稳定（见论文 Figure 4）。

**动态物体:** 在移动车辆区域，AFNet 的误差比纯多视图分支降低了 21%，证明了融合单目信息能有效处理动态场景 。

**真实 SLAM 噪声测试:** 作者使用了 ORB-SLAM2 生成的真实（带有漂移的）位姿进行测试，AFNet 依然表现最好 。

## 结论

这篇论文证明了在实际的自动驾驶场景中，完全依赖几何约束（MVS）是危险的。AFNet 通过自适应地结合单目语义信息和多视图几何信息，不仅提高了精度，更重要的是极大地提升了系统在面对位姿噪声和动态场景时的鲁棒性 。

## 可学习知识点

**极线几何与 Cost Volume (Epipolar Geometry & Cost Volume):**

- *原理:* 为什么 MVS 需要位姿？因为已知位姿才能画出极线，才能在极线上搜索匹配点（构建 Cost Volume）。
- *直觉:* 论文中的 Multi-view branch 就是一个经典的 MVSNet 变体。理解它有助于你理解深度学习如何模拟传统的立体匹配。

**尺度模糊 (Scale Ambiguity):**

- *概念:* 单目相机无法知道物体的真实大小（是玩具车还是真车？）。
- *论文中的体现:* 单目分支虽然鲁棒但精度受限，而多视图分支利用相机移动（基线）提供的几何约束解决了尺度问题。融合就是为了兼得二者之长。

**光度一致性假设 (Photometric Consistency):**

- *核心公式:* $I_1(p) \approx I_2(p')$。如果在 $p$ 点的深度估计正确且位姿正确，投影过去的像素颜色应该是一样的。
- *论文应用:* $M_w$ (Warping Confidence) 本质上就是利用这个假设来检测错误。如果不一致（Error 大），说明假设破裂（位姿错、深度错或物体动了），这时候就该切回单目模式。

**鲁棒性 (Robustness) 在 SLAM 中的重要性:**

- 教科书上的算法常假设位姿完美，但实战中 SLAM 经常“抖动”。这篇论文教你在设计算法时要考虑到输入的缺陷（Input Degradation）。
