# Group-wise Correlation Stereo Network

2019CVPR

## 关键词

stereo matching, cost volume, correlation, 3D CNN



## 解决问题

现有：

- concatenation直接拼接：
- - 信息丰富，显存代价大
  - e.g. PSMNet
- Full Correlation全相关：
- - 计算高效，信息表达不足
  - e.g. FlowNet-C

## 核心思想

- 左右特征在通道维度上分组，在组内进行相关性计算，构建多通道且高效的cost volume

## 方法

- 左右图像过2D CNN提取多尺度特征
- 构建group-wise correlation cost volume
- 使用3D CNN进行正则化与视差回归

### 公式

#### 全相关 (Full Correlation)

$$ C_{corr}(d, x, y) = \frac{1}{N_c} \langle \mathbf{f}_l(x, y), \mathbf{f}_r(x-d, y) \rangle $$

- **符号含义**:
  - $$\mathbf{f}_l, \mathbf{f}_r$$: 左右图像的特征向量。
  - $$N_c$$: 特征的总通道数。
  - $$\langle \cdot, \cdot \rangle$$: 向量内积（点积），即将两个向量对应元素相乘再求和。
  - $$d$$: 视差。

特点：

- **物理意义**: 显式地计算了两个特征向量的余弦相似度（未归一化）。数值越大，表示越相似。
- **输出维度**: **1 通道**。无论输入特征有多少通道（比如 320 维），输出在每个视差等级上只有一个数值。

#### 直接拼接 (Direct Concatenation)

$$ C_{concat}(d, x, y) = \text{Concat} \{ \mathbf{f}_l(x, y), \mathbf{f}_r(x-d, y) \} $$

**符号含义**:

- $$\text{Concat} \{ \cdot \}$$: 在通道维度（Channel Dimension）上进行拼接操作。

特点：

- **物理意义**: 仅仅是将左右特征放在一起，网络本身此时不知道它们的相似关系。后续的 3D 卷积网络（3D CNN）需要从零开始学习“什么样的特征组合代表匹配，什么样的代表不匹配”。
- **输出维度**: **$$2N_c$$ 通道**。如果输入是 320 维，拼接后变成 640 维。

#### 分组相关代价体 (Group-wise correlation volume)

$$ C_{gwc}(d, x, y, g) = \frac{1}{N_c/N_g} \langle \mathbf{f}_l^g(x, y), \mathbf{f}_r^g(x-d, y) \rangle $$

符号：

- **$$C_{gwc}$$**: 分组相关代价体 (The Group-wise Correlation Volume)。
- **$$d$$**: 视差 (Disparity)，即右图特征向左平移的像素量。
- **$$x, y$$**: 像素坐标。
- **$$g$$**: 当前计算的是第 $$g$$ 个分组 (Group index)，范围是 $$0$$ 到 $$N_g-1$$。
- **$$N_c$$**: 特征提取网络输出的总通道数 (Total Channels)。
- **$$N_g$$**: 分组的数量 (Number of Groups)。
- **$$N_c/N_g$$**: 每一个分组内包含的通道数。
- **$$\mathbf{f}_l^g, \mathbf{f}_r^g$$**: 分别表示左图和右图的**第 $$g$$ 个特征组**。
  - 这就是把总特征 $$\mathbf{f}$$ 沿着通道维度切开，取第 $$g$$ 份。
- **$$\langle \cdot, \cdot \rangle$$**: 向量的**内积 (Inner Product)**。
- **$$\mathbf{f}_r^g(x-d, y)$$**: 表示将右图特征在 x 轴方向平移 $$d$$ 个单位后，在位置 $$(x,y)$$ 处的值（对应原始右图的 $$(x-d, y)$$）。

---

#### 物理过程

1. **分组 (Splitting)**:
    假设提取出的特征 $$\mathbf{f}$$ 有 320 个通道 ($$N_c=320$$)。设定分组数 $$N_g=40$$。
    那么，网络会将这 320 个通道平均切成 40 组，每组有 $$320/40 = 8$$ 个通道。

2. **相关性计算 (Correlation)**:
    对于第 $$g$$ 组（比如第0组），取出左图的那8个通道的向量，和右图（平移 $$d$$ 后）对应位置的那8个通道的向量，做**点积**运算。
    - 公式前面的 $$\frac{1}{N_c/N_g}$$ 是一个归一化系数（相当于求平均），防止数值过大。

3. **堆叠 (Packing)**:
    对所有的组 ($$g=0 \dots 39$$) 都算一遍，会得到 40 个数值。
    把这 40 个数值拼起来，就得到了在视差 $$d$$、位置 $$(x,y)$$ 处的代价向量（长度为 $$N_g$$）。
    对所有视差 $$d$$ 重复此步骤，最终形成 4D 代价体。

#### 总结

这个公式的精髓在于：**它既不像“全相关”那样把所有通道压缩成一个数（丢失信息），也不像“拼接”那样保留所有通道（计算太重），而是取了个折中，算出了 $$N_g$$ 个维度的相似度特征。**

## 主要创新

- 提出group-wise Correlation
- 信息量和复杂度之间取得平衡
- 可替代传统concatenation/full correlation

## 实验和结果

- 数据集：Scene Flow， KITTI
- 结论：
- - 精度接近甚至优于concatenation
  - 显著降低显存和计算量
- Ablation：group数影响性能，存在trade-off

## 知识点

- 几何：stereo matching pipeline
- 传统立体+slam：
- - matching cost，cost volume，dense & sparse depth, stereo在slam中的位置
- Dense Matching：
- - CNN特征提取，Correlation layer，cost volume learning，3D CNN
  - e.g. FlowNet-C，PSMNet，GwcNet，RAFT

# Cost volume

目的：在双目立体中，左图像素点(x,y)是右图中的哪个点

- 在右图中尝试(x,y),(x-1,y),(x-2,y)等，出一个序列：

```css
d=0   d=1   d=2   d=3   ...
[ 3.2, 1.1, 0.4, 2.7, ... ]
```

称为cost curve：代价曲线

对整张图进行上述操作，得到一个映射关系：

```css
(x, y, d) → cost
```

称为cost volume

- 假设所有点的视差都是d（取一个d），可以得到一整张图的cost矩阵

---

直接选最小cost？

- 噪声，遮挡，重复纹理，会使单点匹配不稳定

**空间一致性假设**：

- 临近像素深度差不多
- 表面连续
- 因此：(x,y,d),邻居(x±1,y,d±1)也要看（上下文优化）

## 深度学习中的cost volume

C(x,y,d) 到 C(x,y,d,c)

- 多了channel维度
- channel表示不同的尺度/不同语义的相似性
- e.g. GwcNet：每个group是一个通道

# RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching

2021

Stereo Matching, Disparity Estimation, Recurrent Refinement, Global Correlation, RAFT

## 研究问题

- 给定一对左右视图（Rectified Stereo Images），预测每个像素的**视差（disparity）**。

主要挑战：

- 大视差范围难以准确匹配
- 遮挡、弱纹理区域匹配不稳定
- 传统 3D cost volume 显存和计算开销大
- 模型泛化能力差（对真实数据不鲁棒）

## 核心思想

**RAFT-Stereo 将立体匹配问题建模为一个“迭代优化问题”**：

- 从一个初始视差场开始，通过神经网络反复迭代更新视差，而不是一次性预测最终结果。

特点：

- 不显式构建 3D cost volume
- 使用 **全局相关（all-pairs correlation）**
- 使用 **循环神经网络（Conv-GRU）** 进行迭代更新

## 方法

### 特征提取（Feature Extraction）

- 左右图像分别通过 CNN 提取特征
- 使用共享权重，保证特征空间一致性

------

### 全局相关体（All-pairs Correlation）

- 计算左图每个像素与右图所有水平位置的相关性
- 构成 **1D 全局相关体**
- 只沿水平方向计算，符合 stereo 几何约束

------

### 多层相关金字塔（Multilevel Correlation Pyramid）

- 对全局相关体进行多尺度下采样
- 不同尺度负责不同范围的视差搜索
- 提升效率并扩大感受野

------

### 迭代更新模块（Recurrent Update）

- 使用 **Conv-GRU**
- 输入：
  - 当前视差估计
  - 多尺度相关特征
  - 左图上下文特征
- 输出：
  - 视差增量（Δd）

通过多次迭代（通常 10–20 次）逐步细化视差。

### 结构

```mathematica
Left Image  ──► CNN ─┐
                     ├─► All-pairs Correlation ─► Correlation Pyramid
Right Image ─► CNN ──┘                                    │
                                                          ▼
                                   Conv-GRU (iterative refinement)
                                                          ▼
                                              Disparity Map

```

## 创新点

- 将 RAFT 的迭代优化思想引入 Stereo Matching
- 使用 1D 全局相关代替传统 3D cost volume
- 多尺度相关金字塔提升效率和精度
- 通过递归更新显著提升泛化能力

## 实验与结果

- 在 **Scene Flow、KITTI** 等数据集上达到或接近 SOTA
- 对真实场景具有更好的泛化能力
- 在遮挡区域和细结构区域表现稳定

## 结论

将立体匹配视为一个“可学习的迭代优化过程”，可以在精度、效率和泛化性上同时超越传统方法。

## 知识点

- 一次性回归（one-shot prediction）
- 迭代优化（iterative refinement）
- GRU
- Pyramid

## 总结

- RAFT-Stereo 首先在左右图特征之间构建全局 correlation volume，并在此基础上形成多尺度的 correlation pyramid。
- 在每一次迭代中，模型以当前视差估计为中心，在各尺度的 correlation pyramid 上进行局部 lookup，获取局部匹配代价的形状信息。
- 这些多尺度的局部相关特征与上下文信息一起输入 convGRU，用于预测视差的增量更新。
- 通过多次迭代，模型在 coarse-to-fine 的效果下逐步优化视差估计。

# Attention Concatenation Volume for Accurate and Efficient Stereo Matching

2022

Stereo Matching, Cost Volume, Attention Concatenation Volume, Channel-wise Attention, Correlation Volume, Accuracy–Efficiency Trade-off



## 背景

**Concatenation-based Cost Volume**

- 表达能力强
- 计算量与显存开销巨大（3D CNN 成本高）

**Correlation-based Cost Volume**

- 计算高效
- 表达能力不足，容易丢失特征信息

## 核心思想

一种新的 Cost Volume 构建方式 ——**Attention Concatenation Volume（ACV）**

- 使用 **correlation 信息生成 attention 权重**，再用 attention 引导特征拼接，从而保留有效信息、抑制冗余通道。

## 方法

### Attention Concatenation Volume（ACV）

构建过程分为三步：

1. **Correlation 计算**
   - 对左右特征在不同 disparity 下计算内积
   - 得到 2D correlation map
2. **Attention 权重生成**
   - 由 correlation map 通过轻量网络生成
   - Attention 为 **channel-wise、disparity-dependent**
3. **Attention-guided Feature Concatenation**
   - 对左右特征分别进行通道加权
   - 再进行拼接，形成 ACV

最终得到的信息更加紧凑、判别性更强的 cost volume。

### 网络整体结构

- 使用 2D CNN 提取左右图像特征
- 构建 Correlation Volume
- 生成 Attention Map
- 构建 Attention Concatenation Volume
- 使用 3D CNN 进行 cost aggregation
- 通过 regression 得到 disparity

## 方法优势

- **信息表达能力强**
  - 保留拼接特征的完整性
- **计算效率高**
  - Attention 为 2D 结构
  - 减少 3D CNN 的无效计算
- **显存占用低**
  - 相比传统 concatenation volume 更轻量
- **模块可插拔**
  - 可替换传统 cost volume 结构

## 实验结果

在 **Scene Flow、KITTI 2012 / 2015** 上验证

相比 PSMNet、GWCNet：

- 精度更高或相当
- 推理速度更快
- 显存需求更低

在弱纹理和复杂场景下表现更稳定

## 结论

- ACVNet 通过 **Attention-guided Concatenation Volume**，有效平衡了 **accuracy 与 efficiency**，
- 是一种实用且可扩展的 stereo matching cost volume 设计。

## 伪代码

```python
fL = FeatureExtractor(left_image)        # 左图特征 (H×W×C)
fR = FeatureExtractor(right_image)       # 右图特征 (H×W×C)

for d in disparities:                    # 遍历所有视差
    corr[d] = dot(fL(x,y), fR(x-d,y))    # 相关性（相似度）

attn = AttentionNet(corr)                # 由相关性生成注意力权重

for d in disparities:
    fL_d = attn[d] ⊙ fL(x,y)             # 通道加权左特征
    fR_d = attn[d] ⊙ fR(x-d,y)            # 通道加权右特征
    cost[d] = concat(fL_d, fR_d)          # Attention-guided 拼接

disparity = Regression(3D_CNN(cost))     # 代价聚合 + 回归

```

# Iterative Geometry Encoding Volume for Stereo Matching

2023

Stereo Matching, Disparity Estimation, Iterative Refinement, Geometry Encoding Volume, Epipolar Geometry, Correlation-based Matching, Recurrent Update, Memory-efficient Stereo

## 研究背景

立体匹配（Stereo Matching）的核心目标是：

- 根据左右视图估计每个像素的视差（disparity）。

传统深度学习方法通常构建 **3D cost volume（H×W×D）**，并通过 3D CNN 进行正则化与回归（如 PSMNet、GC-Net）。
 但这类方法存在明显缺点：

- 计算量和显存开销随视差范围线性增长
- 推理速度慢，难以扩展到高分辨率或大视差
- 一次性预测，缺乏逐步修正能力

**迭代式方法（如 RAFT-Stereo）** 避免了完整 3D cost volume，但对立体视觉中的**几何约束利用不够显式**。

## 核心思想

本文提出 **Iterative Geometry Encoding Volume（IGE Volume）**，将**显式立体几何约束**引入**迭代视差优化框架**中。

核心思想可以概括为三点：

- **不构建完整 3D cost volume**
- **利用极线几何，仅在 disparity 维度进行匹配**
- **通过迭代方式逐步更新视差估计**

立体匹配被重新表述为一个**迭代优化问题**，而非一次性回归问题。

## 方法

### 网络整体结构

整体框架由以下几个模块组成：

- **特征提取网络（Feature Encoder）**对左右图像提取多尺度特征。

- **几何编码体（Geometry Encoding Volume）**在 disparity 维度对左右特征进行相关性计算，并编码几何信息。

- **迭代更新模块（Iterative Update Module）**使用循环结构（GRU-like）逐步更新 disparity。

- **视差预测（Disparity Prediction）**每一次迭代都会输出 refined disparity。

### Geometry Encoding Volume（核心贡献）

与传统 3D cost volume 不同，IGE Volume：

- **不显式构建 (x, y, d) 三维体**
- 利用左右图像已校正的假设：
  - 匹配仅发生在**同一水平行**
- 对每个像素：
  - 在当前 disparity 估计附近
  - 进行局部搜索
  - 编码几何与相关性信息

几何编码体中包含的信息包括：

- 左右特征的相关性（correlation）
- disparity 偏移（Δd）
- 当前 disparity 估计
- 局部几何上下文

该设计显式引入立体几何约束，提高匹配效率与鲁棒性。

### Iterative Disparity Refinement

视差估计过程采用迭代更新方式：

- 初始化 disparity（通常为 0 或粗估计）
- 从 Geometry Encoding Volume 中采样几何信息
- 使用 GRU-like 更新模块预测 disparity 增量
- 更新 disparity
- 重复上述过程若干次

该过程与 RAFT 类似，但针对 stereo 任务引入了显式几何编码。

## 实验与结果

在主流立体匹配数据集（如 Scene Flow、KITTI）上取得了：

- 更高精度
- 更低显存占用
- 更快推理速度

相比传统 cost volume 方法，在大视差范围下优势明显。

# MonSter Marry Monodepth to Stereo Unleashes Power

2025

## 关键词

- **立体匹配 (Stereo Matching)**
- **单目深度估计 (Monocular Depth Estimation)**
- **互补学习 (Complementary Learning)**
- **病态区域 (Ill-posed Regions)**：指无纹理、反光、遮挡区域。
- **尺度与偏移恢复 (Scale and Shift Recovery)**

## 解决的问题

- **立体匹配的缺陷**：在**弱纹理（Textureless）**、**反光（Reflective）**、**细小结构（Thin structures）**和**遮挡**区域，基于像素匹配的方法无法计算正确的视差（Disparity），因为找不到对应的特征点 。
- **单目深度的缺陷**：虽然单目方法对结构理解很好（能看出墙是平的），但它估计出的深度是“相对的”，存在严重的**尺度（Scale）和偏移（Shift）模糊** 。

**核心痛点**：现有的方法要么只做匹配（Stereo），要么只做估计（Mono），或者简单融合，无法同时获得“高鲁棒性”和“绝对尺度” 。

## 核心思想

**“联姻（Marry）”**：将立体匹配重构为“单目深度估计 + 逐像素尺度偏移恢复”的过程 。

- 利用 **Stereo** 提供的物理几何约束（Metric cues）来修正 **Mono** 的尺度和偏移。
- 利用 **Mono** 提供的强大语义先验（Structural priors）来引导 **Stereo** 在匹配失效区域的预测 。

## 方法

MonSter 采用了双分支架构，并通过迭代的方式互相优化。

- **架构概览**：包含一个单目深度分支（基于 DepthAnything V2）和一个立体匹配分支（基于 IGEV），以及一个互细化模块 。
- **关键步骤**：
  1. **全局对齐 (Global Scale-Shift Alignment)**：
     - 先用最小二乘法，将单目深度 $$D_M$$ 粗略对齐到双目视差 $$D_S$$ 的空间：$$D_{M}^{0}=s_{G}D_{M}+t_{G}$$ 。
     - **SLAM 直觉**：这就像在做单目 SLAM 初始化时，用几帧已知的位姿来对齐地图的尺度。
  2. **双目引导对齐 (SGA - Stereo Guided Alignment)**：
     - **目的**：修正单目深度的局部误差。
     - **原理**：利用双目匹配的**置信度（Confidence）**作为权重。在双目匹配可信的地方（纹理丰富处），计算残差 $$\Delta t$$ 来微调单目深度的偏移，使其变成精确的“单目视差” 。
  3. **单目引导细化 (MGR - Mono Guided Refinement)**：
     - **目的**：修复双目匹配在白墙、反光处的空洞。
     - **原理**：将修正后的单目几何特征作为条件（Condition），输入到双目分支的 GRU（循环单元）中。告诉网络：“这里虽然没纹理，但单目分支说它是个平面，请照着这个结构修补视差” 。

## 创新点

- **解耦范式**：首次提出将立体匹配解耦为“单目估计”和“尺度偏移恢复”两个子问题，打破了传统只靠匹配 Cost Volume 的限制 。
- **SGA & MGR 模块**：设计了具体的交叉注意力/引导机制（Cross-branch guidance），让两个分支根据各自的“自信程度”动态交换信息，而不是简单的加权平均 。
- **零样本泛化能力**：通过利用预训练单目模型的先验，MonSter 即使只在合成数据（Scene Flow）上训练，也能在真实数据集（KITTI）上取得 SOTA 效果 。

## 实验与结果 (Experiments & Results)

- **霸榜表现**：在 SceneFlow, KITTI 2012, KITTI 2015, Middlebury, ETH3D 五大主流榜单上均排名 **第一 ($$1^{st}$$)** 。
- **精度提升**：
  - Scene Flow 数据集 EPE（端点误差）降至 **0.37px**，比之前的最佳方法（SOTA）提升了 **15.91%** 。
  - 在 ETH3D（高难度真实场景）上，比之前的最佳方法提升了 **49.5%** 。
- **病态区域分析**：在反光区域（Reflective Regions）和弱纹理区域，误差显著低于 IGEV 和 CREStereo 。

## 结论 (Conclusion)

MonSter 证明了单目深度先验对于立体匹配是极具价值的补充。通过显式地建模尺度和偏移（Scale and Shift），可以有效地融合两者的优势，显著提高算法在非理想环境下的鲁棒性和精度 。

## 知识点

- **几何与先验的互补 (Geometry vs. Priors)**：
  - *概念*：SLAM 是基于几何的（三角测量），Deep Learning 是基于先验的（数据统计）。
  - *学习点*：理解为什么纯几何方法在白墙失效（Cost Volume 峰值不明显），而深度学习可以“猜”出深度。这篇论文是两者结合的典范。
- **尺度模糊 (Scale Ambiguity)**：
  - *概念*：单目相机无法知道物体的真实大小（Scale），只能知道相对关系。
  - *学习点*：论文中的公式 $$D = s \cdot d + t$$ 是处理单目深度的通用数学模型。在单目 SLAM 初始化或闭环校正中，我们本质上也是在解算这个 $$s$$。
- **视差与深度的关系 (Disparity vs. Depth)**：
  - *公式*：$$z = \frac{f \cdot B}{d}$$ （$$z$$: 深度, $$f$$: 焦距, $$B$$: 基线, $$d$$: 视差）。
  - *学习点*：论文中所有的操作都在“视差域”进行，因为视差与图像像素是线性对应的，更适合卷积神经网络处理。
- **代价体 (Cost Volume)**：
  - *概念*：立体匹配的核心数据结构。
  - *学习点*：了解 IGEV 提到的 Geometry Encoding Volume，这是现代双目深度估计和 MVS（多视图立体视觉）的基石。

## 总结 (Summary)

MonSter 是一篇将**深度学习先验**成功引入**传统多视图几何问题**的佳作。它没有抛弃传统的立体匹配流程（Feature Matching, Cost Volume），而是用单目深度作为一个强大的“辅助约束”，专门解决传统几何方法无法处理的**病态区域**。

**对于 SLAM 研究的启示**：未来的 SLAM 系统可能不再是纯几何的，而是“神经辅助的几何系统”（Neuro-geometric system），在几何算不对的地方，让神经网络来“脑补”，再用几何约束来防止网络“瞎编”。
