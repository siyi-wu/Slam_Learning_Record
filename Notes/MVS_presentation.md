涉及到的文章：

* MVSNet: Depth Inference for Unstructured Multi-view Stereo
* Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching
* IterMVS: Iterative Probability Estimation for Efficient Multi-View Stereo
* MVSFORMER++: REVEALING THE DEVIL IN TRANSFORMER’S DETAILS FOR MULTI-VIEW STEREO
* Group-wise Correlation Stereo Network
* TransMVSNet: Global Context-aware Multi-view Stereo Network with Transformers

主线是Muti-view Stereo

# Muti-view Stereo

Muti-view Stereo，即多视图几何，主要是解决在已知相机位姿的情况下，怎么从多视角的2D图像，恢复出3D结构的问题。3D结构，指的可以是2D的深度图，稠密的点云，或者三维模型。

在将要介绍的文章当中，MVSNet是MVS的“开山之作”。它提出了标准的MVS pipeline，后续的文章几乎都是在它的基础上进行修改或创新，或者借鉴了MVSNet的主要思想。Cascade Cost volume（CasMVSNet）引入了由粗到细（Coarse to fine）的机制，降低了MVSNet显存使用；IterMVS用基于GRU的循环神经网络，在架构上进行了修改，用时间换取空间，进一步压缩了显存。MVSFormer++则是结合了Transformer架构，解决了弱纹理/高反光的匹配难点，这极大提升了极端场景下的代价体质量。

而GwcNet和TransMVSNet用于辅助理解上述MVS论文中一些重点。其中GwcNet是立体匹配中承上启下的一篇文章，它重新定义了Cost Volume的构建方法（代价组相关）；TransMVSNet是业内首次尝试将Transformer引入到MVS中，推动了MVS领域从纯CNN架构向 Transformer 全局注意力机制的演进。

这里将详细介绍MVSNet，同时一并介绍其他文章解决的问题及创新点。个人理解恐有错误，望理解。

# MVSNet

## 背景

MVSNet其实是对传统立体匹配标准流程的“深度学习化”。在传统的立体匹配中，我们通常使用手工设计的度量标准（如NCC归一化互相关）来计算像素块之间的相似度。但这种手工特征在面对弱纹理、反光或光照剧烈变化的区域时，往往极其脆弱，容易产生错误的匹配代价。到初始代价后，传统方法会使用SGM（半全局匹配）等算法在多个一维方向上进行聚合，以过滤噪声并平滑视差。虽然经典，但 SGM 本质上还是基于人工设定的平滑惩罚项，难以适应复杂的真实 3D 表面。

随着深度学习的发展，SurfaceNet和LSM率先尝试用神经网络解决多视图立体匹配。它们将整个 3D 空间划分成一个个体素（Voxel），把多视角的图像特征投影到这个巨大的3D网格中，然后用3D卷积去分类每个体素是不是物体的表面。这导致显存消耗巨大。同时由于分辨率受限于Voxel的大小，这一组trade-off导致这两种方法不能扩展到大规模重建。

在这个背景下，MVSNet出现了：它结合了平面扫掠（传统几何）和特征提取（深度学习）的方法，放弃了全局体素网格，而以参考相机的视锥体为基础，以可微单应性变换构建出Cost Volume，完成了MVS端到端的实现。

## 模型架构

![image-20260318194221861](./MVS_presentation.assets/image-20260318194221861.png)

总体上，MVSNet分为特征提取、单应性变换、代价体正则化、深度图优化几部分。

### 特征提取

传统方法（例如SGM）使用的是原始像素，而MVSNet先用CNN将图像转换为高维特征图。这里用到的是共享权重的特征提取方法，将原图的长宽都缩小为原图的1/4。这一步降采样实际上是为了后续 降显存而做的。

### 可微单应性变换

![image-20260318211150796](./MVS_presentation.assets/image-20260318211150796.png)

这一步的目的是将每个source图像的特征通过单应矩阵变换到reference的坐标系，不过这里实际上是$x_i=H_i(d)x_1$，也就是先从reference图像映射到source坐标系下，然后通过逆向映射得到。

在这一步之后，我们可以得到N个[H/4,W/4,D,32]的代价体。其中D是指D个深度平面。这个代价体里面记载的是：这么多像素，在32个特征下，D个深度的每个像素的匹配相似度。

### 代价度量

这里选择使用方差作为代价度量。方差可以直接测量特征差异，比如某个地方被认为两个不同的深度都是正确的，此处方差就会增大，这可以为后续3D CNN提供信息。此处将N个代价体合为一个。

### 代价体正则化

代价体正则化用一个类3D U-Net，让代价体过这个网络，聚合32通道的特征，得到每个像素在不同深度D下的概率。

### Soft Argmin

![image-20260318215411213](./MVS_presentation.assets/image-20260318215411213.png)

沿着整条射线，从最近的深度平面到最远的平面的值求期望。

这里没有使用argmax找最大值，因为：这会导致3D重建的结果出现阶梯；同时由于其不可导，不能端到端训练神经网络。

这一步可以使最终的输出为[H/4,W/4,1]，也就是输出参考图的深度图。

不过虽然求期望能实现亚像素精度并且回传梯度，但是如果遇到遮挡或者反光，概率分布变成双峰，那么求期望可能会把深度算到双峰中间。这一步在后续会解决。

### 深度图细化

