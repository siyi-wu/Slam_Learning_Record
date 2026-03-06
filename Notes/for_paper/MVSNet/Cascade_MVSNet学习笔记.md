# Cascade Cost Volume学习笔记

基本信息：

Cascade Cost Volume for High-Resolution Multi-View Stereo  and Stereo Matching

作者：Xiaodong Gu, Zhiwen Fan, Zuozhuo Dai, Siyu Zhu,  Feitong Tan, Ping Tan

Alibaba A.I. Labs, Simon Fraser University

原文：[Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching](https://arxiv.org/abs/1912.06378)

---

## 总述

### 摘要

* 在MVSNet中，算法需要构建3D代价体，（相机的视锥体上的很多平面作为长宽之后的第三个维度）
* 因此cost volume在生成高分辨率时会收到限制，因为开销会呈立方级增长
* 主要贡献是提出一种**兼具内存和时间效率的代价体积构建方法**，作为现有的基于3D代价体积的多视图立体视觉和立体匹配方法的补充
* **方法**：coarse and fine：由粗到细搜索

### 结论

**主要贡献**：将原本单一的代价体分解为包含多个阶段的级联形式；然后通过利用前一阶段生成的深度图，缩小当前阶段的深度搜索范围，减少需要计算的深度假设平面的总数。接着使用更高空间分辨率的代价体生成包含更多精细细节的输出。

- 是对现有所有基于3D代价体的MVS和立体匹配算法的补充

### 总结

- **分解 (Decompose)：** 打破单次构建庞大 3D 体积的传统，将其拆解为多个级联阶段。
- **缩减 (Narrow & Reduce)：** 利用上一阶段的粗略深度图，大幅缩小当前阶段的深度搜索区间（Z轴），从而成倍减少需要计算的“假设平面”数量。
- **提质 (Finer Details)：** 把从深度搜索域省下来的显存，全部投资到 2D 空间分辨率（X和Y轴）上，从而生成边缘极其锐利的精细 3D 模型。
