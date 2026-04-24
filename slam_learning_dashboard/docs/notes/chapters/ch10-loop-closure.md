# 第10章专题：回环检测与全局一致性


## 学习目标

1. 理解“漂移”产生原因以及回环的纠偏机制。
2. 掌握词袋回环检测与几何验证的组合流程。
3. 学会把回环边接入位姿图优化。

## 核心概念

- 回环候选：外观相似帧检索（DBoW2/NetVLAD 等）。
- 几何验证：PnP + RANSAC 或 Sim3 验证。
- 位姿图优化：新增回环约束后做全局一致化。

## 关键公式

相似变换约束（尺度漂移场景）：

$$
\mathbf{T}_{ij}^{\text{meas}} \approx \mathbf{T}_i^{-1}\mathbf{T}_j
$$

图优化目标：

$$
\min_{\{\mathbf{T}_k\}} \sum_{(i,j)\in\mathcal{E}} \|\log\left((\mathbf{T}_{ij}^{\text{meas}})^{-1}\mathbf{T}_i^{-1}\mathbf{T}_j\right)\|^2
$$

## 与代码的映射模板

- 回环线程：候选检索 -> 几何验证 -> 发布约束。
- 图优化线程：融合 odom 边与 loop 边并重优化。
- 地图更新：重定位关键帧、更新地图点可见性关系。

## 本地源码落点（已对齐）

你的学习源码：
- `code/ch10/README.md`
- `code/ch10/pose_graph_g2o_SE3.cpp`
- `code/ch10/pose_graph_g2o_lie_algebra.cpp`
- `code/ch10/sphere.g2o`

## 实践任务

1. 给现有 VO 结果注入模拟漂移并验证回环纠偏效果。
2. 比较“仅外观回环”与“外观+几何验证”误检率差异。
