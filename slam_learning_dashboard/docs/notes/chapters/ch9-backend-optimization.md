# 第9章专题：后端优化进阶

## 学习目标

1. 理解图优化在 SLAM 全链路中的作用。
2. 能区分 GN / LM / DogLeg 的适用场景。
3. 能把残差块、状态变量、先验约束组织成可扩展后端。

## 核心概念

- 状态向量：位姿、速度、偏置、路标点。
- 约束来源：视觉重投影、IMU 预积分、回环约束。
- 稀疏结构：利用块稀疏 Jacobian 和 Schur 补提高效率。

## 关键公式

目标函数：

$$
\min_{\mathbf{x}} \sum_i \rho_i\left(\|\mathbf{r}_i(\mathbf{x})\|^2_{\mathbf{\Omega}_i}\right)
$$

高斯牛顿线性化：

$$
\mathbf{H}\Delta\mathbf{x}=\mathbf{b},\quad
\mathbf{H}=\sum_i \mathbf{J}_i^T\mathbf{\Omega}_i\mathbf{J}_i
$$

## 与代码的映射模板

- 顶点类：状态定义与增量更新（SE(3) 左乘/右乘）。
- 边类：残差 `computeError()` 与雅可比 `linearizeOplus()`。
- 求解器：线性求解器 + 非线性策略 + 终止条件。

## 本地源码落点（已对齐）

你的学习源码：
- `code/ch9/README.md`
- `code/ch9/bundle_adjustment_g2o.cpp`
- `code/ch9/bundle_adjustment_ceres.cpp`
- `code/ch9/SnavelyReprojectionError.h`

## 实践任务

1. 在 `g2o` 示例中加入鲁棒核，比较外点场景下误差变化。
2. 手写一版数值差分雅可比，对拍解析雅可比正确性。
3. 为后端模块设计统一日志（迭代次数、阻尼系数、cost 曲线）。
