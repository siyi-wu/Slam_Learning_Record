# 第4章理论：李群与李代数

## 1. 为什么要引入李群李代数

旋转矩阵与位姿矩阵有约束（正交、行列式为 1），直接优化不方便。  
李群 / 李代数提供了：
1. 最小参数化（局部 3 维或 6 维增量）。
2. 指数映射与对数映射（群和代数之间可互转）。
3. 适合迭代优化的“扰动更新”框架。

## 2. 基本集合

$$
SO(3)=\{\mathbf{R}\in\mathbb{R}^{3\times3}\mid \mathbf{R}\mathbf{R}^\top=\mathbf{I},\det(\mathbf{R})=1\}
$$

$$
SE(3)=
\left\{
\mathbf{T}=
\begin{bmatrix}
\mathbf{R}&\mathbf{t}\\
\mathbf{0}^\top&1
\end{bmatrix}
\mid \mathbf{R}\in SO(3), \mathbf{t}\in\mathbb{R}^3
\right\}
$$

指数映射：

$$
\mathbf{R}=\exp(\boldsymbol{\phi}^{\wedge}),\quad
\mathbf{T}=\exp(\boldsymbol{\xi}^{\wedge})
$$

优化中的左扰动更新常写成：

$$
\mathbf{T}\leftarrow \exp(\delta\boldsymbol{\xi}^{\wedge})\mathbf{T}
$$

## 3. 对SLAM优化的意义

1. 误差在切空间中线性化，雅可比推导更自然。
2. 每次只求一个小增量 $\delta\boldsymbol{\xi}$，避免直接在约束空间硬优化。
3. 与 BA、位姿图优化天然兼容。

## 4. 代码映射

1. `code/ch4/useSophus.cpp`
2. `code/ch4/example/trajectoryError.cpp`
3. `code/ch4/example/groundtruth.txt`
4. `code/ch4/example/estimated.txt`

对应关系：
1. `useSophus.cpp`：`SO3/SE3` 的构造、`exp/log` 与李代数操作。
2. `trajectoryError.cpp`：姿态误差的几何计算与评估思路。

## 5. 实战提醒

1. 明确左扰动还是右扰动，二者雅可比表达不同。
2. 避免大角度一次更新，必要时做多次小步迭代。
3. 注意 `so3` / `se3` 的向量排列约定（旋转在前还是平移在前）。

## 6. 深入阅读

1. [李群与李代数（专题版）](./lie-group-lie-algebra.md)

## 7. 网络资料（精选）

1. Sophus 官方仓库（Lie Group 实现）：[strasdat/Sophus](https://github.com/strasdat/Sophus)
2. 《视觉SLAM十四讲》第一版代码（ch4 对应 Sophus）：[gaoxiang12/slambook](https://github.com/gaoxiang12/slambook)
3. 《视觉SLAM十四讲》第二版代码（ch4 目录）：[gaoxiang12/slambook2](https://github.com/gaoxiang12/slambook2)
