# 第4章理论：李群与李代数

## 1. 物理概念

李群/李代数用于给旋转与位姿提供最小参数化和局部线性化工具，是优化中的核心语言。

## 2. 数学公式

$$
\mathbf{R} = \exp(\boldsymbol{\phi}^\wedge),\quad
\mathbf{T} = \exp(\boldsymbol{\xi}^\wedge)
$$

扰动更新常写为：

$$
\mathbf{T} \leftarrow \exp(\delta\boldsymbol{\xi}^\wedge)\mathbf{T}
$$

## 3. 代码映射

- `code/ch4/useSophus.cpp`
- `code/ch7/pose_estimation_3d2d.cpp`

## 4. 深入阅读

- [李群与李代数（专题版）](./lie-group-lie-algebra.md)
