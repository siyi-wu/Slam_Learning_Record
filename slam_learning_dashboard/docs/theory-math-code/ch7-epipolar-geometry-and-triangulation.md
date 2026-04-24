# 第7章理论：对极几何与三角化

## 1. 物理概念

双视图中，同一空间点的匹配必须满足对极约束；由此可估计位姿并恢复三维点。

## 2. 数学公式

$$
\mathbf{y}_2^\top \mathbf{E} \mathbf{y}_1 = 0,\quad
\mathbf{E}=[\mathbf{t}]_\times\mathbf{R}
$$

三角化可视为在噪声下求两条观测光线的最优交点。

## 3. 代码映射

- `code/ch7/pose_estimation_2d2d.cpp`
- `code/ch7/triangulation.cpp`
- `code/ch7/pose_estimation_3d2d.cpp`

## 4. 深入阅读

- [对极几何（专题版）](./epipolar-geometry.md)
