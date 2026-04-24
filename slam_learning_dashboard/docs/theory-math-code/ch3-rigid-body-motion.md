# 第3章理论：三维刚体运动

## 1. 物理概念

相机运动由旋转与平移组成，SLAM 中大量问题都在做坐标系变换。

## 2. 数学公式

$$
\mathbf{p}_2 = \mathbf{R}_{21}\mathbf{p}_1 + \mathbf{t}_{21}
$$

齐次形式：

$$
\begin{bmatrix}
\mathbf{p}_2 \\
1
\end{bmatrix}
=
\mathbf{T}_{21}
\begin{bmatrix}
\mathbf{p}_1 \\
1
\end{bmatrix},\quad
\mathbf{T}_{21}\in SE(3)
$$

## 3. 代码映射

- `code/ch3/useGeometry/useGeometry.cpp`
- `code/ch3/examples/coordinateTransform.cpp`

## 4. 深入阅读

- [三维空间刚体运动（专题版）](./rigid-body-motion.md)
