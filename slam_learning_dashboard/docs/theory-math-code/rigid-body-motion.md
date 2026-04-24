# 三维空间刚体运动：从物理位姿到代码变换

## 1. 物理概念

相机在空间中的运动可以拆成两部分：
- 旋转：机体朝向变化
- 平移：机体位置变化

SLAM 中最基本的问题是：已知同一个三维点在不同坐标系中的表示，如何相互转换。

## 2. 数学公式

设点 $\mathbf{p}$ 在坐标系 1 下的坐标为 $\mathbf{p}_1$，在坐标系 2 下为 $\mathbf{p}_2$，则：

$$
\mathbf{p}_2 = \mathbf{R}_{21}\mathbf{p}_1 + \mathbf{t}_{21}
$$

写成齐次形式：

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
\end{bmatrix}, \quad
\mathbf{T}_{21}=\begin{bmatrix}
\mathbf{R}_{21} & \mathbf{t}_{21} \\
\mathbf{0}^\top & 1
\end{bmatrix}
$$

其中：
- $\mathbf{R}_{21} \in SO(3)$
- $\mathbf{t}_{21} \in \mathbb{R}^3$
- $\mathbf{T}_{21} \in SE(3)$

## 3. 代码映射

本地示例文件：
- `code/ch3/useGeometry/useGeometry.cpp`

关键映射：
1. 旋转向量 -> 旋转矩阵
- 公式：$\mathbf{R}=\exp(\boldsymbol{\phi}^\wedge)$
- 代码：第 16 行构造 `AngleAxisd`，第 22 行转 `toRotationMatrix()`。

2. 构造 SE(3) 变换
- 公式：$\mathbf{T}=[\mathbf{R},\mathbf{t}]$
- 代码：第 39 行 `Isometry3d::Identity()`，第 40-41 行分别 `rotate` 与 `pretranslate`。

3. 点坐标变换
- 公式：$\mathbf{p}' = \mathbf{R}\mathbf{p}+\mathbf{t}$
- 代码：第 45 行 `Vector3d v_transformed = T * v;`。

## 4. 对学习者的工程提醒

1. 任何坐标变换先写清楚“从哪个系到哪个系”，避免把 $\mathbf{T}_{cw}$ 与 $\mathbf{T}_{wc}$ 混用。
2. 代码里统一使用 `T_ab * p_b = p_a` 这样的命名约定，能极大减少调试时间。
