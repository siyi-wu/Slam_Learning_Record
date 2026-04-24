# 非线性优化：从残差定义到雅可比实现

## 1. 物理概念

SLAM 后端的目标是“找一组状态，使观测误差最小”。  
例如 PnP 中，状态是相机位姿，误差是投影点与观测点的像素偏差。

## 2. 数学公式

以 3D-2D 重投影为例，目标函数：

$$
\min_{\mathbf{T}} \sum_i \|\mathbf{e}_i\|^2,
\quad
\mathbf{e}_i = \mathbf{u}_i - \pi(\mathbf{T}\mathbf{P}_i)
$$

Gauss-Newton 线性化：

$$
\mathbf{e}(\delta\boldsymbol{\xi}) \approx \mathbf{e} + \mathbf{J}\delta\boldsymbol{\xi}
$$

法方程：

$$
\mathbf{H}\delta\boldsymbol{\xi} = \mathbf{b},
\quad
\mathbf{H}=\sum_i \mathbf{J}_i^\top\mathbf{J}_i,
\quad
\mathbf{b}=-\sum_i \mathbf{J}_i^\top\mathbf{e}_i
$$

更新：

$$
\mathbf{T} \leftarrow \exp(\delta\boldsymbol{\xi}^\wedge)\mathbf{T}
$$

## 3. 代码映射 I：手写 Gauss-Newton

文件：`code/ch7/pose_estimation_3d2d.cpp`

1. 残差定义
- 公式：$\mathbf{e}_i = \mathbf{u}_i - \hat{\mathbf{u}}_i$
- 代码：第 195-198 行。

2. 雅可比矩阵 $\mathbf{J}$
- 代码：第 200-212 行直接给出解析雅可比。

3. 构建法方程
- 代码：第 214-215 行累计 `H` 和 `b`。

4. 解线性方程
- 代码：第 218-219 行 `dx = H.ldlt().solve(b)`。

5. 李群更新
- 代码：第 233 行 `pose = Sophus::SE3d::exp(dx) * pose;`

## 4. 代码映射 II：g2o 中的同一件事

同文件第 267-359 行：
- `computeError()`（第 273-279 行）对应残差计算。
- `linearizeOplus()`（第 281-296 行）对应雅可比计算。
- `optimizer.optimize(10)`（第 353 行）对应迭代优化。

也就是：你手写 GN 的每一步，g2o 都以“顶点 + 边”框架化封装了。

## 5. 代码映射 III：直接法中的链式法则

文件：`code/ch8/direct_method.cpp`

直接法残差：

$$
e = I_1(\mathbf{u}) - I_2(\pi(\mathbf{T}\mathbf{P}))
$$

链式求导：

$$
\frac{\partial e}{\partial \boldsymbol{\xi}} =
\frac{\partial e}{\partial I}
\frac{\partial I}{\partial \mathbf{u}}
\frac{\partial \mathbf{u}}{\partial \boldsymbol{\xi}}
$$

代码对应：
- $\frac{\partial \mathbf{u}}{\partial \boldsymbol{\xi}}$：第 258-270 行 `J_pixel_xi`
- $\frac{\partial I}{\partial \mathbf{u}}$：第 272-275 行 `J_img_pixel`
- 合成雅可比：第 278 行 `J = -1.0 * (...)`
- 正规方程累计：第 280-282 行

## 6. 实战提醒

1. 优化不收敛先查三件事：雅可比符号、更新方向、尺度归一化。
2. 手写雅可比先做数值梯度对拍，再接入 g2o/Ceres。
3. 观察 `cost` 是否单调下降（本地代码第 226-230 行已有保护逻辑）。
