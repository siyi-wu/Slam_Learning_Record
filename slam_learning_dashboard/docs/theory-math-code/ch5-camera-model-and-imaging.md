# 第5章理论：相机模型与成像几何

## 1. 物理概念

相机把三维世界投影到二维图像，SLAM 观测本质上是“投影后的像素约束”。

## 2. 数学公式

针孔模型：

$$
\begin{aligned}
\mathbf{p}_c &= \mathbf{R}\mathbf{p}_w + \mathbf{t} \\
u &= f_x X/Z + c_x \\
v &= f_y Y/Z + c_y
\end{aligned}
$$

重投影误差：

$$
\mathbf{e} = \mathbf{u}_{obs} - \pi(\mathbf{T}_{cw}\mathbf{P}_w)
$$

## 3. 代码映射

- `code/ch5/imageBasics/undistortImage.cpp`
- `code/ch5/stereoVIsion/stereoVision.cpp`
- `code/ch5/rgbd/joinMap.cpp`

## 4. 实战提醒

1. 相机内参与畸变模型必须先标定再做几何估计。
2. 坐标系和单位要在整个工程中保持一致。
