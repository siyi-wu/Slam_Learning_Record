# 第5章理论：相机模型与成像几何

## 1. 章节主线

第5章把“空间几何”落到“像素观测”上。  
SLAM里绝大多数误差最终都体现在像素重投影误差。

## 2. 针孔模型

世界点先变到相机坐标系，再投影到像素平面：

$$
\mathbf{p}_c = \mathbf{R}\mathbf{p}_w + \mathbf{t}
$$

$$
\begin{aligned}
u &= f_x \frac{X}{Z} + c_x \\
v &= f_y \frac{Y}{Z} + c_y
\end{aligned}
$$

紧凑写法：

$$
\mathbf{u} = \pi(\mathbf{T}_{cw}\mathbf{P}_w)
$$

## 3. 畸变模型（常用）

归一化平面点 $(x,y)$，$r^2=x^2+y^2$。  
典型径向 + 切向畸变：

$$
\begin{aligned}
x_d &= x(1+k_1r^2+k_2r^4+k_3r^6)+2p_1xy+p_2(r^2+2x^2)\\
y_d &= y(1+k_1r^2+k_2r^4+k_3r^6)+p_1(r^2+2y^2)+2p_2xy
\end{aligned}
$$

这一步决定了“像素是否可信”，是前端匹配和后端优化精度的地基。

## 4. 重投影误差

$$
\mathbf{e} = \mathbf{u}_{obs} - \pi(\mathbf{T}_{cw}\mathbf{P}_w)
$$

BA / PnP / 位姿优化都在最小化这类误差，只是变量不同（只估位姿，或位姿+地图点一起估）。

## 5. 代码映射

1. `code/ch5/imageBasics/imageBasics.cpp`
2. `code/ch5/imageBasics/undistortImage.cpp`
3. `code/ch5/stereoVIsion/stereoVision.cpp`
4. `code/ch5/rgbd/joinMap.cpp`

对应关系：
1. `undistortImage.cpp`：畸变参数如何影响像素坐标。
2. `stereoVision.cpp`：双目视差到深度的基础流程。
3. `joinMap.cpp`：RGB-D 点云融合与坐标变换。

## 6. 实战提醒

1. 不同分辨率下内参要同步缩放，不能直接照搬。
2. 标定质量差会直接导致后续轨迹飘和重建扭曲。
3. 先检查重投影误差统计，再调更复杂的优化参数。

## 7. 网络资料（精选）

1. OpenCV 相机标定教程（官方）：[Camera calibration With OpenCV](https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html)
2. OpenCV Python 标定教程（含畸变公式，官方）：[Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
3. 《视觉SLAM十四讲》第一版代码（ch5 对应 OpenCV）：[gaoxiang12/slambook](https://github.com/gaoxiang12/slambook)
4. 《视觉SLAM十四讲》第二版代码（ch5 目录）：[gaoxiang12/slambook2](https://github.com/gaoxiang12/slambook2)
