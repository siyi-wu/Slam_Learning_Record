# 第8章理论：光流与直接法

## 1. 章节主线

第8章把“特征点几何”扩展到“像素强度优化”。
直接法不显式提取特征，而是直接最小化灰度误差。

## 2. 数学公式

光度误差：

$$
e = I_1(\mathbf{u}) - I_2(\pi(\mathbf{T}\mathbf{P}))
$$

链式求导：

$$
\frac{\partial e}{\partial \boldsymbol{\xi}}=
\frac{\partial e}{\partial I}
\frac{\partial I}{\partial \mathbf{u}}
\frac{\partial \mathbf{u}}{\partial \boldsymbol{\xi}}
$$

## 3. 稀疏光流与直接法的关系

1. 光流法：局部块内假设亮度一致，估计像素位移。
2. 直接法 VO：把像素位移和相机位姿联系起来，直接优化位姿。
3. 二者共用“光度一致性 + 金字塔 + 迭代线性化”的基本框架。

## 4. 代码映射

1. `code/ch8/optical_flow.cpp`
2. `code/ch8/direct_method.cpp`
3. `code/ch8/CMakeLists.txt`

对应关系：
1. `optical_flow.cpp`：Lucas-Kanade 稀疏光流迭代求解。
2. `direct_method.cpp`：直接法位姿估计与光度误差最小化。

## 5. 实战提醒

1. 直接法对光照变化敏感，建议做金字塔与鲁棒核处理。
2. 像素梯度质量会直接影响收敛稳定性。
3. 初值很关键，通常结合上一帧位姿或IMU预测提高收敛域。

## 6. 网络资料（精选）

1. OpenCV 光流主文档（`calcOpticalFlowPyrLK`）：[video track module](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
2. OpenCV 光流教程（Lucas-Kanade）：[tutorial_optical_flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
3. 《视觉SLAM十四讲》第一版代码（ch8）：[gaoxiang12/slambook](https://github.com/gaoxiang12/slambook)
4. 《视觉SLAM十四讲》第二版代码（ch8）：[gaoxiang12/slambook2](https://github.com/gaoxiang12/slambook2)
