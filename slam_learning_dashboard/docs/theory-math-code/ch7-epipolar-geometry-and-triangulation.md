# 第7章理论：对极几何与三角化

## 1. 章节主线

本章解决“仅靠两帧图像能恢复什么几何信息”：
1. 2D-2D 对极约束估计相对运动（方向 + 旋转）。
2. 在已知位姿与内参后，把匹配点恢复成 3D（Triangulation）。
3. 进一步连接到 3D-2D（PnP）和 3D-3D（ICP）位姿估计。

## 2. 数学公式

本质矩阵约束：

$$
\mathbf{y}_2^\top \mathbf{E} \mathbf{y}_1 = 0,\quad
\mathbf{E}=[\mathbf{t}]_\times\mathbf{R}
$$

其中 $\mathbf{y}_1,\mathbf{y}_2$ 是归一化平面点。  
由本质矩阵分解得到 $(\mathbf{R}, \mathbf{t})$ 时，平移只确定方向，尺度仍不确定（单目尺度歧义）。

三角化可理解为“噪声下两条观测射线的最优交点”：

$$
\mathbf{P}_w = \arg\min_{\mathbf{P}} \sum_{i=1}^{2}
\left\|\mathbf{u}_i-\pi(\mathbf{T}_{ciw}\mathbf{P})\right\|^2
$$

## 3. 常见估计链路

1. 2D-2D：`findEssentialMat + recoverPose`，适合初始化和短基线相对位姿估计。
2. 3D-2D：PnP（DLT/P3P/EPnP + BA），用于已知地图点时的重定位和跟踪。
3. 3D-3D：SVD/ICP，适合已有深度或点云对齐。

## 4. 代码映射

1. `code/ch7/pose_estimation_2d2d.cpp`
2. `code/ch7/triangulation.cpp`
3. `code/ch7/pose_estimation_3d2d.cpp`
4. `code/ch7/pose_estimation_3d3d.cpp`

对应关系：
1. `pose_estimation_2d2d.cpp`：对极几何 + 姿态恢复。
2. `triangulation.cpp`：双视图匹配点恢复三维点。
3. `pose_estimation_3d2d.cpp`：PnP 与重投影约束。
4. `pose_estimation_3d3d.cpp`：点云配准（刚体变换）。

## 5. 实战提醒

1. 对极几何对匹配外点很敏感，先做好 RANSAC 内点筛选。
2. 小视差或纯旋转场景下三角化会退化，初始化需谨慎。
3. 任何位姿估计都建议用重投影误差做最终健康度检查。

## 6. 深入阅读

1. [对极几何（专题版）](./epipolar-geometry.md)

## 7. 网络资料（精选）

1. OpenCV `findEssentialMat/recoverPose` 官方文档：[calib3d](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
2. OpenCV `solvePnP` 说明页（PnP族方法）：[solvePnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
3. 《视觉SLAM十四讲》第一版代码（ch7）：[gaoxiang12/slambook](https://github.com/gaoxiang12/slambook)
4. 《视觉SLAM十四讲》第二版代码（ch7）：[gaoxiang12/slambook2](https://github.com/gaoxiang12/slambook2)
