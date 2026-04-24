# 对极几何：从两帧约束到三角化重建

## 1. 物理概念

两台相机（或同一相机两时刻）观测同一空间点时，匹配点不是任意关系，而要满足对极约束。  
这个约束让我们可以：
- 估计相对位姿（`R, t`）
- 进行三角化恢复 3D 点

## 2. 数学公式

归一化相机坐标下，对极约束：

$$
\mathbf{y}_2^\top \mathbf{E} \mathbf{y}_1 = 0,
\quad \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}
$$

其中：

$$
[\mathbf{t}]_\times =
\begin{bmatrix}
0 & -t_z & t_y \\
t_z & 0 & -t_x \\
-t_y & t_x & 0
\end{bmatrix}
$$

三角化本质是求两条观测光线的交点（在噪声下为最小二乘意义下的最接近点）。

## 3. 代码映射 I：对极约束验证

文件：`code/ch7/pose_estimation_2d2d.cpp`

1. 构造反对称矩阵 $[\mathbf{t}]_\times$
- 代码：第 51-54 行 `t_x`。

2. 逐匹配点验证 $\mathbf{y}_2^\top[\mathbf{t}]_\times\mathbf{R}\mathbf{y}_1$
- 代码：第 60-66 行。
- 对应公式：理想情况下结果接近 0。

3. 本质矩阵与位姿恢复
- `findEssentialMat`：第 152 行
- `recoverPose`：第 163 行

## 4. 代码映射 II：三角化

文件：`code/ch7/triangulation.cpp`

1. 投影矩阵构造
- 第 165-173 行构造 $\mathbf{P}_1, \mathbf{P}_2$。

2. 调用三角化
- 第 184 行 `cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);`

3. 齐次坐标转非齐次坐标
- 第 188-190 行 `x /= x.at<float>(3, 0)`。

## 5. 和 SLAM 前端的关系

1. 对极几何用于剔除错配、提供两帧几何一致性。
2. 三角化把“视觉观测”转换成“可优化的 3D 地标”。
3. 后续 BA 再联合优化位姿与地标，提高精度。

## 6. 实战提醒

1. 先把像素坐标变成归一化相机坐标再做几何约束，避免内参污染。
2. 纯旋转或小基线场景下三角化深度不稳定，要结合视差和重投影误差筛选。
