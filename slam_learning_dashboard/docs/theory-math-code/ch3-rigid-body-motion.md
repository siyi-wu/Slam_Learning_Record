# 第3章理论：三维刚体运动

## 1. 核心任务

这一章回答两个问题：
1. 如何表示三维空间中的姿态与位姿。
2. 如何在不同坐标系之间稳定、无歧义地做变换。

SLAM 的大量计算都可归结为“点从坐标系 A 变到坐标系 B”。

## 2. 表示方法与关系

点坐标变换：

$$
\mathbf{p}_2 = \mathbf{R}_{21}\mathbf{p}_1 + \mathbf{t}_{21}
$$

齐次坐标形式：

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
\mathbf{T}_{21}=
\begin{bmatrix}
\mathbf{R}_{21} & \mathbf{t}_{21}\\
\mathbf{0}^\top & 1
\end{bmatrix}\in SE(3)
$$

常见姿态表示优缺点：
1. 旋转矩阵：直观、可复合，但冗余（9参+正交约束）。
2. 欧拉角：可读性好，但可能万向锁。
3. 四元数：紧凑且数值稳定，工程最常用。
4. 轴角 / 旋转向量：适合做小量更新与优化。

## 3. 工程里的关键约束

1. 坐标系命名必须统一，例如 `T_cw`（world 到 camera）或 `T_wc`（camera 到 world）要始终一致。
2. 单位必须统一（米、像素、弧度）。
3. 旋转矩阵应保持正交，数值计算后常需重正交化或归一化四元数。

## 4. 代码映射

1. `code/ch3/useEigen/eigenMatrix.cpp`
2. `code/ch3/useGeometry/useGeometry.cpp`
3. `code/ch3/examples/coordinateTransform.cpp`
4. `code/ch3/examples/plotTrajectory.cpp`

对应关系：
1. `eigenMatrix.cpp`：矩阵、向量与线性代数基础。
2. `useGeometry.cpp`：几何类（旋转、四元数、变换）与互转。
3. `coordinateTransform.cpp`：点在不同坐标系之间的变换流程。

## 5. 实战提醒

1. 调位姿错误时，先排查坐标系方向，再看公式实现。
2. 组合位姿时明确左乘还是右乘，和你定义的参考系一致。
3. 四元数每次更新后做归一化，防止长序列漂移。

## 6. 网络资料（精选）

1. Eigen 几何模块教程（官方）：[Eigen Geometry Tutorial](https://www.eigen.tuxfamily.org/dox-devel/group__TutorialGeometry.html)
2. 《视觉SLAM十四讲》第一版代码（含 ch3 示例）：[gaoxiang12/slambook](https://github.com/gaoxiang12/slambook)
3. 《视觉SLAM十四讲》第二版代码（含 ch3 目录）：[gaoxiang12/slambook2](https://github.com/gaoxiang12/slambook2)
4. ORB-SLAM2 论文（工程系统视角）：[DOI:10.1109/TRO.2017.2705103](https://doi.org/10.1109/TRO.2017.2705103)
