# 第13章专题：视觉惯导融合（VIO）


## 学习目标

1. 理解纯视觉系统在快速运动/弱纹理下的失效模式。
2. 掌握 IMU 预积分与状态扩维思想。
3. 学会把视觉残差与惯性残差统一到同一优化框架。

## 核心概念

- IMU 预积分：在两关键帧间压缩高频惯导测量。
- 状态变量扩展：位姿、速度、偏置联合估计。
- 时间同步与外参标定：VIO 精度的关键前提。

## 关键公式

连续时间模型离散化后，状态递推可写为：

$$
\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k, \mathbf{n}_k)
$$

优化目标示意：

$$
\min \sum \|\mathbf{r}_{\text{vision}}\|^2 + \sum \|\mathbf{r}_{\text{imu}}\|^2 + \sum \|\mathbf{r}_{\text{prior}}\|^2
$$

## 与代码的映射模板

- 预积分模块：缓存 IMU，形成帧间约束。
- 前端：视觉跟踪给初值。
- 后端：滑窗优化联合估计状态。

## 本地源码落点（已对齐）

你的学习源码：
- `code/ch13/README.md`
- `code/ch13/src/frontend.cpp`
- `code/ch13/src/backend.cpp`
- `code/ch13/src/map.cpp`
- `code/ch13/src/visual_odometry.cpp`
- `code/ch13/app/run_kitti_stereo.cpp`

## 实践任务

1. 先在仿真数据上验证 bias 漂移补偿是否生效。
2. 增加时间偏移参数，观察对轨迹误差的影响。
