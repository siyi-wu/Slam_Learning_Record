# 第13章理论：视觉惯导融合

## 1. 物理概念

视觉提供几何约束，IMU 提供高频运动先验。  
VIO 的关键是把两类信息在统一状态中融合，并处理偏置与时间同步误差。

## 2. 数学公式

状态向量常写为：

$$
\mathbf{x} = [\mathbf{R},\mathbf{p},\mathbf{v},\mathbf{b}_g,\mathbf{b}_a]
$$

滑窗目标函数：

$$
\min_{\mathbf{x}} \sum \|\mathbf{r}_{vision}\|^2 + \sum \|\mathbf{r}_{imu}\|^2 + \sum \|\mathbf{r}_{prior}\|^2
$$

IMU 预积分残差由姿态、速度、位置三部分组成，并显式依赖陀螺/加计偏置。

## 3. 代码映射

文件：
- `code/ch13/src/frontend.cpp`
- `code/ch13/src/backend.cpp`
- `code/ch13/src/visual_odometry.cpp`
- `code/ch13/app/run_kitti_stereo.cpp`

对应关系：
1. 前端提供视觉跟踪结果与关键帧触发。
2. 后端负责滑窗优化和状态更新。
3. 系统主流程中，传感器数据按时间顺序驱动状态传播和校正。

## 4. 实战提醒

1. 偏置建模不准确会直接导致尺度和姿态漂移。
2. 时间戳对齐问题通常比优化器本身更致命。
3. 调参顺序建议：外参/时间同步 -> IMU 噪声 -> 视觉鲁棒核。
