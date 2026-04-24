# 第12章专题：稠密重建与点云地图


## 学习目标

1. 理解稀疏地图与稠密地图在系统中的角色差异。
2. 掌握从深度/视差到点云融合的关键步骤。
3. 学会使用 PCL 进行点云滤波与地图压缩。

## 核心概念

- 深度融合：多帧深度在统一世界坐标系下融合。
- 点云降采样：体素滤波平衡精度与内存开销。
- 地图表示：点云、TSDF、ESDF 等。

## 关键公式

像素到相机坐标：

$$
Z = d/s,\quad
X = (u-c_x)Z/f_x,\quad
Y = (v-c_y)Z/f_y
$$

相机到世界坐标：

$$
\mathbf{p}_w = \mathbf{T}_{wc}\mathbf{p}_c
$$

## 与代码的映射模板

- 输入：RGB-D 或双目深度。
- 处理：反投影 -> 位姿变换 -> 点云拼接 -> VoxelGrid。
- 输出：`map.pcd` 或可视化窗口。

## 本地源码落点（已对齐）

你的学习源码：
- `code/ch12/README.md`
- `code/ch12/dense_RGBD/pointcloud_mapping.cpp`
- `code/ch12/dense_RGBD/surfel_mapping.cpp`
- `code/ch12/dense_RGBD/octomap_mapping.cpp`
- `code/ch12/dense_mono/dense_mapping.cpp`

## 实践任务

1. 对不同体素大小（1cm, 2cm, 5cm）比较地图大小与细节保真。
2. 增加统计面板：点数、内存占用、融合帧数。
