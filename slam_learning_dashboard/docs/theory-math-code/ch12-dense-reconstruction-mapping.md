# 第12章理论：稠密重建与地图表示

## 1. 章节主线

稀疏 SLAM 关注定位与关键结构，稠密重建关注场景几何细节。  
核心问题是把多帧深度统一到同一世界坐标系并做融合压缩。

## 2. 数学公式

像素反投影：

$$
Z = d/s,\quad
X = (u-c_x)Z/f_x,\quad
Y = (v-c_y)Z/f_y
$$

坐标变换与融合：

$$
\mathbf{p}_w = \mathbf{T}_{wc}\mathbf{p}_c,\quad
\mathcal{M} \leftarrow \mathcal{M} \cup \{\mathbf{p}_w\}
$$

## 3. 常见地图表示

1. 点云地图：实现简单，便于可视化，常配合体素滤波压缩。
2. Surfel 地图：每个面元包含法向与半径，局部表面表达更稳定。
3. 占据栅格/OctoMap：显式表达占据、空闲、未知，适合导航规划。
4. TSDF 体素：适合高质量曲面融合（如 KinectFusion 类方法）。

## 4. 代码映射

文件：
- `code/ch12/dense_RGBD/pointcloud_mapping.cpp`
- `code/ch12/dense_RGBD/surfel_mapping.cpp`
- `code/ch12/dense_RGBD/octomap_mapping.cpp`
- `code/ch12/dense_mono/dense_mapping.cpp`

对应关系：
1. `pointcloud_mapping` 对应点级融合与体素降采样。
2. `surfel_mapping` 对应面元表示，强调法向和局部表面质量。
3. `octomap_mapping` 对应占据概率地图表示。

## 5. 实战提醒

1. 先做坐标系统一，再做滤波和可视化。
2. 体素大小决定“精度-内存-速度”三方平衡。
3. 稠密地图用于感知/导航时，要加入动态物体抑制策略。

## 6. 网络资料（精选）

1. OctoMap 官方网站：[octomap.github.io](https://octomap.github.io/)
2. OctoMap 官方仓库：[OctoMap/octomap](https://github.com/OctoMap/octomap)
3. OctoMap 论文 DOI：[10.1007/s10514-012-9321-0](https://doi.org/10.1007/s10514-012-9321-0)
4. PCL 体素滤波文档（`VoxelGrid`）：[pcl::VoxelGrid](https://pointclouds.org/documentation/classpcl_1_1_voxel_grid.html)
5. KinectFusion 论文页面（Microsoft Research）：[KinectFusion](https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-3d-reconstruction-and-interaction-using-a-moving-depth-camera/)
