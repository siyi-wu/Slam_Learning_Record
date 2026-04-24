# 第12章理论：稠密重建与地图表示

## 1. 物理概念

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

## 3. 代码映射

文件：
- `code/ch12/dense_RGBD/pointcloud_mapping.cpp`
- `code/ch12/dense_RGBD/surfel_mapping.cpp`
- `code/ch12/dense_RGBD/octomap_mapping.cpp`
- `code/ch12/dense_mono/dense_mapping.cpp`

对应关系：
1. `pointcloud_mapping` 对应点级融合与体素降采样。
2. `surfel_mapping` 对应面元表示，强调法向和局部表面质量。
3. `octomap_mapping` 对应占据概率地图表示。

## 4. 实战提醒

1. 先做坐标系统一，再做滤波和可视化。
2. 体素大小决定“精度-内存-速度”三方平衡。
3. 稠密地图用于感知/导航时，要加入动态物体抑制策略。
