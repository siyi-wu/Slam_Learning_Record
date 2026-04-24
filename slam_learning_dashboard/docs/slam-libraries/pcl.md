# PCL：点云处理与三维重建

## 对齐级别
- ：全书后续常用方向，你本地 `code` 目前暂未出现 PCL 示例。

## 1. 核心数据结构

1. 点类型：
- `pcl::PointXYZ`
- `pcl::PointXYZRGB`
- `pcl::PointNormal`

2. 点云容器：
- `pcl::PointCloud<T>`
- 常用指针别名：`pcl::PointCloud<T>::Ptr`

3. 常见处理模块：
- 滤波：`pcl::VoxelGrid`
- 配准：`pcl::IterativeClosestPoint` (ICP)
- 法线估计：`pcl::NormalEstimation`

## 2. 适用范围

PCL 在 SLAM 中常用于：
- 稠密/半稠密点云构建
- 点云降采样与去噪
- 点云配准（里程计初值后的精配准）
- 三维地图可视化与保存

## 3. 典型 API 范例

### 3.1 创建与保存点云

```cpp
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int main() {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    // 添加一个点（示例）
    pcl::PointXYZRGB p;
    p.x = 1.0f; p.y = 2.0f; p.z = 3.0f;
    p.r = 255; p.g = 0; p.b = 0;
    cloud->points.push_back(p);

    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = false;

    pcl::io::savePCDFileBinary("map.pcd", *cloud);
    return 0;
}
```

### 3.2 体素降采样

```cpp
#include <pcl/filters/voxel_grid.h>

void Downsample(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& in,
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr& out) {
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel;
    voxel.setInputCloud(in);

    // 体素大小越大，点云越稀疏
    voxel.setLeafSize(0.02f, 0.02f, 0.02f);
    voxel.filter(*out);
}
```

### 3.3 ICP 配准

```cpp
#include <pcl/registration/icp.h>

void RunICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr& src,
            const pcl::PointCloud<pcl::PointXYZ>::Ptr& tgt) {
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(src);
    icp.setInputTarget(tgt);

    pcl::PointCloud<pcl::PointXYZ> aligned;
    icp.align(aligned);

    if (icp.hasConverged()) {
        // 返回 4x4 刚体变换
        Eigen::Matrix4f T = icp.getFinalTransformation();
        (void)T;
    }
}
```

## 4. 常见坑与建议

1. 点云坐标系必须与相机/世界坐标系约定一致。
2. ICP 依赖初值，建议先用视觉里程计提供初始变换。
3. 降采样与法线估计参数要配套调节，否则会损失几何细节。
