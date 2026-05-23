//                                     // 文件头注释开始
// Created by gaoxiang on 19-4-25.     // 记录原作者和创建日期
//                                     // 文件头注释结束

#include <pcl/point_cloud.h>             // 引入 PCL 点云容器定义
#include <pcl/point_types.h>             // 引入 PCL 点类型定义，例如 PointXYZRGB 和 PointXYZRGBNormal
#include <pcl/io/pcd_io.h>               // 引入 PCL PCD 文件读写接口
#include <pcl/visualization/pcl_visualizer.h> // 引入 PCL 可视化器，用于显示重建网格
#include <pcl/kdtree/kdtree_flann.h>      // 引入 KdTree 搜索结构，用于邻域查询
#include <pcl/surface/surfel_smoothing.h> // 引入 Surfel 平滑相关接口
#include <pcl/surface/mls.h>              // 引入移动最小二乘 MLS 表面重建接口
#include <pcl/surface/gp3.h>              // 引入贪心投影三角化 GP3 接口
#include <pcl/surface/impl/mls.hpp>       // 引入 MLS 模板实现，保证模板实例化可用

// typedefs
typedef pcl::PointXYZRGB PointT;          // 定义输入点类型：带 RGB 颜色的三维点
typedef pcl::PointCloud<PointT> PointCloud; // 定义输入点云类型
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr; // 定义输入点云智能指针类型
typedef pcl::PointXYZRGBNormal SurfelT;   // 定义 Surfel 点类型：包含位置、颜色和法线
typedef pcl::PointCloud<SurfelT> SurfelCloud; // 定义 Surfel 点云类型
typedef pcl::PointCloud<SurfelT>::Ptr SurfelCloudPtr; // 定义 Surfel 点云智能指针类型

SurfelCloudPtr reconstructSurface(
        const PointCloudPtr &input, float radius, int polynomial_order) { // 根据输入点云估计平滑 Surfel 表面
    pcl::MovingLeastSquares<PointT, SurfelT> mls; // 创建移动最小二乘对象，输入 XYZRGB，输出带法线 Surfel
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>); // 创建邻域搜索用的 KdTree
    mls.setSearchMethod(tree);          // 设置 MLS 使用的邻域搜索结构
    mls.setSearchRadius(radius);        // 设置 MLS 搜索半径
    mls.setComputeNormals(true);        // 要求 MLS 同时估计每个点的法线
    mls.setSqrGaussParam(radius * radius); // 设置高斯权重参数，通常取搜索半径平方
    mls.setPolynomialOrder(polynomial_order); // 设置局部曲面拟合的多项式阶数
    mls.setInputCloud(input);           // 设置待处理的输入点云
    SurfelCloudPtr output(new SurfelCloud); // 创建输出 Surfel 点云
    mls.process(*output);               // 执行 MLS 平滑和法线估计
    return (output);                    // 返回重建得到的 Surfel 点云
}

pcl::PolygonMeshPtr triangulateMesh(const SurfelCloudPtr &surfels) { // 使用 Surfel 点云构建三角网格
    // Create search tree*
    pcl::search::KdTree<SurfelT>::Ptr tree(new pcl::search::KdTree<SurfelT>); // 创建用于三角化邻域搜索的 KdTree
    tree->setInputCloud(surfels);       // 将 Surfel 点云设置为 KdTree 输入

    // Initialize objects
    pcl::GreedyProjectionTriangulation<SurfelT> gp3; // 创建贪心投影三角化对象
    pcl::PolygonMeshPtr triangles(new pcl::PolygonMesh); // 创建输出三角网格

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius(0.05);          // 设置三角边允许连接的最大搜索距离

    // Set typical values for the parameters
    gp3.setMu(2.5);                     // 设置最近邻距离乘子，控制搜索范围自适应扩张
    gp3.setMaximumNearestNeighbors(100); // 设置三角化时最多考虑的近邻数量
    gp3.setMaximumSurfaceAngle(M_PI / 4); // 设置允许连接的最大法线夹角，45 度
    gp3.setMinimumAngle(M_PI / 18);      // 设置生成三角形的最小内角，10 度
    gp3.setMaximumAngle(2 * M_PI / 3);   // 设置生成三角形的最大内角，120 度
    gp3.setNormalConsistency(true);      // 要求三角化结果保持法线方向一致

    // Get result
    gp3.setInputCloud(surfels);         // 设置三角化输入 Surfel 点云
    gp3.setSearchMethod(tree);          // 设置三角化使用的 KdTree 搜索结构
    gp3.reconstruct(*triangles);        // 执行贪心投影三角化并写入 triangles

    return triangles;                   // 返回生成的三角网格
}

int main(int argc, char **argv) {        // 程序入口，argv[1] 应为输入 PCD 文件路径

    // Load the points
    PointCloudPtr cloud(new PointCloud); // 创建输入点云对象
    if (argc == 0 || pcl::io::loadPCDFile(argv[1], *cloud)) { // 检查参数并尝试读取 PCD 点云
        cout << "failed to load point cloud!"; // 读取失败时输出错误信息
        return 1;                         // 返回非零值表示程序异常结束
    }
    cout << "point cloud loaded, points: " << cloud->points.size() << endl; // 输出读取到的点数

    // Compute surface elements
    cout << "computing normals ... " << endl; // 提示开始用 MLS 计算法线和 Surfel
    double mls_radius = 0.05, polynomial_order = 2; // 设置 MLS 搜索半径和局部拟合多项式阶数
    auto surfels = reconstructSurface(cloud, mls_radius, polynomial_order); // 对输入点云进行 MLS 表面重建

    // Compute a greedy surface triangulation
    cout << "computing mesh ... " << endl; // 提示开始三角网格重建
    pcl::PolygonMeshPtr mesh = triangulateMesh(surfels); // 用 Surfel 点云进行贪心投影三角化

    cout << "display mesh ... " << endl; // 提示开始显示网格
    pcl::visualization::PCLVisualizer vis; // 创建 PCL 可视化窗口
    vis.addPolylineFromPolygonMesh(*mesh, "mesh frame"); // 添加网格线框，便于观察三角形边界
    vis.addPolygonMesh(*mesh, "mesh");    // 添加完整三角网格表面
    vis.resetCamera();                    // 重置相机视角以便看到整个网格
    vis.spin();                           // 进入可视化循环，窗口关闭前程序会停在这里
}
