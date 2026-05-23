#include <iostream>                       // 引入标准输入输出流，用于 cout/cerr 打印信息
#include <fstream>                        // 引入文件输入输出流，用于读取 pose.txt

using namespace std;                      // 使用标准库命名空间，简化 vector、ifstream、cout 等写法

#include <opencv2/core/core.hpp>          // 引入 OpenCV 核心模块，提供 cv::Mat 等数据结构
#include <opencv2/highgui/highgui.hpp>    // 引入 OpenCV 图像读写模块，用于 imread
#include <Eigen/Geometry>                 // 引入 Eigen 几何模块，用于相机位姿和四元数
#include <boost/format.hpp>               // 引入 boost::format，用于格式化图像文件路径
#include <pcl/point_types.h>              // 引入 PCL 点类型定义，例如 PointXYZRGB
#include <pcl/io/pcd_io.h>                // 引入 PCL PCD 文件读写接口
#include <pcl/filters/voxel_grid.h>       // 引入体素滤波器，用于点云降采样
#include <pcl/visualization/pcl_visualizer.h> // 引入 PCL 可视化器头文件，本例未实际显示
#include <pcl/filters/statistical_outlier_removal.h> // 引入统计滤波器，用于去除离群点

int main(int argc, char **argv) {         // 程序入口，argc/argv 此处未使用
    vector<cv::Mat> colorImgs, depthImgs; // 保存五帧彩色图和深度图
    vector<Eigen::Isometry3d> poses;      // 保存五帧相机到世界坐标系的位姿

    ifstream fin("./data/pose.txt");      // 打开保存相机位姿的文本文件
    if (!fin) {                           // 检查位姿文件是否成功打开
        cerr << "cannot find pose file" << endl; // 文件不存在时输出错误信息
        return 1;                         // 返回非零值表示程序异常结束
    }

    for (int i = 0; i < 5; i++) {         // 依次读取示例中的五帧 RGB-D 数据
        boost::format fmt("./data/%s/%d.%s"); // 定义图像文件名格式，例如 ./data/color/1.png
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str())); // 读取第 i+1 帧彩色图
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(), -1)); // 以原始格式读取第 i+1 帧深度图

        double data[7] = {0};             // 保存一行位姿数据：tx, ty, tz, qx, qy, qz, qw
        for (int i = 0; i < 7; i++) {     // 读取平移和四元数的七个数值
            fin >> data[i];               // 将当前数值写入 data 数组
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]); // 按 w,x,y,z 顺序构造四元数
        Eigen::Isometry3d T(q);           // 用四元数初始化刚体变换的旋转部分
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2])); // 给刚体变换添加平移部分
        poses.push_back(T);               // 保存当前帧位姿，后面用于把点投到世界坐标系
    }

    // 计算点云并拼接
    // 相机内参
    double cx = 319.5;                    // 相机主点 x 坐标
    double cy = 239.5;                    // 相机主点 y 坐标
    double fx = 481.2;                    // 相机 x 方向焦距
    double fy = -480.0;                   // 相机 y 方向焦距，数据集坐标约定下为负
    double depthScale = 5000.0;           // 深度图尺度因子，原始深度值除以它得到米

    cout << "正在将图像转换为点云..." << endl; // 提示开始生成点云地图

    // 定义点云使用的格式：这里用的是XYZRGB
    typedef pcl::PointXYZRGB PointT;      // 定义带 RGB 颜色的三维点类型别名
    typedef pcl::PointCloud<PointT> PointCloud; // 定义该点类型对应的点云类型别名

    // 新建一个点云
    PointCloud::Ptr pointCloud(new PointCloud); // 创建最终累积的全局点云
    for (int i = 0; i < 5; i++) {         // 遍历每一帧 RGB-D 图像
        PointCloud::Ptr current(new PointCloud); // 创建当前帧单独的点云
        cout << "转换图像中: " << i + 1 << endl; // 输出当前正在处理的帧编号
        cv::Mat color = colorImgs[i];     // 取出当前帧彩色图
        cv::Mat depth = depthImgs[i];     // 取出当前帧深度图
        Eigen::Isometry3d T = poses[i];   // 取出当前帧相机位姿
        for (int v = 0; v < color.rows; v++) // 遍历图像每一行像素
            for (int u = 0; u < color.cols; u++) { // 遍历图像每一列像素
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 读取当前像素的原始深度值
                if (d == 0) continue;     // 深度为 0 表示无效测量，跳过该像素
                Eigen::Vector3d point;    // 创建相机坐标系下的三维点
                point[2] = double(d) / depthScale; // 将原始深度值换算为以米为单位的 z 坐标
                point[0] = (u - cx) * point[2] / fx; // 根据针孔模型反投影得到 x 坐标
                point[1] = (v - cy) * point[2] / fy; // 根据针孔模型反投影得到 y 坐标
                Eigen::Vector3d pointWorld = T * point; // 使用当前帧位姿把点变换到世界坐标系

                PointT p;                 // 创建一个 PCL 彩色点
                p.x = pointWorld[0];      // 写入世界坐标系 x 坐标
                p.y = pointWorld[1];      // 写入世界坐标系 y 坐标
                p.z = pointWorld[2];      // 写入世界坐标系 z 坐标
                p.b = color.data[v * color.step + u * color.channels()]; // 读取 OpenCV BGR 顺序中的蓝色通道
                p.g = color.data[v * color.step + u * color.channels() + 1]; // 读取绿色通道
                p.r = color.data[v * color.step + u * color.channels() + 2]; // 读取红色通道
                current->points.push_back(p); // 将当前彩色三维点加入当前帧点云
            }
        // depth filter and statistical removal
        PointCloud::Ptr tmp(new PointCloud); // 创建临时点云，用于接收当前帧滤波结果
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter; // 创建统计离群点滤波器
        statistical_filter.setMeanK(50);    // 每个点统计其 50 个近邻的平均距离
        statistical_filter.setStddevMulThresh(1.0); // 距离超过 1 个标准差阈值的点会被剔除
        statistical_filter.setInputCloud(current); // 设置待滤波的当前帧点云
        statistical_filter.filter(*tmp);    // 执行统计滤波并输出到 tmp
        (*pointCloud) += *tmp;              // 将滤波后的当前帧点云累加到全局点云
    }

    pointCloud->is_dense = false;          // 标记点云可能包含无效点或非紧密采样
    cout << "点云共有" << pointCloud->size() << "个点." << endl; // 输出滤波前的全局点数

    // voxel filter
    pcl::VoxelGrid<PointT> voxel_filter;  // 创建体素网格滤波器，用于降采样点云
    double resolution = 0.03;             // 设置体素边长为 0.03 米
    voxel_filter.setLeafSize(resolution, resolution, resolution); // 设置 x/y/z 三个方向的体素分辨率
    PointCloud::Ptr tmp(new PointCloud);  // 创建临时点云，用于保存降采样结果
    voxel_filter.setInputCloud(pointCloud); // 设置待降采样的全局点云
    voxel_filter.filter(*tmp);            // 执行体素滤波
    tmp->swap(*pointCloud);               // 用降采样后的点云替换原始全局点云

    cout << "滤波之后，点云共有" << pointCloud->size() << "个点." << endl; // 输出降采样后的点数

    pcl::io::savePCDFileBinary("map.pcd", *pointCloud); // 将最终点云以二进制 PCD 格式保存到 map.pcd
    return 0;                            // 程序正常结束
}
