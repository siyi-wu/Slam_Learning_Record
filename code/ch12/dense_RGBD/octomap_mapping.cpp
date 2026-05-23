#include <iostream>                       // 引入标准输入输出流，用于 cout/cerr 打印信息
#include <fstream>                        // 引入文件输入输出流，用于读取 pose.txt

using namespace std;                      // 使用标准库命名空间，简化 vector、ifstream、cout 等写法

#include <opencv2/core/core.hpp>          // 引入 OpenCV 核心模块，提供 cv::Mat 等数据结构
#include <opencv2/highgui/highgui.hpp>    // 引入 OpenCV 图像读写模块，用于 imread

#include <octomap/octomap.h>              // 引入 OctoMap，用于构建八叉树占据地图

#include <Eigen/Geometry>                 // 引入 Eigen 几何模块，用于位姿和四元数计算
#include <boost/format.hpp>               // 引入 boost::format，用于格式化图像文件路径

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

    cout << "正在将图像转换为 Octomap ..." << endl; // 提示开始生成八叉树地图

    // octomap tree
    octomap::OcTree tree(0.01);           // 创建分辨率为 0.01 米的八叉树占据地图

    for (int i = 0; i < 5; i++) {         // 遍历每一帧 RGB-D 图像
        cout << "转换图像中: " << i + 1 << endl; // 输出当前正在处理的帧编号
        cv::Mat color = colorImgs[i];     // 取出当前帧彩色图，用它的尺寸遍历像素
        cv::Mat depth = depthImgs[i];     // 取出当前帧深度图，用于反投影三维点
        Eigen::Isometry3d T = poses[i];   // 取出当前帧相机位姿

        octomap::Pointcloud cloud;        // 创建 OctoMap 格式的点云，用于插入八叉树

        for (int v = 0; v < color.rows; v++) // 遍历图像每一行像素
            for (int u = 0; u < color.cols; u++) { // 遍历图像每一列像素
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 读取当前像素的原始深度值
                if (d == 0) continue;     // 深度为 0 表示无效测量，跳过该像素
                Eigen::Vector3d point;    // 创建相机坐标系下的三维点
                point[2] = double(d) / depthScale; // 将原始深度值换算为以米为单位的 z 坐标
                point[0] = (u - cx) * point[2] / fx; // 根据针孔模型反投影得到 x 坐标
                point[1] = (v - cy) * point[2] / fy; // 根据针孔模型反投影得到 y 坐标
                Eigen::Vector3d pointWorld = T * point; // 使用当前帧位姿把点变换到世界坐标系
                // 将世界坐标系的点放入点云
                cloud.push_back(pointWorld[0], pointWorld[1], pointWorld[2]); // 添加三维点到 OctoMap 点云
            }

        // 将点云存入八叉树地图，给定原点，这样可以计算投射线
        tree.insertPointCloud(cloud, octomap::point3d(T(0, 3), T(1, 3), T(2, 3))); // 从相机中心向点云投射，更新占据和空闲体素
    }

    // 更新中间节点的占据信息并写入磁盘
    tree.updateInnerOccupancy();          // 根据叶子节点状态更新八叉树中间节点占据概率
    cout << "saving octomap ... " << endl; // 提示开始保存八叉树地图
    tree.writeBinary("octomap.bt");       // 将 OctoMap 以二进制格式保存到 octomap.bt
    return 0;                             // 程序正常结束
}
