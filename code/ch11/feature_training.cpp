#include "DBoW3/DBoW3.h"                  // 引入 DBoW3 库，用于创建和保存视觉词袋字典
#include <opencv2/core/core.hpp>          // 引入 OpenCV 核心模块，提供 Mat 等基础数据结构
#include <opencv2/highgui/highgui.hpp>    // 引入 OpenCV 图像读写和窗口显示相关接口
#include <opencv2/features2d/features2d.hpp> // 引入 OpenCV 二维特征模块，用于 ORB 特征提取
#include <iostream>                       // 引入标准输入输出流，用于 cout 打印信息
#include <vector>                         // 引入 vector 容器，用于保存图像和描述子
#include <string>                         // 引入 string 字符串类型，用于拼接图像路径

using namespace cv;                       // 使用 OpenCV 命名空间，简化 Mat、ORB 等类型写法
using namespace std;                      // 使用标准库命名空间，简化 cout、vector、string 等写法

/***************************************************
 * 本节演示了如何根据data/目录下的十张图训练字典
 * ************************************************/

int main( int argc, char** argv ) {       // 程序入口，argc/argv 此处未使用
    // read the image
    cout<<"reading images... "<<endl;     // 提示开始读取训练图像
    vector<Mat> images;                   // 创建图像容器，用于保存读入的十张图片
    for ( int i=0; i<10; i++ )            // 依次读取 data 目录下编号 1 到 10 的图像
    {
        string path = "./data/"+to_string(i+1)+".png"; // 根据循环编号拼接当前图像路径
        images.push_back( imread(path) ); // 读取图像并追加到 images 容器中
    }
    // detect ORB features
    cout<<"detecting ORB features ... "<<endl; // 提示开始提取 ORB 特征
    Ptr< Feature2D > detector = ORB::create(); // 创建 ORB 特征检测器和描述子计算器
    vector<Mat> descriptors;             // 创建描述子容器，每张图像对应一个 Mat 描述子矩阵
    for ( Mat& image:images )            // 遍历所有读入的图像
    {
        vector<KeyPoint> keypoints;      // 保存当前图像检测到的 ORB 关键点
        Mat descriptor;                  // 保存当前图像计算得到的 ORB 描述子
        detector->detectAndCompute( image, Mat(), keypoints, descriptor ); // 同时完成关键点检测和描述子计算
        descriptors.push_back( descriptor ); // 将当前图像的描述子加入训练集合
    }

    // create vocabulary
    cout<<"creating vocabulary ... "<<endl; // 提示开始根据描述子训练视觉词典
    DBoW3::Vocabulary vocab;           // 创建一个空的 DBoW3 视觉词典对象
    vocab.create( descriptors );       // 使用所有图像的 ORB 描述子训练词典
    cout<<"vocabulary info: "<<vocab<<endl; // 打印训练完成后的词典信息
    vocab.save( "vocabulary.yml.gz" ); // 将词典保存为压缩的 YAML 文件，供后续回环检测使用
    cout<<"done"<<endl;                // 提示词典训练和保存完成

    return 0;                          // 正常结束程序
}                                      // main 函数结束
