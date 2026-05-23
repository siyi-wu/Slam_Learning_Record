#include "DBoW3/DBoW3.h"                  // 引入 DBoW3 库，用于训练和保存视觉词袋字典
#include <opencv2/core/core.hpp>          // 引入 OpenCV 核心模块，提供 Mat 等基础数据结构
#include <opencv2/highgui/highgui.hpp>    // 引入 OpenCV 图像读写相关接口，例如 imread
#include <opencv2/features2d/features2d.hpp> // 引入 OpenCV 特征模块，用于提取 ORB 特征
#include <iostream>                       // 引入标准输入输出流，用于 cout 打印运行信息
#include <vector>                         // 引入 vector 容器，用于保存文件名、时间戳和描述子
#include <string>                         // 引入 string 字符串类型，用于保存路径和文件名

using namespace cv;                       // 使用 OpenCV 命名空间，简化 Mat、ORB 等类型写法
using namespace std;                      // 使用标准库命名空间，简化 string、vector、cout 等写法


int main( int argc, char** argv )         // 程序入口，argv[1] 需要传入数据集目录
{
    string dataset_dir = argv[1];         // 从命令行参数读取 RGB-D 数据集根目录
    ifstream fin ( dataset_dir+"/associate.txt" ); // 打开关联文件，文件中记录 RGB 图和深度图的对应关系
    if ( !fin )                           // 判断 associate.txt 是否成功打开
    {
        cout<<"please generate the associate file called associate.txt!"<<endl; // 提示用户先生成关联文件
        return 1;                         // 文件打开失败，返回非零值表示程序异常结束
    }

    vector<string> rgb_files, depth_files; // 分别保存 RGB 图像路径和深度图像路径
    vector<double> rgb_times, depth_times; // 分别保存 RGB 图像时间戳和深度图像时间戳
    while ( !fin.eof() )                  // 循环读取关联文件直到文件末尾
    {
        string rgb_time, rgb_file, depth_time, depth_file; // 临时保存当前行的时间戳和相对文件路径
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file; // 从当前行读取 RGB 时间、RGB 文件、深度时间、深度文件
        rgb_times.push_back ( atof ( rgb_time.c_str() ) ); // 将 RGB 时间戳从字符串转换为 double 并保存
        depth_times.push_back ( atof ( depth_time.c_str() ) ); // 将深度时间戳从字符串转换为 double 并保存
        rgb_files.push_back ( dataset_dir+"/"+rgb_file ); // 拼接并保存 RGB 图像的完整路径
        depth_files.push_back ( dataset_dir+"/"+depth_file ); // 拼接并保存深度图像的完整路径

        if ( fin.good() == false )        // 如果读取状态异常，说明已经读到无效行或文件结束
            break;                        // 跳出读取循环，避免继续处理无效数据
    }
    fin.close();                          // 关闭关联文件

    cout<<"generating features ... "<<endl; // 提示开始提取所有 RGB 图像的特征
    vector<Mat> descriptors;             // 保存每张图像提取到的 ORB 描述子
    Ptr< Feature2D > detector = ORB::create(); // 创建 ORB 特征检测器和描述子计算器
    int index = 1;                       // 用于输出当前处理图像的序号
    for ( string rgb_file:rgb_files )    // 遍历所有 RGB 图像路径
    {
        Mat image = imread(rgb_file);    // 从磁盘读取当前 RGB 图像
        vector<KeyPoint> keypoints;      // 保存当前图像检测到的 ORB 关键点
        Mat descriptor;                  // 保存当前图像计算得到的 ORB 描述子
        detector->detectAndCompute( image, Mat(), keypoints, descriptor ); // 提取关键点并计算描述子
        descriptors.push_back( descriptor ); // 将当前图像的描述子加入词典训练数据
        cout<<"extracting features from image " << index++ <<endl; // 输出当前已处理的图像序号
    }
    cout<<"extract total "<<descriptors.size()*500<<" features."<<endl; // 粗略输出特征总数，默认每张图约 500 个 ORB 特征

    // create vocabulary
    cout<<"creating vocabulary, please wait ... "<<endl; // 提示开始训练较大的视觉词典
    DBoW3::Vocabulary vocab;           // 创建一个空的 DBoW3 视觉词典对象
    vocab.create( descriptors );       // 使用数据集中的 ORB 描述子训练词典
    cout<<"vocabulary info: "<<vocab<<endl; // 输出训练完成后的词典信息
    vocab.save( "vocab_larger.yml.gz" ); // 将训练好的大词典保存到文件
    cout<<"done"<<endl;                // 提示词典生成流程完成

    return 0;                          // 正常结束程序
}                                      // main 函数结束
