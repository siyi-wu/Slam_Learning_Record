#include <iostream>                                      // 标准输入输出流，用于 cout/endl 等
#include <opencv2/opencv.hpp>                            // OpenCV 总头文件，包含图像、特征、几何等常用模块
// #include "extra.h" // used in opencv2                 // OpenCV2 版本中可能需要的额外头文件
using namespace std;                                     // 使用 std 命名空间，简化标准库类型书写
using namespace cv;                                      // 使用 cv 命名空间，简化 OpenCV 类型书写

void find_feature_matches(                               // 声明特征匹配函数
    const Mat &img_1, const Mat &img_2,                  // 输入的两张图像
    std::vector<KeyPoint> &keypoints_1,                  // 输出第一张图像的关键点
    std::vector<KeyPoint> &keypoints_2,                  // 输出第二张图像的关键点
    std::vector<DMatch> &matches                         // 输出筛选后的匹配关系
);                                                       // 函数声明结束

void pose_estimation_2d2d(                               // 声明 2D-2D 位姿估计函数
    const std::vector<KeyPoint> &keypoints_1,             // 输入第一张图像的关键点
    const std::vector<KeyPoint> &keypoints_2,             // 输入第二张图像的关键点
    const std::vector<DMatch> &matches,                   // 输入两张图像之间的匹配关系
    Mat &R, Mat &t                                        // 输出两帧之间的旋转 R 和平移方向 t
);                                                       // 函数声明结束

void triangulation(                                      // 声明三角测量函数
    const vector<KeyPoint> &keypoint_1,                   // 输入第一张图像关键点
    const vector<KeyPoint> &keypoint_2,                   // 输入第二张图像关键点
    const std::vector<DMatch> &matches,                   // 输入匹配关系
    const Mat &R, const Mat &t,                           // 输入由 2D-2D 估计得到的相对位姿
    vector<Point3d> &points                               // 输出三角化得到的三维点
);                                                       // 函数声明结束

/// 作图用
inline cv::Scalar get_color(float depth) {               // 根据深度生成绘图颜色
    float up_th = 50, low_th = 10, th_range = up_th - low_th; // 设置深度颜色映射的上下限和范围
    if (depth > up_th) depth = up_th;                    // 将过大的深度截断到上限
    if (depth < low_th) depth = low_th;                  // 将过小的深度截断到下限
    return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range)); // 返回按深度变化的 BGR 颜色
}                                                        // get_color 函数结束

// 像素坐标转相机归一化坐标
Point2f pixel2cam(const Point2d &p, const Mat &K);        // 声明像素坐标到归一化相机坐标的转换函数

int main(int argc, char **argv) {                        // 主函数入口，argc 为参数个数，argv 为参数列表
    if (argc != 3) {                                     // 检查命令行参数数量是否正确
        cout << "usage: triangulation img1 img2" << endl; // 打印程序使用方式
        return 1;                                        // 参数错误时返回非零值
    }                                                    // 参数检查结束
    //-- 读取图像
    Mat img_1 = imread(argv[1], IMREAD_COLOR);           // 读取第一张彩色图像
    Mat img_2 = imread(argv[2], IMREAD_COLOR);           // 读取第二张彩色图像

    vector<KeyPoint> keypoints_1, keypoints_2;           // 存放两张图像中的 ORB 关键点
    vector<DMatch> matches;                              // 存放筛选后的特征匹配
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches); // 提取并匹配两张图像的特征点
    cout << "一共找到了" << matches.size() << "组匹配点" << endl; // 输出有效匹配数量

    //-- 估计两张图像间运动
    Mat R, t;                                            // 存放由对极约束恢复出的旋转矩阵和平移方向
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t); // 根据 2D-2D 匹配估计相对位姿

    //-- 三角化
    vector<Point3d> points;                              // 存放三角测量恢复出的三维点
    triangulation(keypoints_1, keypoints_2, matches, R, t, points); // 用相对位姿和匹配点三角化空间点

    //-- 验证三角化点与特征点的重投影关系
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // 定义相机内参矩阵
    Mat img1_plot = img_1.clone();                         // 复制第一张图像用于绘制结果
    Mat img2_plot = img_2.clone();                         // 复制第二张图像用于绘制结果
    for (int i = 0; i < matches.size(); i++) {             // 遍历每一个匹配点和对应三角化点
        // 第一个图
        float depth1 = points[i].z;                        // 第一帧坐标系下三角化点的深度
        cout << "depth: " << depth1 << endl;               // 输出当前点的深度
        Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K); // 将第一帧匹配点转换为归一化相机坐标
        cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2); // 在第一张图像中按深度颜色绘制匹配点

        // 第二个图
        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t; // 将第一帧三维点变换到第二帧坐标系
        float depth2 = pt2_trans.at<double>(2, 0);          // 读取该点在第二帧坐标系下的深度
        cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2); // 在第二张图像中按深度颜色绘制匹配点
    }                                                       // 绘制循环结束
    cv::imshow("img 1", img1_plot);                        // 显示第一张绘制结果图
    cv::imshow("img 2", img2_plot);                        // 显示第二张绘制结果图
    cv::waitKey();                                         // 等待按键，保持窗口显示

    return 0;                                              // 主函数正常结束
}                                                          // main 函数结束

void find_feature_matches(const Mat &img_1, const Mat &img_2, // 定义特征匹配函数
    std::vector<KeyPoint> &keypoints_1,                    // 第一张图像关键点输出
    std::vector<KeyPoint> &keypoints_2,                    // 第二张图像关键点输出
    std::vector<DMatch> &matches) {                        // 筛选后的匹配输出
    //-- 初始化
    Mat descriptors_1, descriptors_2;                      // 存放两张图像中关键点的描述子
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();          // 创建 ORB 特征点检测器
    Ptr<DescriptorExtractor> descriptor = ORB::create();    // 创建 ORB 描述子计算器
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" ); // OpenCV2 中创建 ORB 检测器的写法
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" ); // OpenCV2 中创建 ORB 描述子的写法
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming"); // 创建汉明距离暴力匹配器
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);                   // 在第一张图像中检测 ORB 关键点
    detector->detect(img_2, keypoints_2);                   // 在第二张图像中检测 ORB 关键点

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1); // 计算第一张图像关键点的描述子
    descriptor->compute(img_2, keypoints_2, descriptors_2); // 计算第二张图像关键点的描述子

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;                                   // 临时保存所有原始匹配结果
    // BFMatcher matcher ( NORM_HAMMING );                  // 另一种直接创建 BFMatcher 的写法
    matcher->match(descriptors_1, descriptors_2, match);    // 对两组描述子进行一对一匹配

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;                  // 初始化最小和最大描述子距离

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {          // 遍历第一张图像的每个描述子匹配
        double dist = match[i].distance;                    // 读取当前匹配的描述子距离
        if (dist < min_dist) min_dist = dist;               // 更新最小距离
        if (dist > max_dist) max_dist = dist;               // 更新最大距离
    }                                                       // 描述子距离统计结束

    printf("-- Max dist : %f \n", max_dist);                // 打印最大描述子距离
    printf("-- Min dist : %f \n", min_dist);                // 打印最小描述子距离

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {          // 遍历所有原始匹配
        if (match[i].distance <= max(2 * min_dist, 30.0)) { // 保留距离足够小的匹配
        matches.push_back(match[i]);                        // 将通过阈值筛选的匹配加入输出
        }                                                   // 单个匹配筛选结束
    }                                                       // 匹配筛选结束
}                                                           // find_feature_matches 函数结束

void pose_estimation_2d2d(                                  // 定义 2D-2D 位姿估计函数
    const std::vector<KeyPoint> &keypoints_1,                // 输入第一张图像的关键点
    const std::vector<KeyPoint> &keypoints_2,                // 输入第二张图像的关键点
    const std::vector<DMatch> &matches,                      // 输入两张图像之间的匹配关系
    Mat &R, Mat &t) {                                        // 输出旋转矩阵 R 和平移方向 t
    // 相机内参,TUM Freiburg2
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // 定义 TUM Freiburg2 数据集的相机内参矩阵

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;                                 // 存放第一张图像中匹配点的像素坐标
    vector<Point2f> points2;                                 // 存放第二张图像中匹配点的像素坐标

    for (int i = 0; i < (int) matches.size(); i++) {         // 遍历所有匹配关系
        points1.push_back(keypoints_1[matches[i].queryIdx].pt); // 保存第一张图像中的匹配点坐标
        points2.push_back(keypoints_2[matches[i].trainIdx].pt); // 保存第二张图像中的匹配点坐标
    }                                                        // 匹配点坐标转换结束

    //-- 计算本质矩阵
    Point2d principal_point(325.1, 249.7);        //相机主点, TUM dataset标定值 // 定义相机主点
    int focal_length = 521;            //相机焦距, TUM dataset标定值          // 定义相机焦距
    Mat essential_matrix;                                 // 存放本质矩阵 E
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point); // 根据匹配点和相机内参估计本质矩阵

    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point); // 从本质矩阵恢复相对旋转和平移方向
}                                                            // pose_estimation_2d2d 函数结束

void triangulation(                                          // 定义三角测量函数
    const vector<KeyPoint> &keypoint_1,                       // 输入第一张图像关键点
    const vector<KeyPoint> &keypoint_2,                       // 输入第二张图像关键点
    const std::vector<DMatch> &matches,                       // 输入匹配关系
    const Mat &R, const Mat &t,                               // 输入两帧之间的相对位姿
    vector<Point3d> &points) {                                // 输出三角化得到的三维点
    Mat T1 = (Mat_<float>(3, 4) <<                            // 定义第一帧相机投影矩阵，第一帧作为世界坐标原点
        1, 0, 0, 0,                                           // T1 第一行
        0, 1, 0, 0,                                           // T1 第二行
        0, 0, 1, 0);                                          // T1 第三行
    Mat T2 = (Mat_<float>(3, 4) <<                            // 定义第二帧相机投影矩阵，由 R 和 t 组成
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0), // T2 第一行
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0), // T2 第二行
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)  // T2 第三行
    );                                                        // T2 构造结束

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // 定义相机内参矩阵
    vector<Point2f> pts_1, pts_2;                            // 存放两张图像中匹配点的归一化相机坐标
    for (DMatch m:matches) {                                  // 遍历所有匹配关系
        // 将像素坐标转换至相机坐标
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K)); // 将第一张图像匹配点转为归一化相机坐标
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K)); // 将第二张图像匹配点转为归一化相机坐标
    }                                                         // 匹配点坐标转换结束

    Mat pts_4d;                                               // 存放三角化得到的齐次四维点
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);       // 根据两个相机矩阵和对应归一化坐标三角化三维点

    // 转换成非齐次坐标
    for (int i = 0; i < pts_4d.cols; i++) {                   // 遍历每一个三角化点
        Mat x = pts_4d.col(i);                                // 取出第 i 个齐次坐标点
        x /= x.at<float>(3, 0); // 归一化                    // 用第四维 w 归一化齐次坐标
        Point3d p(                                            // 构造非齐次三维点
        x.at<float>(0, 0),                                    // 三维点 x 坐标
        x.at<float>(1, 0),                                    // 三维点 y 坐标
        x.at<float>(2, 0)                                     // 三维点 z 坐标
        );                                                     // Point3d 构造结束
        points.push_back(p);                                  // 将三维点加入输出数组
    }                                                         // 齐次转非齐次结束
}                                                             // triangulation 函数结束

Point2f pixel2cam(const Point2d &p, const Mat &K) {           // 定义像素坐标到归一化相机坐标的转换函数
    return Point2f                                            // 返回二维归一化坐标
        (                                                     // Point2f 构造参数开始
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),      // x_n = (u - c_x) / f_x
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)       // y_n = (v - c_y) / f_y
        );                                                    // Point2f 构造结束
}                                                             // pixel2cam 函数结束
