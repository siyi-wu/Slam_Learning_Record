#include <iostream>                                      // 标准输入输出流，用于 cout/endl 等
#include <opencv2/core/core.hpp>                         // OpenCV 核心数据结构，如 Mat、Point 等
#include <opencv2/features2d/features2d.hpp>             // OpenCV 2D 特征检测、描述和匹配接口
#include <opencv2/highgui/highgui.hpp>                   // OpenCV 图像读写和窗口显示接口
#include <opencv2/calib3d/calib3d.hpp>                   // OpenCV 相机标定和 PnP 等几何函数
#include <Eigen/Core>                                    // Eigen 核心矩阵和向量类型
#include <g2o/core/base_vertex.h>                        // g2o 顶点基类
#include <g2o/core/base_unary_edge.h>                    // g2o 一元边基类
#include <g2o/core/sparse_optimizer.h>                   // g2o 稀疏图优化器
#include <g2o/core/block_solver.h>                       // g2o 块求解器
#include <g2o/core/solver.h>                             // g2o 求解器基础接口
#include <g2o/core/optimization_algorithm_gauss_newton.h>// g2o 高斯牛顿优化算法
#include <g2o/solvers/dense/linear_solver_dense.h>       // g2o 稠密线性求解器
#include <sophus/se3.hpp>                                // Sophus SE3 李群/李代数类型
#include <chrono>                                        // C++ 时间库，用于统计耗时

using namespace std;                                     // 使用 std 命名空间，简化标准库类型书写
using namespace cv;                                      // 使用 cv 命名空间，简化 OpenCV 类型书写

void find_feature_matches(                               // 声明特征匹配函数
  const Mat &img_1, const Mat &img_2,                    // 输入的两张彩色图像
  std::vector<KeyPoint> &keypoints_1,                    // 输出第一张图像的关键点
  std::vector<KeyPoint> &keypoints_2,                    // 输出第二张图像的关键点
  std::vector<DMatch> &matches);                         // 输出筛选后的匹配关系

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);        // 声明像素坐标到归一化相机坐标的转换函数

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d; // 带内存对齐的二维 Eigen 向量数组
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d; // 带内存对齐的三维 Eigen 向量数组

void bundleAdjustmentG2O(                                // 声明基于 g2o 的 Bundle Adjustment 函数
  const VecVector3d &points_3d,                          // 输入三维点，位于第一帧相机坐标系
  const VecVector2d &points_2d,                          // 输入二维像素点，位于第二帧图像
  const Mat &K,                                          // 输入相机内参矩阵
  Sophus::SE3d &pose                                     // 输出优化后的位姿
);                                                       // 函数声明结束

// BA by gauss-newton
void bundleAdjustmentGaussNewton(                        // 声明手写高斯牛顿 BA 函数
  const VecVector3d &points_3d,                          // 输入三维点，位于第一帧相机坐标系
  const VecVector2d &points_2d,                          // 输入二维像素点，位于第二帧图像
  const Mat &K,                                          // 输入相机内参矩阵
  Sophus::SE3d &pose                                     // 输出优化后的位姿
);                                                       // 函数声明结束

int main(int argc, char **argv) {                        // 主函数入口，argc 为参数个数，argv 为参数列表
  if (argc != 5) {                                       // 检查命令行参数数量是否正确
    cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl; // 打印程序使用方式
    return 1;                                            // 参数错误时返回非零值
  }                                                      // 参数检查结束
  //-- 读取图像
  Mat img_1 = imread(argv[1], IMREAD_COLOR);             // 读取第一张彩色图像
  Mat img_2 = imread(argv[2], IMREAD_COLOR);             // 读取第二张彩色图像
  assert(img_1.data && img_2.data && "Can not load images!"); // 确认两张图像都成功读取

  vector<KeyPoint> keypoints_1, keypoints_2;             // 存放两张图像中的 ORB 关键点
  vector<DMatch> matches;                                // 存放筛选后的特征匹配
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches); // 提取并匹配两张图像的特征点
  cout << "一共找到了" << matches.size() << "组匹配点" << endl; // 输出有效匹配数量

  // 建立3D点
  Mat d1 = imread(argv[3], IMREAD_UNCHANGED);            // 读取第一张图像对应的深度图，保持原始 16 位单通道格式
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // 定义相机内参矩阵
  vector<Point3f> pts_3d;                                // 存放由第一帧深度恢复出的三维点
  vector<Point2f> pts_2d;                                // 存放第二帧中与三维点对应的二维像素点
  for (DMatch m:matches) {                               // 遍历每一组特征匹配
    ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)]; // 按第一帧关键点位置读取深度值
    if (d == 0)                                          // 判断深度值是否无效
      continue;                                          // 跳过没有深度的匹配点
    float dd = d / 5000.0;                               // 将深度图数值按比例尺转换为米
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K); // 将第一帧关键点像素坐标转换为归一化相机坐标
    pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd)); // 用归一化坐标和深度恢复三维点
    pts_2d.push_back(keypoints_2[m.trainIdx].pt);        // 保存第二帧中对应关键点的像素坐标
  }                                                      // 3D-2D 对应关系构建结束

  cout << "3d-2d pairs: " << pts_3d.size() << endl;      // 输出可用于 PnP 的 3D-2D 点对数量

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now(); // 记录 OpenCV PnP 开始时间
  Mat r, t;                                             // r 存储旋转向量，t 存储平移向量
  solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false);       // 调用 OpenCV 的 PnP 求解相机位姿
  Mat R;                                                // 存放由旋转向量转换出的旋转矩阵
  cv::Rodrigues(r, R);                                  // 用 Rodrigues 公式将旋转向量转换为旋转矩阵
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now(); // 记录 OpenCV PnP 结束时间
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1); // 计算耗时
  cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl; // 输出 OpenCV PnP 耗时

  cout << "R=" << endl << R << endl;                     // 输出旋转矩阵
  cout << "t=" << endl << t << endl;                     // 输出平移向量

  VecVector3d pts_3d_eigen;                              // 存放转换为 Eigen 格式的三维点
  VecVector2d pts_2d_eigen;                              // 存放转换为 Eigen 格式的二维点
  for (size_t i = 0; i < pts_3d.size(); ++i) {           // 遍历所有 3D-2D 点对
    pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z)); // 将 OpenCV 三维点转为 Eigen 三维向量
    pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y)); // 将 OpenCV 二维点转为 Eigen 二维向量
  }                                                      // Eigen 数据转换结束

  cout << "calling bundle adjustment by gauss newton" << endl; // 提示开始手写高斯牛顿 BA
  Sophus::SE3d pose_gn;                                  // 存放高斯牛顿优化得到的 SE3 位姿
  t1 = chrono::steady_clock::now();                      // 记录高斯牛顿 BA 开始时间
  bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn); // 使用手写高斯牛顿优化位姿
  t2 = chrono::steady_clock::now();                      // 记录高斯牛顿 BA 结束时间
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1); // 计算高斯牛顿 BA 耗时
  cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl; // 输出高斯牛顿 BA 耗时

  cout << "calling bundle adjustment by g2o" << endl;    // 提示开始 g2o BA
  Sophus::SE3d pose_g2o;                                 // 存放 g2o 优化得到的 SE3 位姿
  t1 = chrono::steady_clock::now();                      // 记录 g2o BA 开始时间
  bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o); // 使用 g2o 优化位姿
  t2 = chrono::steady_clock::now();                      // 记录 g2o BA 结束时间
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1); // 计算 g2o BA 耗时
  cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl; // 输出 g2o BA 耗时
  return 0;                                              // 主函数正常结束
}                                                        // main 函数结束

void find_feature_matches(const Mat &img_1, const Mat &img_2, // 定义特征匹配函数
                          std::vector<KeyPoint> &keypoints_1, // 第一张图像关键点输出
                          std::vector<KeyPoint> &keypoints_2, // 第二张图像关键点输出
                          std::vector<DMatch> &matches) {     // 筛选后的匹配输出
  //-- 初始化
  Mat descriptors_1, descriptors_2;                       // 存放两张图像中关键点的描述子
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();           // 创建 ORB 特征点检测器
  Ptr<DescriptorExtractor> descriptor = ORB::create();     // 创建 ORB 描述子计算器
  // use this if you are in OpenCV2
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" ); // OpenCV2 中创建 ORB 检测器的写法
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" ); // OpenCV2 中创建 ORB 描述子的写法
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming"); // 创建汉明距离暴力匹配器
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);                    // 在第一张图像中检测 ORB 关键点
  detector->detect(img_2, keypoints_2);                    // 在第二张图像中检测 ORB 关键点

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);  // 计算第一张图像关键点的描述子
  descriptor->compute(img_2, keypoints_2, descriptors_2);  // 计算第二张图像关键点的描述子

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> match;                                    // 临时保存所有原始匹配结果
  // BFMatcher matcher ( NORM_HAMMING );                   // 另一种直接创建 BFMatcher 的写法
  matcher->match(descriptors_1, descriptors_2, match);     // 对两组描述子进行一对一匹配

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;                   // 初始化最小和最大描述子距离

  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {           // 遍历第一张图像的每个描述子匹配
    double dist = match[i].distance;                       // 读取当前匹配的描述子距离
    if (dist < min_dist) min_dist = dist;                  // 更新最小距离
    if (dist > max_dist) max_dist = dist;                  // 更新最大距离
  }                                                        // 描述子距离统计结束

  printf("-- Max dist : %f \n", max_dist);                 // 打印最大描述子距离
  printf("-- Min dist : %f \n", min_dist);                 // 打印最小描述子距离

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {           // 遍历所有原始匹配
    if (match[i].distance <= max(2 * min_dist, 30.0)) {    // 保留距离足够小的匹配
      matches.push_back(match[i]);                         // 将通过阈值筛选的匹配加入输出
    }                                                      // 单个匹配筛选结束
  }                                                        // 匹配筛选结束
}                                                          // find_feature_matches 函数结束

Point2d pixel2cam(const Point2d &p, const Mat &K) {        // 定义像素坐标到归一化相机坐标的转换函数
  return Point2d                                           // 返回二维归一化坐标
    (                                                        // Point2d 构造参数开始
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),       // x_n = (u - c_x) / f_x
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)        // y_n = (v - c_y) / f_y
    );                                                       // Point2d 构造结束
}                                                          // pixel2cam 函数结束

void bundleAdjustmentGaussNewton(                          // 定义手写高斯牛顿 BA 函数
  const VecVector3d &points_3d,                             // 输入三维点
  const VecVector2d &points_2d,                             // 输入二维观测点
  const Mat &K,                                             // 输入相机内参矩阵
  Sophus::SE3d &pose) {                                     // 输入输出待优化的 SE3 位姿
  typedef Eigen::Matrix<double, 6, 1> Vector6d;              // 定义 6 维向量类型，对应 se(3) 更新量
  const int iterations = 10;                                 // 设置最大迭代次数
  double cost = 0, lastCost = 0;                             // 当前迭代代价和上一次迭代代价
  double fx = K.at<double>(0, 0);                            // 读取相机焦距 f_x
  double fy = K.at<double>(1, 1);                            // 读取相机焦距 f_y
  double cx = K.at<double>(0, 2);                            // 读取相机主点 c_x
  double cy = K.at<double>(1, 2);                            // 读取相机主点 c_y

  for (int iter = 0; iter < iterations; iter++) {            // 开始高斯牛顿迭代
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero(); // 初始化 Hessian 近似矩阵 H
    Vector6d b = Vector6d::Zero();                           // 初始化右端项 b

    cost = 0;                                                // 清零当前迭代总代价
    // compute cost
    for (int i = 0; i < points_3d.size(); i++) {             // 遍历所有 3D-2D 观测
      Eigen::Vector3d pc = pose * points_3d[i];              // 将三维点从第一帧坐标系变换到当前相机坐标系
      double inv_z = 1.0 / pc[2];                            // 计算深度倒数 1/Z
      double inv_z2 = inv_z * inv_z;                         // 计算深度倒数平方 1/Z^2
      Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy); // 将相机坐标投影到像素平面

      Eigen::Vector2d e = points_2d[i] - proj;               // 计算重投影误差，观测值减预测值

      cost += e.squaredNorm();                               // 累加平方误差作为优化代价
      Eigen::Matrix<double, 2, 6> J;                          // 定义误差关于 se(3) 更新量的雅可比矩阵
      J << -fx * inv_z,                                      // 第一行：误差 u 对平移 x 的偏导
        0,                                                   // 第一行：误差 u 对平移 y 的偏导
        fx * pc[0] * inv_z2,                                 // 第一行：误差 u 对平移 z 的偏导
        fx * pc[0] * pc[1] * inv_z2,                         // 第一行：误差 u 对旋转 x 的偏导
        -fx - fx * pc[0] * pc[0] * inv_z2,                   // 第一行：误差 u 对旋转 y 的偏导
        fx * pc[1] * inv_z,                                  // 第一行：误差 u 对旋转 z 的偏导
        0,                                                   // 第二行：误差 v 对平移 x 的偏导
        -fy * inv_z,                                         // 第二行：误差 v 对平移 y 的偏导
        fy * pc[1] * inv_z2,                                 // 第二行：误差 v 对平移 z 的偏导
        fy + fy * pc[1] * pc[1] * inv_z2,                    // 第二行：误差 v 对旋转 x 的偏导
        -fy * pc[0] * pc[1] * inv_z2,                        // 第二行：误差 v 对旋转 y 的偏导
        -fy * pc[0] * inv_z;                                 // 第二行：误差 v 对旋转 z 的偏导

      H += J.transpose() * J;                                // 累加高斯牛顿近似 Hessian：H = J^T J
      b += -J.transpose() * e;                               // 累加右端项：b = -J^T e
    }                                                        // 所有观测的线性化结束

    Vector6d dx;                                             // 存放求解得到的位姿增量
    dx = H.ldlt().solve(b);                                  // 使用 LDLT 分解求解线性方程 H dx = b

    if (isnan(dx[0])) {                                      // 检查求解结果是否出现 NaN
      cout << "result is nan!" << endl;                      // 输出数值异常提示
      break;                                                 // 数值异常时退出迭代
    }                                                        // NaN 检查结束

    if (iter > 0 && cost >= lastCost) {                      // 若当前代价不再下降，则认为更新无效
      // cost increase, update is not good
      cout << "cost: " << cost << ", last cost: " << lastCost << endl; // 输出当前代价和上一次代价
      break;                                                 // 代价上升时终止优化
    }                                                        // 代价变化检查结束

    // update your estimation
    pose = Sophus::SE3d::exp(dx) * pose;                     // 将李代数增量映射到 SE3 并左乘更新位姿
    lastCost = cost;                                         // 保存当前代价供下一次迭代比较

    cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl; // 输出当前迭代次数和代价
    if (dx.norm() < 1e-6) {                                  // 判断增量是否足够小
      // converge
      break;                                                 // 增量很小时认为已经收敛
    }                                                        // 收敛判断结束
  }                                                          // 高斯牛顿迭代结束

  cout << "pose by g-n: \n" << pose.matrix() << endl;        // 输出手写高斯牛顿优化后的位姿矩阵
}                                                            // bundleAdjustmentGaussNewton 函数结束

/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> { // 定义 g2o 位姿顶点，维度为 6，估计值类型为 Sophus::SE3d
public:                                                      // 以下成员为公开接口
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;                           // 为含 Eigen 成员的类启用内存对齐的 new 操作

  virtual void setToOriginImpl() override {                  // 重写顶点复位函数
    _estimate = Sophus::SE3d();                              // 将位姿估计重置为单位变换
  }                                                          // setToOriginImpl 函数结束

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {     // 重写顶点更新函数，接收 g2o 求出的增量
    Eigen::Matrix<double, 6, 1> update_eigen;                 // 定义 6 维李代数增量
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5]; // 将数组形式的增量拷贝到 Eigen 向量
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;  // 将李代数增量映射到 SE3 后左乘更新当前估计
  }                                                          // oplusImpl 函数结束

  virtual bool read(istream &in) override {return true;}      // g2o 顶点读取接口，此例中不需要读文件，直接返回 true

  virtual bool write(ostream &out) const override {return true;} // g2o 顶点写入接口，此例中不需要写文件，直接返回 true
};                                                           // VertexPose 类定义结束

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> { // 定义投影误差一元边，误差维度为 2
public:                                                                            // 以下成员为公开接口
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;                                                 // 为含 Eigen 成员的类启用内存对齐的 new 操作

  EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {} // 构造函数，保存三维点和相机内参

  virtual void computeError() override {                         // 重写误差计算函数
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]); // 取出该边连接的位姿顶点
    Sophus::SE3d T = v->estimate();                               // 读取当前位姿估计
    Eigen::Vector3d pos_pixel = _K * (T * _pos3d);                 // 将三维点变换到相机坐标系后乘以内参得到齐次像素坐标
    pos_pixel /= pos_pixel[2];                                    // 用第三维归一化，得到非齐次像素坐标
    _error = _measurement - pos_pixel.head<2>();                   // 误差为观测像素坐标减预测像素坐标
  }                                                               // computeError 函数结束

  virtual void linearizeOplus() override {                        // 重写雅可比计算函数
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]); // 取出该边连接的位姿顶点
    Sophus::SE3d T = v->estimate();                               // 读取当前位姿估计
    Eigen::Vector3d pos_cam = T * _pos3d;                         // 将三维点变换到当前相机坐标系
    double fx = _K(0, 0);                                         // 读取焦距 f_x
    double fy = _K(1, 1);                                         // 读取焦距 f_y
    double cx = _K(0, 2);                                         // 读取主点 c_x
    double cy = _K(1, 2);                                         // 读取主点 c_y
    double X = pos_cam[0];                                        // 读取相机坐标 X
    double Y = pos_cam[1];                                        // 读取相机坐标 Y
    double Z = pos_cam[2];                                        // 读取相机坐标 Z
    double Z2 = Z * Z;                                            // 计算 Z 的平方
    _jacobianOplusXi                                              // 填写误差对位姿李代数增量的 2x6 雅可比矩阵
      << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z, // 第一行对应 u 方向误差
      0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z; // 第二行对应 v 方向误差
  }                                                               // linearizeOplus 函数结束

  virtual bool read(istream &in) override {return true;}          // g2o 边读取接口，此例中不需要读文件，直接返回 true

  virtual bool write(ostream &out) const override {return true;}  // g2o 边写入接口，此例中不需要写文件，直接返回 true

private:                                                         // 以下成员为私有数据
  Eigen::Vector3d _pos3d;                                        // 该边关联的三维点坐标
  Eigen::Matrix3d _K;                                            // 相机内参矩阵
};                                                               // EdgeProjection 类定义结束

void bundleAdjustmentG2O(                                        // 定义基于 g2o 的 BA 函数
  const VecVector3d &points_3d,                                  // 输入三维点
  const VecVector2d &points_2d,                                  // 输入二维观测点
  const Mat &K,                                                  // 输入相机内参矩阵
  Sophus::SE3d &pose) {                                          // 输出优化后的 SE3 位姿

  // 构建图优化，先设定g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; // 定义块求解器类型：位姿维度 6，路标点维度 3
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 定义稠密线性求解器类型
  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(       // 创建 g2o 高斯牛顿优化算法对象
  make_unique<BlockSolverType>(make_unique<LinearSolverType>())); // 创建块求解器和线性求解器，并交给优化算法
  g2o::SparseOptimizer optimizer;                                // 创建稀疏图优化器
  optimizer.setAlgorithm(solver);                                // 设置优化器使用的求解算法
  optimizer.setVerbose(true);                                    // 打开 g2o 调试输出

  // vertex
  VertexPose *vertex_pose = new VertexPose();                    // 创建相机位姿顶点
  vertex_pose->setId(0);                                         // 设置位姿顶点 ID 为 0
  vertex_pose->setEstimate(Sophus::SE3d());                      // 将位姿初值设为单位变换
  optimizer.addVertex(vertex_pose);                              // 将位姿顶点加入图优化器

  // K
  Eigen::Matrix3d K_eigen;                                       // 创建 Eigen 格式的相机内参矩阵
  K_eigen <<                                                     // 将 OpenCV Mat 中的内参拷贝到 Eigen 矩阵
          K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2), // 内参矩阵第一行
    K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),       // 内参矩阵第二行
    K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);       // 内参矩阵第三行

  // edges
  int index = 1;                                                 // 初始化边的 ID，从 1 开始避免和顶点 ID 冲突
  for (size_t i = 0; i < points_2d.size(); ++i) {                // 遍历所有 3D-2D 观测
    auto p2d = points_2d[i];                                     // 取出当前二维像素观测
    auto p3d = points_3d[i];                                     // 取出当前对应三维点
    EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);     // 创建一个投影误差边
    edge->setId(index);                                          // 设置该边的 ID
    edge->setVertex(0, vertex_pose);                             // 将该边连接到位姿顶点
    edge->setMeasurement(p2d);                                   // 设置该边的二维观测值
    edge->setInformation(Eigen::Matrix2d::Identity());           // 设置信息矩阵为单位矩阵，表示两个像素方向权重相同
    optimizer.addEdge(edge);                                     // 将投影误差边加入图优化器
    index++;                                                     // 边 ID 自增
  }                                                              // 所有边添加结束

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now(); // 记录 g2o 优化开始时间
  optimizer.setVerbose(true);                                    // 再次设置输出调试信息
  optimizer.initializeOptimization();                            // 初始化图优化
  optimizer.optimize(10);                                        // 执行 10 次迭代优化
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now(); // 记录 g2o 优化结束时间
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1); // 计算 g2o 优化耗时
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl; // 输出 g2o 优化耗时
  cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl; // 输出 g2o 估计的位姿矩阵
  pose = vertex_pose->estimate();                                // 将优化结果写回输出参数
}                                                                // bundleAdjustmentG2O 函数结束
