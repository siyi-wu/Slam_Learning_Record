#include <iostream>                                      // 标准输入输出流，用于 cout/endl 等
#include <opencv2/core/core.hpp>                         // OpenCV 核心数据结构，如 Mat、Point 等
#include <opencv2/features2d/features2d.hpp>             // OpenCV 2D 特征检测、描述和匹配接口
#include <opencv2/highgui/highgui.hpp>                   // OpenCV 图像读写和窗口显示接口
#include <opencv2/calib3d/calib3d.hpp>                   // OpenCV 相机标定和几何计算相关接口
#include <Eigen/Core>                                    // Eigen 核心矩阵和向量类型
#include <Eigen/Dense>                                   // Eigen 稠密矩阵运算
#include <Eigen/Geometry>                                // Eigen 几何变换相关类型
#include <Eigen/SVD>                                     // Eigen SVD 分解算法
#include <g2o/core/base_vertex.h>                        // g2o 顶点基类
#include <g2o/core/base_unary_edge.h>                    // g2o 一元边基类
#include <g2o/core/block_solver.h>                       // g2o 块求解器
#include <g2o/core/optimization_algorithm_gauss_newton.h>// g2o 高斯牛顿优化算法头文件
#include <g2o/core/optimization_algorithm_levenberg.h>   // g2o Levenberg-Marquardt 优化算法
#include <g2o/solvers/dense/linear_solver_dense.h>       // g2o 稠密线性求解器
#include <chrono>                                        // C++ 时间库，用于统计耗时
#include <sophus/se3.hpp>                                // Sophus SE3 李群/李代数类型

using namespace std;                                     // 使用 std 命名空间，简化标准库类型书写
using namespace cv;                                      // 使用 cv 命名空间，简化 OpenCV 类型书写

void find_feature_matches(                               // 声明特征匹配函数
  const Mat &img_1, const Mat &img_2,                    // 输入的两张图像
  std::vector<KeyPoint> &keypoints_1,                    // 输出第一张图像的关键点
  std::vector<KeyPoint> &keypoints_2,                    // 输出第二张图像的关键点
  std::vector<DMatch> &matches);                         // 输出筛选后的匹配关系

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);        // 声明像素坐标到归一化相机坐标的转换函数

void pose_estimation_3d3d(                                // 声明 3D-3D 位姿估计函数
  const vector<Point3f> &pts1,                            // 输入第一帧相机坐标系下的三维点
  const vector<Point3f> &pts2,                            // 输入第二帧相机坐标系下的三维点
  Mat &R, Mat &t                                          // 输出旋转矩阵 R 和平移向量 t
);                                                        // 函数声明结束

void bundleAdjustment(                                    // 声明基于 g2o 的 3D-3D BA 函数
  const vector<Point3f> &points_3d,                       // 输入第一组三维点
  const vector<Point3f> &points_2d,                       // 输入第二组三维点，变量名保留原代码写法
  Mat &R, Mat &t                                          // 输入输出待优化的 R 和 t
);                                                        // 函数声明结束

/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> { // 定义 g2o 位姿顶点，维度为 6，估计值类型为 SE3
public:                                                     // 以下成员为公开接口
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;                          // 为含 Eigen 成员的类启用内存对齐的 new 操作

  virtual void setToOriginImpl() override {                 // 重写顶点复位函数
    _estimate = Sophus::SE3d();                             // 将位姿估计重置为单位变换
  }                                                         // setToOriginImpl 函数结束

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {   // 重写顶点更新函数，接收 g2o 求出的 6 维增量
    Eigen::Matrix<double, 6, 1> update_eigen;               // 定义 se(3) 李代数增量向量
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5]; // 将数组增量拷贝到 Eigen 向量
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate; // 将李代数增量映射到 SE3 后左乘更新位姿
  }                                                         // oplusImpl 函数结束

  virtual bool read(istream &in) override {return true;}    // g2o 顶点读取接口，此例不从文件读取，直接返回 true

  virtual bool write(ostream &out) const override {return true;} // g2o 顶点写入接口，此例不写文件，直接返回 true
};                                                          // VertexPose 类定义结束

/// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> { // 定义 3D-3D 误差边，误差维度为 3
public:                                                                                        // 以下成员为公开接口
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;                                                             // 为含 Eigen 成员的类启用内存对齐的 new 操作

  EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {} // 构造函数，保存第二帧中的三维点

  virtual void computeError() override {                         // 重写误差计算函数
    const VertexPose *pose = static_cast<const VertexPose *> ( _vertices[0] ); // 取出该边连接的位姿顶点
    _error = _measurement - pose->estimate() * _point;            // 误差 = 第一帧三维观测 - 当前位姿变换后的第二帧三维点
  }                                                               // computeError 函数结束

  virtual void linearizeOplus() override {                        // 重写雅可比计算函数
    VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);   // 取出该边连接的位姿顶点
    Sophus::SE3d T = pose->estimate();                            // 读取当前 SE3 位姿估计
    Eigen::Vector3d xyz_trans = T * _point;                       // 将第二帧三维点变换到第一帧坐标系
    _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity(); // 误差对平移增量的雅可比
    _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans); // 误差对旋转增量的雅可比，hat 表示反对称矩阵
  }                                                               // linearizeOplus 函数结束

  bool read(istream &in) override {return true;}                  // g2o 边读取接口，此例不从文件读取，直接返回 true

  bool write(ostream &out) const override {return true;}          // g2o 边写入接口，此例不写文件，直接返回 true

protected:                                                        // 以下成员对子类可见
  Eigen::Vector3d _point;                                         // 保存第二帧中的三维点坐标
};                                                               // EdgeProjectXYZRGBDPoseOnly 类定义结束

int main(int argc, char **argv) {                                // 主函数入口，argc 为参数个数，argv 为参数列表
  if (argc != 5) {                                               // 检查命令行参数数量是否正确
    cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl; // 打印程序使用方式
    return 1;                                                    // 参数错误时返回非零值
  }                                                              // 参数检查结束
  //-- 读取图像
  Mat img_1 = imread(argv[1], IMREAD_COLOR);                     // 读取第一张彩色图像
  Mat img_2 = imread(argv[2], IMREAD_COLOR);                     // 读取第二张彩色图像

  vector<KeyPoint> keypoints_1, keypoints_2;                     // 存放两张图像中的 ORB 关键点
  vector<DMatch> matches;                                        // 存放筛选后的特征匹配
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches); // 提取并匹配两张图像的特征点
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;  // 输出有效匹配数量

  // 建立3D点
  Mat depth1 = imread(argv[3], IMREAD_UNCHANGED);                // 读取第一张深度图，保持 16 位无符号单通道格式
  Mat depth2 = imread(argv[4], IMREAD_UNCHANGED);                // 读取第二张深度图，保持 16 位无符号单通道格式
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // 定义相机内参矩阵
  vector<Point3f> pts1, pts2;                                    // 分别存放两帧中对应特征点反投影得到的三维点

  for (DMatch m:matches) {                                       // 遍历每一组特征匹配
    ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)]; // 读取第一帧匹配点处的深度值
    ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)]; // 读取第二帧匹配点处的深度值
    if (d1 == 0 || d2 == 0)                                      // 判断任一深度是否无效
      continue;                                                  // 跳过没有有效深度的匹配点
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);        // 将第一帧像素坐标转换为归一化相机坐标
    Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);        // 将第二帧像素坐标转换为归一化相机坐标
    float dd1 = float(d1) / 5000.0;                              // 将第一帧深度值按比例尺转换为米
    float dd2 = float(d2) / 5000.0;                              // 将第二帧深度值按比例尺转换为米
    pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));        // 用第一帧归一化坐标和深度恢复第一帧三维点
    pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));        // 用第二帧归一化坐标和深度恢复第二帧三维点
  }                                                              // 3D-3D 对应关系构建结束

  cout << "3d-3d pairs: " << pts1.size() << endl;                // 输出可用于 3D-3D 位姿估计的点对数量
  Mat R, t;                                                      // 存放估计出的旋转矩阵和平移向量
  pose_estimation_3d3d(pts1, pts2, R, t);                        // 使用 SVD 方法估计两组三维点之间的刚体变换
  cout << "ICP via SVD results: " << endl;                       // 输出 SVD/ICP 结果标题
  cout << "R = " << R << endl;                                   // 输出旋转矩阵
  cout << "t = " << t << endl;                                   // 输出平移向量
  cout << "R_inv = " << R.t() << endl;                           // 输出旋转矩阵的逆，即转置
  cout << "t_inv = " << -R.t() * t << endl;                      // 输出逆变换中的平移部分

  cout << "calling bundle adjustment" << endl;                   // 提示开始 g2o BA 优化

  bundleAdjustment(pts1, pts2, R, t);                            // 使用 g2o 对 SVD 得到的位姿进行优化

  // verify p1 = R * p2 + t
  for (int i = 0; i < 5; i++) {                                  // 打印前 5 组点验证变换关系
    cout << "p1 = " << pts1[i] << endl;                          // 输出第一帧中的三维点
    cout << "p2 = " << pts2[i] << endl;                          // 输出第二帧中的对应三维点
    cout << "(R*p2+t) = " <<                                    // 输出将第二帧点变换到第一帧后的结果
         R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t // 计算 R*p2+t
         << endl;                                                // 结束当前结果输出
    cout << endl;                                                // 输出空行，分隔每组验证结果
  }                                                              // 验证循环结束
}                                                                // main 函数结束

void find_feature_matches(const Mat &img_1, const Mat &img_2,    // 定义特征匹配函数
                          std::vector<KeyPoint> &keypoints_1,    // 第一张图像关键点输出
                          std::vector<KeyPoint> &keypoints_2,    // 第二张图像关键点输出
                          std::vector<DMatch> &matches) {        // 筛选后的匹配输出
  //-- 初始化
  Mat descriptors_1, descriptors_2;                              // 存放两张图像中关键点的描述子
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();                  // 创建 ORB 特征点检测器
  Ptr<DescriptorExtractor> descriptor = ORB::create();            // 创建 ORB 描述子计算器
  // use this if you are in OpenCV2
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" ); // OpenCV2 中创建 ORB 检测器的写法
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" ); // OpenCV2 中创建 ORB 描述子的写法
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming"); // 创建汉明距离暴力匹配器
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);                          // 在第一张图像中检测 ORB 关键点
  detector->detect(img_2, keypoints_2);                          // 在第二张图像中检测 ORB 关键点

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);        // 计算第一张图像关键点的描述子
  descriptor->compute(img_2, keypoints_2, descriptors_2);        // 计算第二张图像关键点的描述子

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> match;                                          // 临时保存所有原始匹配结果
  // BFMatcher matcher ( NORM_HAMMING );                         // 另一种直接创建 BFMatcher 的写法
  matcher->match(descriptors_1, descriptors_2, match);           // 对两组描述子进行一对一匹配

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;                         // 初始化最小和最大描述子距离

  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {                 // 遍历第一张图像的每个描述子匹配
    double dist = match[i].distance;                             // 读取当前匹配的描述子距离
    if (dist < min_dist) min_dist = dist;                        // 更新最小距离
    if (dist > max_dist) max_dist = dist;                        // 更新最大距离
  }                                                              // 描述子距离统计结束

  printf("-- Max dist : %f \n", max_dist);                       // 打印最大描述子距离
  printf("-- Min dist : %f \n", min_dist);                       // 打印最小描述子距离

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {                 // 遍历所有原始匹配
    if (match[i].distance <= max(2 * min_dist, 30.0)) {          // 保留距离足够小的匹配
      matches.push_back(match[i]);                               // 将通过阈值筛选的匹配加入输出
    }                                                            // 单个匹配筛选结束
  }                                                              // 匹配筛选结束
}                                                                // find_feature_matches 函数结束

Point2d pixel2cam(const Point2d &p, const Mat &K) {              // 定义像素坐标到归一化相机坐标的转换函数
  return Point2d(                                                // 返回二维归一化坐标
    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),             // x_n = (u - c_x) / f_x
    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)              // y_n = (v - c_y) / f_y
  );                                                             // Point2d 构造结束
}                                                                // pixel2cam 函数结束

void pose_estimation_3d3d(const vector<Point3f> &pts1,           // 定义 SVD 求解 3D-3D 位姿的函数
                          const vector<Point3f> &pts2,           // 输入第二帧三维点
                          Mat &R, Mat &t) {                      // 输出 R 和 t
  Point3f p1, p2;     // center of mass                          // p1、p2 分别表示两组三维点的质心
  int N = pts1.size();                                           // 获取三维点对数量
  for (int i = 0; i < N; i++) {                                  // 遍历所有三维点
    p1 += pts1[i];                                               // 累加第一组三维点坐标
    p2 += pts2[i];                                               // 累加第二组三维点坐标
  }                                                              // 坐标累加结束
  p1 = Point3f(Vec3f(p1) / N);                                   // 计算第一组三维点质心
  p2 = Point3f(Vec3f(p2) / N);                                   // 计算第二组三维点质心
  vector<Point3f> q1(N), q2(N); // remove the center              // 保存去质心后的两组三维点
  for (int i = 0; i < N; i++) {                                  // 遍历所有三维点
    q1[i] = pts1[i] - p1;                                        // 第一组三维点减去质心
    q2[i] = pts2[i] - p2;                                        // 第二组三维点减去质心
  }                                                              // 去质心结束

  // compute q1*q2^T
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();                   // 初始化协方差矩阵 W
  for (int i = 0; i < N; i++) {                                  // 遍历所有去质心后的点对
    W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose(); // 累加 q1*q2^T
  }                                                              // 协方差矩阵累加结束
  cout << "W=" << W << endl;                                     // 输出协方差矩阵 W

  // SVD on W
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV); // 对 W 做 SVD 分解并计算完整 U、V
  Eigen::Matrix3d U = svd.matrixU();                             // 取出 SVD 分解得到的 U 矩阵
  Eigen::Matrix3d V = svd.matrixV();                             // 取出 SVD 分解得到的 V 矩阵

  cout << "U=" << U << endl;                                     // 输出 U 矩阵
  cout << "V=" << V << endl;                                     // 输出 V 矩阵

  Eigen::Matrix3d R_ = U * (V.transpose());                      // 根据 SVD 结果计算旋转矩阵 R = U*V^T
  if (R_.determinant() < 0) {                                    // 检查旋转矩阵行列式是否为负
    R_ = -R_;                                                    // 若出现反射情况，则翻转符号
  }                                                              // 行列式检查结束
  Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z); // 根据质心关系计算平移 t = p1 - R*p2

  // convert to cv::Mat
  R = (Mat_<double>(3, 3) <<                                     // 将 Eigen 旋转矩阵转换为 OpenCV Mat
    R_(0, 0), R_(0, 1), R_(0, 2),                                // 旋转矩阵第一行
    R_(1, 0), R_(1, 1), R_(1, 2),                                // 旋转矩阵第二行
    R_(2, 0), R_(2, 1), R_(2, 2)                                 // 旋转矩阵第三行
  );                                                             // 旋转矩阵转换结束
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));       // 将 Eigen 平移向量转换为 OpenCV Mat
}                                                                // pose_estimation_3d3d 函数结束

void bundleAdjustment(                                          // 定义基于 g2o 的 3D-3D BA 函数
  const vector<Point3f> &pts1,                                  // 输入第一帧三维点，作为观测目标
  const vector<Point3f> &pts2,                                  // 输入第二帧三维点，作为被变换点
  Mat &R, Mat &t) {                                             // 输入输出 R 和 t
  // 构建图优化，先设定g2o
  typedef g2o::BlockSolverX BlockSolverType;                    // 定义通用块求解器类型
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 定义稠密线性求解器类型
  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmLevenberg(         // 创建 Levenberg-Marquardt 优化算法对象
  make_unique<BlockSolverType>(make_unique<LinearSolverType>())); // 创建块求解器和线性求解器，并交给优化算法
  g2o::SparseOptimizer optimizer;                               // 创建稀疏图优化器
  optimizer.setAlgorithm(solver);                               // 设置优化器使用的求解算法
  optimizer.setVerbose(true);                                   // 打开 g2o 调试输出

  // vertex
  VertexPose *pose = new VertexPose();                          // 创建相机位姿顶点
  pose->setId(0);                                               // 设置位姿顶点 ID 为 0
  pose->setEstimate(Sophus::SE3d());                            // 将位姿初值设为单位变换
  optimizer.addVertex(pose);                                    // 将位姿顶点加入图优化器

  // edges
  for (size_t i = 0; i < pts1.size(); i++) {                    // 遍历所有 3D-3D 点对
    EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly( // 创建一条 3D-3D 误差边
      Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));         // 将第二帧三维点作为待变换点传入边
    edge->setVertex(0, pose);                                   // 将该边连接到位姿顶点
    edge->setMeasurement(Eigen::Vector3d(                       // 设置该边的观测值
      pts1[i].x, pts1[i].y, pts1[i].z));                        // 观测值为第一帧中的对应三维点
    edge->setInformation(Eigen::Matrix3d::Identity());          // 设置信息矩阵为单位矩阵，表示三个方向权重相同
    optimizer.addEdge(edge);                                    // 将误差边加入图优化器
  }                                                             // 所有边添加结束

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now(); // 记录 g2o 优化开始时间
  optimizer.initializeOptimization();                           // 初始化图优化
  optimizer.optimize(10);                                       // 执行 10 次迭代优化
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now(); // 记录 g2o 优化结束时间
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1); // 计算优化耗时
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl; // 输出优化耗时

  cout << endl << "after optimization:" << endl;                // 输出优化完成提示
  cout << "T=\n" << pose->estimate().matrix() << endl;          // 输出优化后的 SE3 位姿矩阵

  // convert to cv::Mat
  Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();       // 从优化后的 SE3 中取出旋转矩阵
  Eigen::Vector3d t_ = pose->estimate().translation();          // 从优化后的 SE3 中取出平移向量
  R = (Mat_<double>(3, 3) <<                                    // 将 Eigen 旋转矩阵转换为 OpenCV Mat
    R_(0, 0), R_(0, 1), R_(0, 2),                               // 旋转矩阵第一行
    R_(1, 0), R_(1, 1), R_(1, 2),                               // 旋转矩阵第二行
    R_(2, 0), R_(2, 1), R_(2, 2)                                // 旋转矩阵第三行
  );                                                            // 旋转矩阵转换结束
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));      // 将 Eigen 平移向量转换为 OpenCV Mat
}                                                               // bundleAdjustment 函数结束
