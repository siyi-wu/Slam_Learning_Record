#include <iostream>                       // 引入标准输入输出流，用于 cout 打印信息
#include <vector>                         // 引入 vector 容器，用于保存图像路径和位姿
#include <fstream>                        // 引入文件输入输出流，用于读取轨迹和深度真值文件

using namespace std;                      // 使用标准库命名空间，简化 vector、string、cout 等写法

#include <boost/timer.hpp>                // 引入 boost 计时器，本文件当前未显式使用

// for sophus
#include <sophus/se3.hpp>                 // 引入 Sophus SE3 李群位姿表示

using Sophus::SE3d;                       // 使用 Sophus 的双精度 SE3 类型

// for eigen
#include <Eigen/Core>                     // 引入 Eigen 核心矩阵和向量类型
#include <Eigen/Geometry>                 // 引入 Eigen 几何模块，例如四元数

using namespace Eigen;                    // 使用 Eigen 命名空间，简化 Vector2d/Vector3d 等类型写法

#include <opencv2/core/core.hpp>          // 引入 OpenCV 核心模块，提供 Mat 等数据结构
#include <opencv2/highgui/highgui.hpp>    // 引入 OpenCV 图像读写和窗口显示接口
#include <opencv2/imgproc/imgproc.hpp>    // 引入 OpenCV 图像处理接口，例如颜色转换

using namespace cv;                       // 使用 OpenCV 命名空间，简化 Mat、imshow 等写法

/**********************************************
* 本程序演示了单目相机在已知轨迹下的稠密深度估计
* 使用极线搜索 + NCC 匹配的方式，与书本的 12.2 节对应
* 请注意本程序并不完美，你完全可以改进它——我其实在故意暴露一些问题(这是借口)。
***********************************************/

// ------------------------------------------------------------------
// parameters
const int boarder = 20;         // 边缘宽度，避免 NCC 窗口或极线搜索越界
const int width = 640;          // 图像宽度
const int height = 480;         // 图像高度
const double fx = 481.2f;       // 相机 x 方向焦距
const double fy = -480.0f;      // 相机 y 方向焦距，数据集坐标约定下为负
const double cx = 319.5f;       // 相机主点 x 坐标
const double cy = 239.5f;       // 相机主点 y 坐标
const int ncc_window_size = 3;    // NCC 取的窗口半宽度
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC窗口面积
const double min_cov = 0.1;     // 收敛判定：最小方差
const double max_cov = 10;      // 发散判定：最大方差

// ------------------------------------------------------------------
// 重要的函数
/// 从 REMODE 数据集读取数据
bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    cv::Mat &ref_depth
);

/**
 * 根据新的图像更新深度估计
 * @param ref           参考图像
 * @param curr          当前图像
 * @param T_C_R         参考图像到当前图像的位姿
 * @param depth         深度
 * @param depth_cov     深度方差
 * @return              是否成功
 */
bool update(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    Mat &depth,
    Mat &depth_cov2
);

/**
 * 极线搜索
 * @param ref           参考图像
 * @param curr          当前图像
 * @param T_C_R         位姿
 * @param pt_ref        参考图像中点的位置
 * @param depth_mu      深度均值
 * @param depth_cov     深度方差
 * @param pt_curr       当前点
 * @param epipolar_direction  极线方向
 * @return              是否成功
 */
bool epipolarSearch(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    const Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Vector2d &pt_curr,
    Vector2d &epipolar_direction
);

/**
 * 更新深度滤波器
 * @param pt_ref    参考图像点
 * @param pt_curr   当前图像点
 * @param T_C_R     位姿
 * @param epipolar_direction 极线方向
 * @param depth     深度均值
 * @param depth_cov2    深度方向
 * @return          是否成功
 */
bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2
);

/**
 * 计算 NCC 评分
 * @param ref       参考图像
 * @param curr      当前图像
 * @param pt_ref    参考点
 * @param pt_curr   当前点
 * @return          NCC评分
 */
double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr);

// 双线性灰度插值
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt) { // 根据浮点像素坐标获取插值灰度值
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))]; // 定位到左上角整数像素的内存地址
    double xx = pt(0, 0) - floor(pt(0, 0)); // 计算横向小数偏移
    double yy = pt(1, 0) - floor(pt(1, 0)); // 计算纵向小数偏移
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0; // 用四邻域加权求灰度并归一化到 0 到 1
}

// ------------------------------------------------------------------
// 一些小工具
// 显示估计的深度图
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);

// 像素到相机坐标系
inline Vector3d px2cam(const Vector2d px) { // 将像素坐标转换为归一化相机坐标
    return Vector3d(
        (px(0, 0) - cx) / fx, // 根据针孔模型计算归一化 x 坐标
        (px(1, 0) - cy) / fy, // 根据针孔模型计算归一化 y 坐标
        1                     // 归一化平面上的 z 坐标固定为 1
    );
}

// 相机坐标系到像素
inline Vector2d cam2px(const Vector3d p_cam) { // 将相机坐标系三维点投影到像素平面
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx, // 根据针孔模型计算像素 x 坐标
        p_cam(1, 0) * fy / p_cam(2, 0) + cy  // 根据针孔模型计算像素 y 坐标
    );
}

// 检测一个点是否在图像边框内
inline bool inside(const Vector2d &pt) { // 判断像素点是否离图像边界足够远
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height; // 保证后续窗口访问不越界
}

// 显示极线匹配
void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr);

// 显示极线
void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr);

/// 评测深度估计
void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate);
// ------------------------------------------------------------------


int main(int argc, char **argv) {         // 程序入口，命令行需要传入数据集路径
    if (argc != 2) {                      // 检查命令行参数数量是否正确
        cout << "Usage: dense_mapping path_to_test_dataset" << endl; // 参数错误时输出用法
        return -1;                        // 返回 -1 表示参数错误
    }

    // 从数据集读取数据
    vector<string> color_image_files;     // 保存彩色图像文件路径列表
    vector<SE3d> poses_TWC;               // 保存每帧相机到世界坐标系的位姿 T_W_C
    Mat ref_depth;                        // 保存参考帧的深度真值，用于评估
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth); // 读取图像路径、位姿和参考深度
    if (ret == false) {                   // 检查数据集读取是否成功
        cout << "Reading image files failed!" << endl; // 读取失败时输出错误信息
        return -1;                        // 返回 -1 表示数据读取失败
    }
    cout << "read total " << color_image_files.size() << " files." << endl; // 输出成功读取的图像数量

    // 第一张图
    Mat ref = imread(color_image_files[0], 0); // 读取第一帧作为灰度参考图像
    SE3d pose_ref_TWC = poses_TWC[0];    // 取出参考帧的相机到世界位姿
    double init_depth = 3.0;             // 深度初始值
    double init_cov2 = 3.0;              // 方差初始值
    Mat depth(height, width, CV_64F, init_depth); // 初始化整幅深度图，每个像素初始深度为 3 米
    Mat depth_cov2(height, width, CV_64F, init_cov2); // 初始化深度方差图，每个像素初始方差相同

    for (int index = 1; index < color_image_files.size(); index++) { // 从第二帧开始逐帧更新参考帧深度
        cout << "*** loop " << index << " ***" << endl; // 输出当前处理的帧编号
        Mat curr = imread(color_image_files[index], 0); // 读取当前帧灰度图像
        if (curr.data == nullptr) continue; // 如果当前图像读取失败，则跳过这一帧
        SE3d pose_curr_TWC = poses_TWC[index]; // 取出当前帧的相机到世界位姿
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;   // 坐标转换关系： T_C_W * T_W_R = T_C_R
        update(ref, curr, pose_T_C_R, depth, depth_cov2); // 使用当前帧观测更新参考帧深度图
        evaludateDepth(ref_depth, depth); // 与深度真值比较并输出误差
        plotDepth(ref_depth, depth);      // 显示真值深度、估计深度和误差
        imshow("image", curr);            // 显示当前处理的灰度图像
        waitKey(1);                       // 短暂等待，让 OpenCV 窗口刷新
    }

    cout << "estimation returns, saving depth map ..." << endl; // 提示深度估计结束并准备保存
    imwrite("depth.png", depth);          // 将估计深度图写入 depth.png
    cout << "done." << endl;              // 输出程序完成提示

    return 0;                             // 程序正常结束
}

bool readDatasetFiles(
    const string &path,                   // 数据集根目录
    vector<string> &color_image_files,    // 输出：彩色图像文件路径
    std::vector<SE3d> &poses,             // 输出：每帧位姿 T_W_C
    cv::Mat &ref_depth) {                 // 输出：参考帧深度真值
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt"); // 打开轨迹文件
    if (!fin) return false;               // 如果轨迹文件打开失败，返回读取失败

    while (!fin.eof()) {                  // 循环读取轨迹文件中的每一帧记录
        // 数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw ，注意是 TWC 而非 TCW
        string image;                     // 保存当前帧图像文件名
        fin >> image;                     // 读取当前帧图像文件名
        double data[7];                   // 保存 tx,ty,tz,qx,qy,qz,qw 七个数值
        for (double &d:data) fin >> d;    // 依次读取位姿数据

        color_image_files.push_back(path + string("/images/") + image); // 拼接完整图像路径并保存
        poses.push_back(
            SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                 Vector3d(data[0], data[1], data[2]))
        );                               // 用四元数和平移构造 SE3 位姿并保存
        if (!fin.good()) break;           // 如果文件状态异常或到达末尾，退出循环
    }
    fin.close();                          // 关闭轨迹文件

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth"); // 打开参考帧深度真值文件
    ref_depth = cv::Mat(height, width, CV_64F); // 创建双精度深度图矩阵
    if (!fin) return false;               // 如果深度文件打开失败，返回读取失败
    for (int y = 0; y < height; y++)      // 遍历深度图每一行
        for (int x = 0; x < width; x++) { // 遍历深度图每一列
            double depth = 0;             // 临时保存文件中的深度值
            fin >> depth;                 // 读取当前像素深度
            ref_depth.ptr<double>(y)[x] = depth / 100.0; // 将厘米单位转换为米并写入深度图
        }

    return true;                          // 所有数据读取成功
}

// 对整个深度图进行更新
bool update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2) {
    for (int x = boarder; x < width - boarder; x++) // 遍历除边界外的每一列像素
        for (int y = boarder; y < height - boarder; y++) { // 遍历除边界外的每一行像素
            // 遍历每个像素
            if (depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov) // 深度已收敛或发散
                continue;
            // 在极线上搜索 (x,y) 的匹配
            Vector2d pt_curr;             // 保存当前帧中匹配到的像素位置
            Vector2d epipolar_direction;  // 保存当前像素对应的极线方向
            bool ret = epipolarSearch(
                ref,                      // 参考图像
                curr,                     // 当前图像
                T_C_R,                    // 参考帧到当前帧的位姿
                Vector2d(x, y),           // 参考帧当前像素
                depth.ptr<double>(y)[x],  // 当前像素的深度均值
                sqrt(depth_cov2.ptr<double>(y)[x]), // 当前像素的深度标准差
                pt_curr,                  // 输出：当前帧匹配像素
                epipolar_direction        // 输出：极线方向
            );                            // 执行极线搜索

            if (ret == false) // 匹配失败
                continue;

            // 取消该注释以显示匹配
            // showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);

            // 匹配成功，更新深度图
            updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2); // 用匹配结果融合深度估计
        }
    return true;                          // 深度图更新流程执行完成
}

// 极线搜索
// 方法见书 12.2 12.3 两节
bool epipolarSearch(
    const Mat &ref, const Mat &curr,
    const SE3d &T_C_R, const Vector2d &pt_ref,
    const double &depth_mu, const double &depth_cov,
    Vector2d &pt_curr, Vector2d &epipolar_direction) {
    Vector3d f_ref = px2cam(pt_ref);      // 将参考像素转换为相机归一化坐标射线
    f_ref.normalize();                    // 将参考射线单位化
    Vector3d P_ref = f_ref * depth_mu;    // 参考帧的 P 向量

    Vector2d px_mean_curr = cam2px(T_C_R * P_ref); // 按深度均值投影的像素
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov; // 用 3 sigma 范围给出深度搜索区间
    if (d_min < 0.1) d_min = 0.1;        // 限制最小深度，避免出现非物理或过近的深度
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min));    // 按最小深度投影的像素
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max));    // 按最大深度投影的像素

    Vector2d epipolar_line = px_max_curr - px_min_curr;    // 极线（线段形式）
    epipolar_direction = epipolar_line;        // 极线方向
    epipolar_direction.normalize();       // 将极线方向单位化，方便沿线采样
    double half_length = 0.5 * epipolar_line.norm();    // 极线线段的半长度
    if (half_length > 100) half_length = 100;   // 我们不希望搜索太多东西

    // 取消此句注释以显示极线（线段）
    // showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

    // 在极线上搜索，以深度均值点为中心，左右各取半长度
    double best_ncc = -1.0;               // 保存目前找到的最高 NCC 分数
    Vector2d best_px_curr;                // 保存最高 NCC 对应的当前帧像素
    for (double l = -half_length; l <= half_length; l += 0.7) { // l+=sqrt(2)
        Vector2d px_curr = px_mean_curr + l * epipolar_direction;  // 待匹配点
        if (!inside(px_curr))             // 检查待匹配点是否在有效图像区域
            continue;                     // 越界则跳过该候选点
        // 计算待匹配点与参考帧的 NCC
        double ncc = NCC(ref, curr, pt_ref, px_curr); // 计算参考窗口和候选窗口之间的 NCC 相似度
        if (ncc > best_ncc) {             // 如果当前候选点更相似
            best_ncc = ncc;               // 更新最佳 NCC 分数
            best_px_curr = px_curr;       // 更新最佳匹配像素
        }
    }
    if (best_ncc < 0.85f)      // 只相信 NCC 很高的匹配
        return false;
    pt_curr = best_px_curr;               // 输出最佳匹配点
    return true;                          // 极线搜索成功
}

double NCC(
    const Mat &ref, const Mat &curr,
    const Vector2d &pt_ref, const Vector2d &pt_curr) { // 计算两个以浮点像素为中心的小窗口 NCC 相似度
    // 零均值-归一化互相关
    // 先算均值
    double mean_ref = 0, mean_curr = 0;   // 初始化参考窗口和当前窗口的灰度均值累加器
    vector<double> values_ref, values_curr; // 参考帧和当前帧的均值
    for (int x = -ncc_window_size; x <= ncc_window_size; x++) // 遍历 NCC 窗口的横向偏移
        for (int y = -ncc_window_size; y <= ncc_window_size; y++) { // 遍历 NCC 窗口的纵向偏移
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0; // 读取参考图像窗口灰度并归一化
            mean_ref += value_ref;        // 累加参考窗口灰度，用于后续求均值

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y)); // 对当前图像浮点窗口位置做双线性采样
            mean_curr += value_curr;      // 累加当前窗口灰度，用于后续求均值

            values_ref.push_back(value_ref); // 保存参考窗口当前像素灰度
            values_curr.push_back(value_curr); // 保存当前窗口当前像素灰度
        }

    mean_ref /= ncc_area;                 // 计算参考窗口平均灰度
    mean_curr /= ncc_area;                // 计算当前窗口平均灰度

    // 计算 Zero mean NCC
    double numerator = 0, demoniator1 = 0, demoniator2 = 0; // 初始化 NCC 分子和两个方差项
    for (int i = 0; i < values_ref.size(); i++) { // 遍历窗口中的每个采样值
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr); // 计算零均值后的互相关项
        numerator += n;                   // 累加 NCC 分子
        demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref); // 累加参考窗口方差项
        demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr); // 累加当前窗口方差项
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);   // 防止分母出现零
}

bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2) {                    // 使用三角化测量结果更新单个像素的高斯深度估计
    // 用三角化计算深度
    SE3d T_R_C = T_C_R.inverse();         // 求当前帧到参考帧的位姿 T_R_C
    Vector3d f_ref = px2cam(pt_ref);      // 将参考帧匹配像素转换为归一化射线
    f_ref.normalize();                    // 将参考帧射线单位化
    Vector3d f_curr = px2cam(pt_curr);    // 将当前帧匹配像素转换为归一化射线
    f_curr.normalize();                   // 将当前帧射线单位化

    // 方程
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // f2 = R_RC * f_cur
    // 转化成下面这个矩阵方程组
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
    Vector3d t = T_R_C.translation();     // 取当前帧到参考帧变换中的平移
    Vector3d f2 = T_R_C.so3() * f_curr;   // 将当前帧射线旋转到参考帧坐标系
    Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2)); // 构造线性方程右端项
    Matrix2d A;                           // 构造用于求两个射线深度的 2x2 线性系统
    A(0, 0) = f_ref.dot(f_ref);           // 写入 A 矩阵第 1 行第 1 列
    A(0, 1) = -f_ref.dot(f2);             // 写入 A 矩阵第 1 行第 2 列
    A(1, 0) = -A(0, 1);                   // 写入 A 矩阵第 2 行第 1 列
    A(1, 1) = -f2.dot(f2);                // 写入 A 矩阵第 2 行第 2 列
    Vector2d ans = A.inverse() * b;       // 求解参考帧深度和当前帧深度
    Vector3d xm = ans[0] * f_ref;           // ref 侧的结果
    Vector3d xn = t + ans[1] * f2;          // cur 结果
    Vector3d p_esti = (xm + xn) / 2.0;      // P的位置，取两者的平均
    double depth_estimation = p_esti.norm();   // 深度值

    // 计算不确定性（以一个像素为误差）
    Vector3d p = f_ref * depth_estimation; // 根据估计深度得到参考帧三维点
    Vector3d a = p - t;                   // 构造从当前相机中心指向估计点的向量
    double t_norm = t.norm();             // 计算两帧相机中心的基线长度
    double a_norm = a.norm();             // 计算当前相机中心到估计点的距离
    double alpha = acos(f_ref.dot(t) / t_norm); // 计算参考射线和基线之间的夹角
    double beta = acos(-a.dot(t) / (a_norm * t_norm)); // 计算当前射线和反向基线之间的夹角
    Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction); // 将当前匹配点沿极线偏移一个像素以模拟测量误差
    f_curr_prime.normalize();             // 将扰动后的当前帧射线单位化
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm); // 计算扰动射线对应的角度
    double gamma = M_PI - alpha - beta_prime; // 根据三角形内角和计算第三个角
    double p_prime = t_norm * sin(beta_prime) / sin(gamma); // 用正弦定理估计扰动后的深度
    double d_cov = p_prime - depth_estimation; // 用深度扰动量近似测量标准差
    double d_cov2 = d_cov * d_cov;        // 将标准差平方得到测量方差

    // 高斯融合
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))]; // 读取该像素原有深度均值
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))]; // 读取该像素原有深度方差

    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2); // 按方差加权融合新旧深度均值
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2); // 根据高斯融合公式计算新方差

    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse; // 写回融合后的深度均值
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2; // 写回融合后的深度方差

    return true;                          // 深度滤波更新成功
}

// 以下是显示和误差评估相关的辅助函数
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate) { // 显示深度真值、估计结果和误差图
    imshow("depth_truth", depth_truth * 0.4); // 缩放显示深度真值，避免数值过大导致显示过亮
    imshow("depth_estimate", depth_estimate * 0.4); // 缩放显示当前估计深度图
    imshow("depth_error", depth_truth - depth_estimate); // 显示真值与估计值的差异
    waitKey(1);                          // 短暂等待，让 OpenCV 窗口刷新
}

void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate) { // 计算并输出深度估计误差
    double ave_depth_error = 0;     // 平均误差
    double ave_depth_error_sq = 0;      // 平方误差
    int cnt_depth_data = 0;              // 统计参与评估的像素数量
    for (int y = boarder; y < depth_truth.rows - boarder; y++) // 遍历去掉边界后的每一行
        for (int x = boarder; x < depth_truth.cols - boarder; x++) { // 遍历去掉边界后的每一列
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x]; // 计算当前像素深度误差
            ave_depth_error += error;    // 累加深度误差
            ave_depth_error_sq += error * error; // 累加深度误差平方
            cnt_depth_data++;            // 有效评估像素计数加一
        }
    ave_depth_error /= cnt_depth_data;   // 计算平均深度误差
    ave_depth_error_sq /= cnt_depth_data; // 计算平均平方误差

    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl; // 输出评估指标
}

void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr) { // 可视化参考点和当前帧匹配点
    Mat ref_show, curr_show;             // 创建用于显示的彩色图像
    cv::cvtColor(ref, ref_show, COLOR_GRAY2BGR); // 将参考灰度图转换为 BGR，便于画彩色圆
    cv::cvtColor(curr, curr_show, COLOR_GRAY2BGR); // 将当前灰度图转换为 BGR，便于画彩色圆

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2); // 在参考图像上标出参考像素
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2); // 在当前图像上标出匹配像素

    imshow("ref", ref_show);             // 显示带标记的参考图像
    imshow("curr", curr_show);           // 显示带标记的当前图像
    waitKey(1);                          // 短暂等待，让 OpenCV 窗口刷新
}

void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr) { // 可视化参考点和当前帧中的极线段

    Mat ref_show, curr_show;             // 创建用于显示的彩色图像
    cv::cvtColor(ref, ref_show, COLOR_GRAY2BGR); // 将参考灰度图转换为 BGR
    cv::cvtColor(curr, curr_show, COLOR_GRAY2BGR); // 将当前灰度图转换为 BGR

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2); // 在参考图像上标出参考像素
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2); // 标出极线最小深度端点
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2); // 标出极线最大深度端点
    cv::line(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
             Scalar(0, 255, 0), 1);      // 在当前图像上绘制极线线段

    imshow("ref", ref_show);             // 显示带参考点的参考图像
    imshow("curr", curr_show);           // 显示带极线的当前图像
    waitKey(1);                          // 短暂等待，让 OpenCV 窗口刷新
}
