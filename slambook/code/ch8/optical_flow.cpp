// 文件头注释：记录该示例程序的创建信息。
// Created by Xiang on 2017/12/19.
//

#include <opencv2/opencv.hpp>  // 引入 OpenCV 主头文件，提供图像读写、特征检测、显示和光流等功能。
#include <string>              // 引入 string 类型，用于保存图像文件路径。
#include <chrono>              // 引入计时工具，用于统计光流计算耗时。
#include <Eigen/Core>          // 引入 Eigen 核心模块，用于矩阵和向量计算。
#include <Eigen/Dense>         // 引入 Eigen 稠密矩阵模块，用于求解线性方程。

using namespace std;           // 使用 std 命名空间，简化 vector、cout、string 等写法。
using namespace cv;            // 使用 cv 命名空间，简化 Mat、KeyPoint、Point2f 等 OpenCV 类型写法。

string file_1 = "./LK1.png";   // 第一张图像的相对路径。
string file_2 = "./LK2.png";   // 第二张图像的相对路径。

/// 光流跟踪器类：保存单层 LK 光流所需的数据，并提供并行计算接口。
class OpticalFlowTracker {
public:  // 公有成员区，外部可以构造对象并调用成员函数。
    OpticalFlowTracker(
        const Mat &img1_,                  // 第一张灰度图像，作为参考图像。
        const Mat &img2_,                  // 第二张灰度图像，作为待跟踪图像。
        const vector<KeyPoint> &kp1_,      // 第一张图像中的关键点。
        vector<KeyPoint> &kp2_,            // 第二张图像中的关键点估计结果。
        vector<bool> &success_,            // 每个关键点是否跟踪成功的标记。
        bool inverse_ = true,              // 是否使用反向 LK：true 表示使用 inverse compositional 思路。
        bool has_initial_ = false) :       // kp2 是否已经提供初始估计。
        img1(img1_),                       // 用传入的第一张图像引用初始化成员 img1。
        img2(img2_),                       // 用传入的第二张图像引用初始化成员 img2。
        kp1(kp1_),                         // 用传入的第一帧关键点引用初始化成员 kp1。
        kp2(kp2_),                         // 用传入的第二帧关键点引用初始化成员 kp2。
        success(success_),                 // 用传入的成功标记引用初始化成员 success。
        inverse(inverse_),                 // 记录是否使用反向 LK。
        has_initial(has_initial_) {}       // 记录是否有第二帧关键点初值。

    void calculateOpticalFlow(const Range &range);  // 对指定下标范围内的关键点计算光流，供 parallel_for_ 调用。

private:  // 私有成员区，只允许类内部访问。
    const Mat &img1;                 // 第一张图像的引用，避免复制图像数据。
    const Mat &img2;                 // 第二张图像的引用，避免复制图像数据。
    const vector<KeyPoint> &kp1;     // 第一张图像关键点的引用。
    vector<KeyPoint> &kp2;           // 第二张图像关键点结果的引用。
    vector<bool> &success;           // 跟踪成功标记的引用。
    bool inverse = true;             // 是否使用反向 LK 的成员开关。
    bool has_initial = false;        // 是否使用 kp2 中已有位置作为初值。
};

/**
 * 单层光流函数声明：只在原始分辨率上执行一次 LK 跟踪。
 * @param [in] img1 第一张图像，即参考帧。
 * @param [in] img2 第二张图像，即目标帧。
 * @param [in] kp1 第一张图像中的关键点。
 * @param [in|out] kp2 第二张图像中的关键点；为空时会按 kp1 的位置初始化。
 * @param [out] success 每个关键点是否跟踪成功。
 * @param [in] inverse 是否使用反向 LK 形式。
 */
void OpticalFlowSingleLevel(
    const Mat &img1,                    // 第一张灰度图像。
    const Mat &img2,                    // 第二张灰度图像。
    const vector<KeyPoint> &kp1,        // 第一张图像中的关键点集合。
    vector<KeyPoint> &kp2,              // 第二张图像中跟踪得到的关键点集合。
    vector<bool> &success,              // 输出每个关键点是否成功。
    bool inverse = false,               // 默认不使用反向形式。
    bool has_initial_guess = false      // 默认认为 kp2 没有初始估计。
);

/**
 * 多层金字塔光流函数声明：从低分辨率到高分辨率逐层细化跟踪结果。
 * 图像金字塔会在函数内部创建。
 * @param [in] img1 第一张图像。
 * @param [in] img2 第二张图像。
 * @param [in] kp1 第一张图像中的关键点。
 * @param [out] kp2 第二张图像中跟踪得到的关键点。
 * @param [out] success 每个关键点是否跟踪成功。
 * @param [in] inverse 是否启用反向 LK 形式。
 */
void OpticalFlowMultiLevel(
    const Mat &img1,                    // 第一张灰度图像。
    const Mat &img2,                    // 第二张灰度图像。
    const vector<KeyPoint> &kp1,        // 第一张图像的关键点集合。
    vector<KeyPoint> &kp2,              // 第二张图像的关键点结果集合。
    vector<bool> &success,              // 输出每个关键点是否成功。
    bool inverse = false                // 默认不使用反向形式。
);

/**
 * 从灰度图中取一个亚像素位置的灰度值，使用双线性插值。
 * @param img 输入灰度图像。
 * @param x 横坐标，可以是小数。
 * @param y 纵坐标，可以是小数。
 * @return 该亚像素位置插值得到的灰度值。
 */

inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // 边界检查：如果坐标越界，就把它截断到图像内部。
    if (x < 0) x = 0;                            // x 小于 0 时取左边界。
    if (y < 0) y = 0;                            // y 小于 0 时取上边界。
    if (x >= img.cols - 1) x = img.cols - 2;     // x 太靠右时留出右侧插值像素。
    if (y >= img.rows - 1) y = img.rows - 2;     // y 太靠下时留出下侧插值像素。

    float xx = x - floor(x);                     // 计算 x 的小数部分，作为水平方向插值权重。
    float yy = y - floor(y);                     // 计算 y 的小数部分，作为竖直方向插值权重。
    int x_a1 = std::min(img.cols - 1, int(x) + 1);  // 右侧相邻像素的 x 坐标。
    int y_a1 = std::min(img.rows - 1, int(y) + 1);  // 下侧相邻像素的 y 坐标。

    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)
    + xx * yy * img.at<uchar>(y_a1, x_a1);       // 按四邻域权重加权，得到双线性插值结果。
}

int main(int argc, char **argv) {  // 程序入口函数。

    // 读取图像，注意第二个参数为 0，表示以灰度图读取，类型是 CV_8UC1 而不是 CV_8UC3。
    Mat img1 = imread(file_1, 0);  // 读取第一张图像作为参考帧。
    Mat img2 = imread(file_2, 0);  // 读取第二张图像作为目标帧。

    // 提取第一张图像中的关键点，这里使用 GFTT，也就是 Good Features To Track。
    vector<KeyPoint> kp1;  // 保存第一张图像检测出的关键点。
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);  // 创建 GFTT 检测器：最多 500 个点，质量阈值 0.01，最小间距 20。
    // 参数含义：最大点数、角点质量比例系数、关键点之间的最小容忍距离（避免扎堆）。
    detector->detect(img1, kp1);  // 在第一张图像上检测关键点并存入 kp1。

    // 下面开始把第一张图像中的关键点跟踪到第二张图像中。
    // 先使用单层 LK 光流，作为对比实验之一。
    vector<KeyPoint> kp2_single;  // 保存单层 LK 在第二张图像中跟踪得到的关键点。
    vector<bool> success_single;  // 保存单层 LK 每个关键点是否跟踪成功。
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);  // 执行单层 LK 光流。

    // 再测试多层金字塔 LK 光流。
    vector<KeyPoint> kp2_multi;  // 保存多层 LK 在第二张图像中跟踪得到的关键点。
    vector<bool> success_multi;  // 保存多层 LK 每个关键点是否跟踪成功。
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();  // 记录多层 LK 开始时间。
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);  // 执行多层 LK，最后一个 true 表示使用反向形式。
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();  // 记录多层 LK 结束时间。
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);  // 计算多层 LK 总耗时。
    cout << "optical flow by gauss-newton: " << time_used.count() << endl;  // 输出自实现 Gauss-Newton 光流耗时。

    // 使用 OpenCV 自带的金字塔 LK 光流作为验证和对比。
    vector<Point2f> pt1, pt2;  // pt1 保存第一张图像点坐标，pt2 保存 OpenCV 跟踪到的第二张图像点坐标。
    for (auto &kp: kp1) pt1.push_back(kp.pt);  // 将 KeyPoint 类型转换为 Point2f 类型。
    vector<uchar> status;  // OpenCV 输出的跟踪状态，非零表示成功。
    vector<float> error;   // OpenCV 输出的误差。
    t1 = chrono::steady_clock::now();  // 记录 OpenCV 光流开始时间。
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);  // 调用 OpenCV 的金字塔 LK 光流。
    t2 = chrono::steady_clock::now();  // 记录 OpenCV 光流结束时间。
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);  // 计算 OpenCV 光流耗时。
    cout << "optical flow by opencv: " << time_used.count() << endl;  // 输出 OpenCV 光流耗时。

    // 绘制三种方法的跟踪结果，方便肉眼比较差异。
    Mat img2_single;  // 用于显示单层 LK 跟踪结果的彩色图。
    cv::cvtColor(img2, img2_single, COLOR_GRAY2BGR);  // 将灰度图转为 BGR 彩色图，便于画绿色圆点和线段。
    for (int i = 0; i < kp2_single.size(); i++) {  // 遍历单层 LK 的所有跟踪点。
        if (success_single[i]) {  // 只绘制跟踪成功的点。
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);  // 在第二帧跟踪位置画绿色圆点。
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));  // 从第一帧位置到第二帧位置画绿色线段。
        }
    }

    Mat img2_multi;  // 用于显示多层 LK 跟踪结果的彩色图。
    cv::cvtColor(img2, img2_multi, COLOR_GRAY2BGR);  // 将第二张灰度图转成 BGR 图像。
    for (int i = 0; i < kp2_multi.size(); i++) {  // 遍历多层 LK 的所有跟踪点。
        if (success_multi[i]) {  // 只绘制跟踪成功的点。
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);  // 在第二帧跟踪位置画绿色圆点。
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));  // 绘制从初始位置到跟踪位置的运动线。
        }
    }

    Mat img2_CV;  // 用于显示 OpenCV LK 跟踪结果的彩色图。
    cv::cvtColor(img2, img2_CV, COLOR_GRAY2BGR);  // 将第二张灰度图转成 BGR 图像。
    for (int i = 0; i < pt2.size(); i++) {  // 遍历 OpenCV 输出的所有跟踪点。
        if (status[i]) {  // 只绘制 OpenCV 认为跟踪成功的点。
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);  // 在第二帧跟踪位置画绿色圆点。
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));  // 绘制从第一帧位置到第二帧位置的绿色线段。
        }
    }

    cv::imshow("tracked single level", img2_single);  // 显示单层 LK 跟踪结果窗口。
    cv::imshow("tracked multi level", img2_multi);    // 显示多层 LK 跟踪结果窗口。
    cv::imshow("tracked by opencv", img2_CV);         // 显示 OpenCV LK 跟踪结果窗口。
    cv::waitKey(0);                                   // 等待按键，防止窗口立即关闭。

    return 0;  // 返回 0，表示程序正常结束。
}

void OpticalFlowSingleLevel(
    const Mat &img1,                 // 第一张灰度图像。
    const Mat &img2,                 // 第二张灰度图像。
    const vector<KeyPoint> &kp1,     // 第一张图像中的关键点。
    vector<KeyPoint> &kp2,           // 第二张图像中的关键点结果。
    vector<bool> &success,           // 每个关键点的跟踪成功标记。
    bool inverse,                    // 是否使用反向 LK。
    bool has_initial) {              // 是否已有 kp2 初始估计。
    kp2.resize(kp1.size());          // 调整 kp2 大小，使其与 kp1 的关键点数量一致。
    success.resize(kp1.size());      // 调整 success 大小，使每个关键点都有一个状态位。
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);  // 构造光流跟踪器对象。
    parallel_for_(Range(0, kp1.size()),
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));  // 并行处理所有关键点。
}

void OpticalFlowTracker::calculateOpticalFlow(const Range &range) {  // 对 range 内的关键点执行 LK 跟踪。
    // 光流迭代参数。
    int half_patch_size = 4;  // 半个 patch 的大小，实际 patch 为 8x8 像素。
    int iterations = 10;      // 每个关键点最多进行 10 次 Gauss-Newton 迭代。
    for (size_t i = range.start; i < range.end; i++) {  // 遍历当前线程负责的关键点下标范围。
        auto kp = kp1[i];     // 取出第一张图像中的当前关键点。
        double dx = 0, dy = 0;  // dx、dy 是需要估计的位移量，初始设为 0。
        if (has_initial) {      // 如果外部已经提供第二帧初值，则从初值开始迭代。
            dx = kp2[i].pt.x - kp.pt.x;  // 用第二帧初值和第一帧坐标的差初始化 x 位移。
            dy = kp2[i].pt.y - kp.pt.y;  // 用第二帧初值和第一帧坐标的差初始化 y 位移。
        }

        double cost = 0, lastCost = 0;  // 当前代价和上一轮代价，用于判断迭代是否变差。
        bool succ = true;               // 当前关键点是否成功跟踪。

        // Gauss-Newton 迭代：最小化两个 patch 之间的灰度误差。
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();  // Hessian 矩阵，近似为 J^T J。
        Eigen::Vector2d b = Eigen::Vector2d::Zero();  // 右端项，近似为 -J^T error。
        Eigen::Vector2d J;                            // 单个像素误差对位移 dx、dy 的雅可比。
        for (int iter = 0; iter < iterations; iter++) {  // 进行最多 iterations 次迭代。
            if (inverse == false) {                    // 正向 LK 中，雅可比随 dx、dy 变化，每次都要重新计算 H 和 b。
                H = Eigen::Matrix2d::Zero();           // 清零 Hessian。
                b = Eigen::Vector2d::Zero();           // 清零右端项。
            } else {
                // 反向 LK 中，雅可比只和第一张图像有关，H 在第一轮之后可以复用。
                b = Eigen::Vector2d::Zero();           // 每轮误差会变化，所以 b 仍需清零重算。
            }

            cost = 0;  // 清零当前迭代的累计误差。

            // 遍历关键点周围的 patch，计算总误差、雅可比、Hessian 和 b。
            for (int x = -half_patch_size; x < half_patch_size; x++)  // 遍历 patch 内的 x 偏移。
                for (int y = -half_patch_size; y < half_patch_size; y++) {  // 遍历 patch 内的 y 偏移。
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);  // 两帧对应 patch 像素的灰度残差。
                    if (inverse == false) {  // 正向 LK：在第二张图像当前估计位置处计算图像梯度。
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                   GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                   GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );  // 用中心差分计算 img2 的 x、y 梯度，并加负号得到残差对位移的雅可比。
                    } else if (iter == 0) {  // 反向 LK：只在第一轮计算第一张图像 patch 的梯度。
                        // 反向模式中，J 在所有迭代中保持不变。
                        // 注意：该 J 不随 dx、dy 更新而改变，因此可以只计算一次，后续只更新误差。
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                   GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                   GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );  // 用中心差分计算 img1 的 x、y 梯度，并加负号得到反向形式的雅可比。
                    }
                    // 根据当前像素残差和雅可比累加线性方程 H * update = b。
                    b += -error * J;       // 累加右端项 -J^T error，这里 J 是 2x1 向量。
                    cost += error * error; // 累加平方误差，作为当前 patch 的代价。
                    if (inverse == false || iter == 0) {  // 正向每轮更新 H；反向只在第一轮更新 H。
                        // 同时更新 Hessian。
                        H += J * J.transpose();  // 累加 J * J^T，得到 2x2 Hessian 近似。
                    }
                }

            // 求解增量 update，也就是本轮对 dx、dy 的修正量。
            Eigen::Vector2d update = H.ldlt().solve(b);  // 使用 LDLT 分解求解 2x2 线性方程。

            if (std::isnan(update[0])) {  // 如果求解结果出现 NaN，说明该点无法可靠跟踪。
                // 当 patch 全黑或全白、缺少梯度时，H 可能不可逆，此时会出现这种情况。
                cout << "update is nan" << endl;  // 输出调试信息。
                succ = false;                     // 标记该关键点跟踪失败。
                break;                            // 退出当前关键点的迭代。
            }

            if (iter > 0 && cost > lastCost) {  // 如果本轮代价比上一轮更大，说明继续迭代可能变差。
                break;                          // 提前停止迭代。
            }

            // 更新当前位移估计。
            dx += update[0];   // 更新 x 方向位移。
            dy += update[1];   // 更新 y 方向位移。
            lastCost = cost;   // 保存本轮代价，供下一轮比较。
            succ = true;       // 当前迭代正常，暂时认为跟踪成功。

            if (update.norm() < 1e-2) {  // 如果增量很小，认为已经收敛。
                // 收敛后无需继续迭代。
                break;                   // 提前退出迭代。
            }
        }

        success[i] = succ;  // 记录当前关键点的跟踪状态。

        // 设置第二张图像中的关键点坐标。
        kp2[i].pt = kp.pt + Point2f(dx, dy);  // 第一帧位置加上估计位移，得到第二帧位置。
    }
}

void OpticalFlowMultiLevel(
    const Mat &img1,                 // 第一张灰度图像。
    const Mat &img2,                 // 第二张灰度图像。
    const vector<KeyPoint> &kp1,     // 第一张图像中的关键点。
    vector<KeyPoint> &kp2,           // 第二张图像中的关键点跟踪结果。
    vector<bool> &success,           // 每个关键点的跟踪成功标记。
    bool inverse) {                  // 是否使用反向 LK。

    // 金字塔参数。
    int pyramids = 4;                         // 金字塔层数，一共 4 层。
    double pyramid_scale = 0.5;               // 每往上一层，图像长宽缩小为原来的一半。
    double scales[] = {1.0, 0.5, 0.25, 0.125};  // 每层相对于原图的尺度系数。

    // 创建两张图像各自的图像金字塔。
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();  // 记录创建金字塔的开始时间。
    vector<Mat> pyr1, pyr2;  // pyr1 保存第一张图像金字塔，pyr2 保存第二张图像金字塔。
    for (int i = 0; i < pyramids; i++) {  // 逐层构建金字塔。
        if (i == 0) {  // 第 0 层就是原始图像。
            pyr1.push_back(img1);  // 将第一张原图放入金字塔第 0 层。
            pyr2.push_back(img2);  // 将第二张原图放入金字塔第 0 层。
        } else {
            Mat img1_pyr, img2_pyr;  // 保存当前层缩放后的两张图像。
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));  // 由上一层缩小得到第一张图像当前层。
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));  // 由上一层缩小得到第二张图像当前层。
            pyr1.push_back(img1_pyr);  // 把第一张图像当前层加入金字塔。
            pyr2.push_back(img2_pyr);  // 把第二张图像当前层加入金字塔。
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();  // 记录创建金字塔结束时间。
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);  // 计算创建金字塔耗时。
    cout << "build pyramid time: " << time_used.count() << endl;  // 输出创建金字塔的耗时。

    // 在金字塔上执行由粗到细的 LK 跟踪。
    vector<KeyPoint> kp1_pyr, kp2_pyr;  // 分别保存当前金字塔层中的参考点和跟踪点。
    for (auto &kp:kp1) {  // 遍历原始图像中的每个关键点。
        auto kp_top = kp;  // 复制一个关键点，用于变换到最顶层金字塔坐标。
        kp_top.pt *= scales[pyramids - 1];  // 将原图坐标缩放到最顶层坐标系。
        kp1_pyr.push_back(kp_top);  // 保存最顶层中的参考关键点。
        kp2_pyr.push_back(kp_top);  // 第二帧初值也先设为同一位置，即假设初始位移为 0。
    }

    for (int level = pyramids - 1; level >= 0; level--) {  // 从最粗的顶层开始，一直跟踪到原图层。
        // 由粗到细：粗层估计大位移，细层逐步修正。
        success.clear();  // 清空上一层的成功标记，准备写入当前层结果。
        t1 = chrono::steady_clock::now();  // 记录当前层跟踪开始时间。
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);  // 在当前层执行单层 LK，并使用 kp2_pyr 作为初始值。
        t2 = chrono::steady_clock::now();  // 记录当前层跟踪结束时间。
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);  // 计算当前层跟踪耗时。
        cout << "track pyr " << level << " cost time: " << time_used.count() << endl;  // 输出当前金字塔层的跟踪耗时。

        if (level > 0) {  // 如果还没有到原图层，需要把点坐标放大到下一层。
            for (auto &kp: kp1_pyr)  // 遍历参考关键点。
                kp.pt /= pyramid_scale;  // 坐标除以 0.5，相当于放大 2 倍，进入下一层坐标系。
            for (auto &kp: kp2_pyr)  // 遍历跟踪关键点。
                kp.pt /= pyramid_scale;  // 同样把第二帧估计点放大 2 倍，作为下一层初值。
        }
    }

    for (auto &kp: kp2_pyr)  // 遍历最终在原图层得到的跟踪点。
        kp2.push_back(kp);   // 将最终结果写入输出 kp2。
}
