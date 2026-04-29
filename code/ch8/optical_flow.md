# `optical_flow.cpp` 代码精读

本文围绕 `optical_flow.cpp` 展开，目标有三条线：

1. 逐行解释代码在做什么。
2. 从 C 和 C++ 的区别出发理解代码中的 C++ 语法。
3. 从 SLAM 理论角度理解 Lucas-Kanade 光流、反向光流和图像金字塔。

## 一、程序整体流程

这份代码实现了一个手写的稀疏 Lucas-Kanade 光流跟踪器，并与 OpenCV 自带的 `calcOpticalFlowPyrLK` 做对比。

整体流程如下：

1. 读取两张灰度图 `LK1.png` 和 `LK2.png`。
2. 在第一张图中用 GFTT 检测角点。
3. 用单层 LK 光流跟踪角点。
4. 用多层图像金字塔 LK 光流跟踪角点。
5. 用 OpenCV 的金字塔 LK 作为对照。
6. 把三种跟踪结果画出来。

在 SLAM 前端中，这类光流常用于在相邻帧之间跟踪特征点，获得二维点对应关系，然后继续用于位姿估计、三角化、局部建图或视觉里程计。

## 二、头文件、命名空间与全局变量

```cpp
// Created by Xiang on 2017/12/19.
```

第 1-3 行只是注释，不参与编译。

```cpp
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
```

第 5 行引入 OpenCV 主头文件，提供 `Mat`、`KeyPoint`、`GFTTDetector`、`imshow`、`calcOpticalFlowPyrLK` 等功能。

第 6 行引入 C++ 标准库字符串 `std::string`。C 中常用 `char *` 表示字符串，C++ 中更推荐 `std::string`，它能自动管理内存并提供更安全的字符串操作。

第 7 行引入 C++ 计时库 `std::chrono`，用于统计手写光流和 OpenCV 光流耗时。

第 8-9 行引入 Eigen，后面用 `Eigen::Matrix2d` 和 `Eigen::Vector2d` 表示 Gauss-Newton 中的 Hessian、右端项和更新量。

```cpp
using namespace std;
using namespace cv;
```

第 11-12 行把 `std` 和 `cv` 命名空间展开。这样可以写 `vector`、`string`、`Mat`，而不用写 `std::vector`、`std::string`、`cv::Mat`。

C 语言没有命名空间。C++ 引入命名空间是为了避免不同库中函数名、类名冲突。

```cpp
string file_1 = "./LK1.png";
string file_2 = "./LK2.png";
```

第 14-15 行定义两个全局字符串变量，表示两张输入图片路径。

C 写法可能是：

```c
const char *file_1 = "./LK1.png";
```

C++ 中 `string` 更像一个对象，自动管理字符数组，使用起来更安全。

## 三、`OpticalFlowTracker` 类

```cpp
class OpticalFlowTracker {
public:
    ...
private:
    ...
};
```

第 18 行定义一个类。C 语言通常用 `struct` 加函数指针或独立函数组织数据和行为；C++ 的 `class` 可以把数据和操作数据的方法封装在一起。

`public` 中的成员可以被外部访问，`private` 中的成员只能在类内部访问。这体现了 C++ 的封装思想。

```cpp
OpticalFlowTracker(
    const Mat &img1_,
    const Mat &img2_,
    const vector<KeyPoint> &kp1_,
    vector<KeyPoint> &kp2_,
    vector<bool> &success_,
    bool inverse_ = true, bool has_initial_ = false) :
    img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
    has_initial(has_initial_) {}
```

第 20-28 行是构造函数。构造函数名字和类名相同，没有返回值，用于创建对象时初始化成员变量。

`const Mat &img1_` 表示以常量引用传入图像：

- `&` 是引用，避免复制整张图像。
- `const` 表示函数内部不会修改这张图。
- C 中通常用指针传大对象，例如 `const Mat *img1`；C++ 引用更自然，也避免空指针语义。

冒号后面的部分叫成员初始化列表：

```cpp
img1(img1_), img2(img2_), kp1(kp1_)
```

它直接初始化类成员。对于引用成员，如 `const Mat &img1`，必须在初始化列表中绑定，不能先默认构造再赋值。

```cpp
void calculateOpticalFlow(const Range &range);
```

第 30 行声明成员函数，用于计算一段关键点的光流。`Range` 来自 OpenCV，配合 `parallel_for_` 做并行。

```cpp
const Mat &img1;
const Mat &img2;
const vector<KeyPoint> &kp1;
vector<KeyPoint> &kp2;
vector<bool> &success;
bool inverse = true;
bool has_initial = false;
```

第 33-39 行是类的私有成员。

- `img1`、`img2` 是两帧图像。
- `kp1` 是第一帧中的关键点。
- `kp2` 是第二帧中估计出来的关键点位置。
- `success` 记录每个点是否跟踪成功。
- `inverse` 控制使用正向 LK 还是反向 LK。
- `has_initial` 表示 `kp2` 中是否已有初值。

这里大量使用引用，是为了让 `OpticalFlowTracker` 直接操作外部传入的 `kp2` 和 `success`，不用复制大数组。

## 四、函数声明

```cpp
void OpticalFlowSingleLevel(...);
void OpticalFlowMultiLevel(...);
```

第 51-78 行是函数声明，也叫函数原型。C 和 C++ 都有这种写法。因为 `main` 在前面调用这些函数，而函数实现放在 `main` 后面，所以需要先声明。

`OpticalFlowSingleLevel` 做单层 LK 光流。

`OpticalFlowMultiLevel` 做图像金字塔 LK 光流。

参数注释中的 `[in]`、`[out]`、`[in|out]` 说明数据流向：

- `[in]`：只读输入。
- `[out]`：函数负责写输出。
- `[in|out]`：既作为输入初值，又会被修改为输出。

## 五、双线性插值 `GetPixelValue`

```cpp
inline float GetPixelValue(const cv::Mat &img, float x, float y)
```

第 88 行定义一个内联函数，用于读取浮点坐标处的灰度值。

为什么需要浮点坐标？光流估计出的位移 `dx, dy` 不是整数，所以第二帧上的对应位置通常落在像素格点之间。必须用插值估计该位置的亮度。

`inline` 是 C++ 中的内联建议，表示函数很短，编译器可以考虑把函数体展开到调用处，减少函数调用开销。

```cpp
if (x < 0) x = 0;
if (y < 0) y = 0;
if (x >= img.cols - 1) x = img.cols - 2;
if (y >= img.rows - 1) y = img.rows - 2;
```

第 89-93 行做边界检查。因为双线性插值要访问 `(x, y)` 附近的四个像素，如果坐标落在最外边，访问 `x + 1` 或 `y + 1` 会越界。

```cpp
float xx = x - floor(x);
float yy = y - floor(y);
```

第 95-96 行计算浮点坐标的小数部分。若 `x = 10.3`，则 `xx = 0.3`。

```cpp
int x_a1 = std::min(img.cols - 1, int(x) + 1);
int y_a1 = std::min(img.rows - 1, int(y) + 1);
```

第 97-98 行得到右侧和下侧像素索引，同时再次用 `std::min` 防止越界。

`std::min` 是 C++ 标准库函数；C 中通常需要自己写宏或三目表达式。

```cpp
return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
     + xx * (1 - yy) * img.at<uchar>(y, x_a1)
     + (1 - xx) * yy * img.at<uchar>(y_a1, x)
     + xx * yy * img.at<uchar>(y_a1, x_a1);
```

第 100-103 行是双线性插值公式。它用左上、右上、左下、右下四个像素的加权和估计浮点坐标亮度。

SLAM 直接法和光流法都经常依赖这种亚像素插值，因为优化变量连续，而图像采样离散。

## 六、`main` 主函数

```cpp
int main(int argc, char **argv)
```

第 106 行是程序入口。`argc` 和 `argv` 是命令行参数，本程序没有使用它们。

```cpp
Mat img1 = imread(file_1, 0);
Mat img2 = imread(file_2, 0);
```

第 109-110 行读取两张灰度图。`0` 表示以单通道灰度图读入。

OpenCV 中 `Mat` 是 C++ 类，内部使用引用计数管理图像数据。C 风格图像结构通常需要手动管理内存，而 `Mat` 可以自动释放资源。

```cpp
vector<KeyPoint> kp1;
Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);
detector->detect(img1, kp1);
```

第 113-115 行检测第一帧关键点。

- `vector<KeyPoint>` 是 C++ 动态数组，类似 C 中手动 `malloc` 的数组，但能自动扩容和释放。
- `Ptr<GFTTDetector>` 是 OpenCV 的智能指针类型，类似 `std::shared_ptr`。
- `GFTTDetector::create(...)` 是静态成员函数，创建 Good Features To Track 角点检测器。
- `detector->detect(img1, kp1)` 调用检测器对象的方法，把角点写入 `kp1`。

GFTT 本质上选择适合跟踪的角点。LK 光流要求局部 patch 有足够纹理，如果 patch 近似纯色，Hessian 会退化，位移不可观。

```cpp
vector<KeyPoint> kp2_single;
vector<bool> success_single;
OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);
```

第 119-121 行调用手写单层 LK。输出是第二帧中的关键点位置和成功标志。

```cpp
vector<KeyPoint> kp2_multi;
vector<bool> success_multi;
chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
cout << "optical flow by gauss-newton: " << time_used.count() << endl;
```

第 124-130 行调用手写多层 LK，并统计耗时。

`chrono::steady_clock` 是单调时钟，适合计时。`auto` 让编译器自动推导变量类型，这是 C++ 特性，C 语言没有。

这里传入 `true` 表示使用 inverse formulation，即反向 LK。

```cpp
vector<Point2f> pt1, pt2;
for (auto &kp: kp1) pt1.push_back(kp.pt);
vector<uchar> status;
vector<float> error;
```

第 133-136 行准备 OpenCV 光流接口需要的数据。

`for (auto &kp: kp1)` 是 C++ 范围 for 循环。它遍历 `kp1` 中每个关键点。`&` 表示引用，避免复制。

`kp.pt` 是关键点的二维坐标，类型是 `Point2f`。

```cpp
cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
```

第 138 行调用 OpenCV 的金字塔 LK 光流。

`status[i]` 表示第 `i` 个点是否跟踪成功，`error[i]` 通常表示该点的误差。

```cpp
Mat img2_single;
cv::cvtColor(img2, img2_single, COLOR_GRAY2BGR);
```

第 144-145 行把灰度图转换为 BGR 彩色图，方便用绿色画圆和线。

```cpp
for (int i = 0; i < kp2_single.size(); i++) {
    if (success_single[i]) {
        cv::circle(...);
        cv::line(...);
    }
}
```

第 146-151 行把单层 LK 成功跟踪的点画出来：

- `circle` 在第二帧位置画圆。
- `line` 从第一帧点位置连到第二帧点位置，显示运动方向和大小。

第 153-169 行对多层 LK 和 OpenCV LK 做同样的可视化。

```cpp
cv::imshow("tracked single level", img2_single);
cv::imshow("tracked multi level", img2_multi);
cv::imshow("tracked by opencv", img2_CV);
cv::waitKey(0);
```

第 171-174 行显示三张结果图。`waitKey(0)` 表示一直等待键盘输入，否则窗口可能一闪而过。

```cpp
return 0;
```

第 176 行返回 0，表示程序正常结束。

## 七、单层光流接口 `OpticalFlowSingleLevel`

```cpp
kp2.resize(kp1.size());
success.resize(kp1.size());
```

第 186-187 行把输出数组调整到和输入关键点数量一致。

`resize` 是 `std::vector` 的成员函数。C 中如果数组大小变化，需要手动重新分配内存。

```cpp
OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
```

第 188 行创建跟踪器对象，并把图像、关键点、结果数组绑定进去。

```cpp
parallel_for_(Range(0, kp1.size()),
              std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
```

第 189-190 行使用 OpenCV 并行接口，把关键点范围分成若干段并行处理。

`std::bind` 是 C++ 函数适配器。它把成员函数 `calculateOpticalFlow` 和对象 `tracker` 绑定起来，形成一个可调用对象。

SLAM 中特征点之间的光流跟踪相互独立，所以天然适合并行。

## 八、核心优化 `calculateOpticalFlow`

```cpp
int half_patch_size = 4;
int iterations = 10;
```

第 195-196 行设置 patch 半径和最大迭代次数。

`half_patch_size = 4` 表示使用大约 `8 x 8` 的图像块。代码循环是 `[-4, 3]`，所以实际是 8 个像素宽。

```cpp
for (size_t i = range.start; i < range.end; i++)
```

第 197 行遍历当前线程负责的一段关键点。

`size_t` 是无符号整数类型，常用于数组下标和大小。

```cpp
auto kp = kp1[i];
double dx = 0, dy = 0;
```

第 198-199 行取出当前关键点，并初始化待估计的二维位移。

`auto kp = kp1[i]` 会复制一个关键点。如果写 `auto &kp = kp1[i]` 则是引用。

```cpp
if (has_initial) {
    dx = kp2[i].pt.x - kp.pt.x;
    dy = kp2[i].pt.y - kp.pt.y;
}
```

第 200-203 行如果有初值，就用 `kp2` 和 `kp1` 的差作为初始位移。

在金字塔光流中，粗层估计会传给细层作为初值，因此这里很重要。

```cpp
double cost = 0, lastCost = 0;
bool succ = true;
```

第 205-206 行初始化当前点的优化代价和成功标志。

```cpp
Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
Eigen::Vector2d b = Eigen::Vector2d::Zero();
Eigen::Vector2d J;
```

第 209-211 行定义 Gauss-Newton 需要的矩阵和向量：

- `H` 是近似 Hessian，即 `J^T J`。
- `b` 是右端项，即 `-J^T e`。
- `J` 是误差对位移的雅可比。

在这个问题里，每个点只估计二维平移 `dx, dy`，所以 `H` 是 `2 x 2`，`b` 和 `J` 是 2 维向量。

```cpp
for (int iter = 0; iter < iterations; iter++)
```

第 212 行开始 Gauss-Newton 迭代。

```cpp
if (inverse == false) {
    H = Eigen::Matrix2d::Zero();
    b = Eigen::Vector2d::Zero();
} else {
    b = Eigen::Vector2d::Zero();
}
```

第 213-219 行根据正向或反向形式决定是否重置 `H`。

正向 LK 中，雅可比取第二帧当前位置的图像梯度，随着 `dx, dy` 改变而改变，所以每轮都要重算 `H`。

反向 LK 中，雅可比取第一帧模板图像的梯度，与 `dx, dy` 无关，所以 `H` 第一轮算完后可以复用，后续只重置 `b`。

```cpp
cost = 0;
```

第 221 行清空当前迭代的总误差。

```cpp
for (int x = -half_patch_size; x < half_patch_size; x++)
    for (int y = -half_patch_size; y < half_patch_size; y++)
```

第 224-225 行遍历当前关键点周围的 patch。

LK 光流不是只比较一个像素，而是比较一个局部窗口。这样能提供更多约束，使二维位移可估计。

```cpp
double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
               GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
```

第 226-227 行计算光度误差：

```text
e = I1(x, y) - I2(x + dx, y + dy)
```

核心假设是灰度不变假设：同一个空间点在相邻帧中的亮度近似不变。

```cpp
if (inverse == false) {
    J = -1.0 * Eigen::Vector2d(
        0.5 * (I2(u + 1, v) - I2(u - 1, v)),
        0.5 * (I2(u, v + 1) - I2(u, v - 1))
    );
}
```

第 228-234 行是正向 LK 的雅可比。由于误差定义为 `I1 - I2`，所以对 `dx, dy` 求导时有负号：

```text
J = - dI2 / d[u, v]
```

图像梯度用中心差分近似。

```cpp
else if (iter == 0) {
    J = -1.0 * Eigen::Vector2d(
        0.5 * (I1(x + 1, y) - I1(x - 1, y)),
        0.5 * (I1(x, y + 1) - I1(x, y - 1))
    );
}
```

第 235-244 行是反向 LK 的雅可比。它使用第一帧模板图像的梯度，只在 `iter == 0` 时计算。

反向法的好处是节省计算量：同一个 patch 的梯度和 Hessian 在迭代中不变。

```cpp
b += -error * J;
cost += error * error;
```

第 246-247 行累加正规方程右端项和总误差。

Gauss-Newton 线性化后解：

```text
H update = b
H = sum(J J^T)
b = sum(-e J)
```

这里代码把 `J` 存成列向量，所以写成 `J * J.transpose()`。

```cpp
if (inverse == false || iter == 0) {
    H += J * J.transpose();
}
```

第 248-251 行累加 Hessian。正向法每轮更新，反向法只在第一轮更新。

```cpp
Eigen::Vector2d update = H.ldlt().solve(b);
```

第 255 行求解 `H update = b`。

`ldlt()` 是 Eigen 提供的矩阵分解方法，适合求解对称矩阵线性方程。这里 `H` 是 `2 x 2`，求解非常快。

```cpp
if (std::isnan(update[0])) {
    cout << "update is nan" << endl;
    succ = false;
    break;
}
```

第 257-262 行检查更新量是否为 NaN。

如果 patch 纹理不足，例如全黑或全白，图像梯度接近 0，`H` 可能不可逆，导致求解失败。

这正对应了 SLAM 中“为什么要跟踪角点而不是任意像素”：只有梯度结构足够丰富的区域，位移才稳定可估。

```cpp
if (iter > 0 && cost > lastCost) {
    break;
}
```

第 264-266 行如果代价变大，就停止迭代。说明当前更新可能让估计变差。

```cpp
dx += update[0];
dy += update[1];
lastCost = cost;
succ = true;
```

第 269-272 行更新位移估计，并记录本轮代价。

```cpp
if (update.norm() < 1e-2) {
    break;
}
```

第 274-277 行如果更新量很小，就认为收敛。

```cpp
success[i] = succ;
kp2[i].pt = kp.pt + Point2f(dx, dy);
```

第 280-283 行写入当前点是否成功，以及第二帧中的估计位置。

## 九、多层金字塔光流 `OpticalFlowMultiLevel`

```cpp
int pyramids = 4;
double pyramid_scale = 0.5;
double scales[] = {1.0, 0.5, 0.25, 0.125};
```

第 296-298 行设置 4 层图像金字塔。每往上一层，图像宽高缩小为原来的一半。

金字塔的作用是处理大位移。原图中 16 像素的运动，在 0.125 倍尺度下只剩 2 像素，更容易满足 LK 的小运动假设。

```cpp
vector<Mat> pyr1, pyr2;
for (int i = 0; i < pyramids; i++) {
    ...
}
```

第 302-316 行构建两张图像各自的金字塔。

```cpp
if (i == 0) {
    pyr1.push_back(img1);
    pyr2.push_back(img2);
}
```

第 304-306 行第 0 层就是原图。

```cpp
cv::resize(pyr1[i - 1], img1_pyr,
           cv::Size(pyr1[i - 1].cols * pyramid_scale,
                    pyr1[i - 1].rows * pyramid_scale));
```

第 309-312 行从上一层缩放得到下一层。

`push_back` 是 `vector` 的追加操作。C 中如果要维护动态数组，需要自己记录容量和长度。

```cpp
cout << "build pyramid time: " << time_used.count() << endl;
```

第 317-319 行输出构建金字塔耗时。

```cpp
vector<KeyPoint> kp1_pyr, kp2_pyr;
for (auto &kp:kp1) {
    auto kp_top = kp;
    kp_top.pt *= scales[pyramids - 1];
    kp1_pyr.push_back(kp_top);
    kp2_pyr.push_back(kp_top);
}
```

第 322-328 行把原图关键点缩放到最顶层金字塔坐标。

`kp2_pyr` 初始设置为 `kp1_pyr`，表示初始位移为 0。

```cpp
for (int level = pyramids - 1; level >= 0; level--)
```

第 330 行从最粗层往最细层跟踪，也就是 coarse-to-fine。

```cpp
success.clear();
OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
```

第 332-334 行在当前金字塔层执行单层 LK。

`has_initial = true` 很关键：当前层的 `kp2_pyr` 不是空白输出，而是上一层传下来的初始估计。

```cpp
if (level > 0) {
    for (auto &kp: kp1_pyr)
        kp.pt /= pyramid_scale;
    for (auto &kp: kp2_pyr)
        kp.pt /= pyramid_scale;
}
```

第 339-344 行准备进入下一层更高分辨率图像。因为下一层尺度扩大 2 倍，所以关键点坐标也要除以 `0.5`，即乘以 2。

```cpp
for (auto &kp: kp2_pyr)
    kp2.push_back(kp);
```

第 347-348 行把最终原图尺度下的跟踪结果写入输出 `kp2`。

## 十、从 C 到 C++：这份代码里的重点语法

### 1. `std::vector` 代替手动动态数组

C 中常见写法：

```c
KeyPoint *kp = malloc(sizeof(KeyPoint) * n);
```

C++ 中使用：

```cpp
vector<KeyPoint> kp1;
kp1.push_back(kp);
kp1.resize(n);
```

好处是自动管理内存、自动记录大小、支持范围 for 遍历。

### 2. 引用 `&` 代替一部分指针传参

```cpp
const Mat &img1
vector<KeyPoint> &kp2
```

引用像变量别名，调用时不用写取地址符。`const Mat &` 表示只读引用，适合传大对象。

C 中通常写：

```c
void func(const Mat *img1, Vector *kp2);
```

### 3. 类封装数据和函数

`OpticalFlowTracker` 把图像、关键点、成功标志和计算函数封装在一起。

C 通常把结构体和函数分开：

```c
typedef struct {
    Mat *img1;
    Mat *img2;
} OpticalFlowTracker;

void calculateOpticalFlow(OpticalFlowTracker *tracker);
```

C++ 可以让对象自己带行为：

```cpp
tracker.calculateOpticalFlow(range);
```

### 4. 构造函数和初始化列表

```cpp
OpticalFlowTracker(...) : img1(img1_), img2(img2_) {}
```

这是 C++ 对象初始化方式。尤其是引用成员和 `const` 成员，必须通过初始化列表初始化。

### 5. 默认参数

```cpp
bool inverse = false
```

C++ 函数可以设置默认参数。调用时如果不传，就自动使用默认值。C 没有原生默认参数。

### 6. `auto` 自动类型推导

```cpp
auto time_used = ...
auto kp_top = kp;
```

`auto` 让编译器根据右侧表达式推导类型。它让复杂模板类型更简洁，但学习时要能判断真实类型。

### 7. 范围 for

```cpp
for (auto &kp: kp1)
```

这是 C++11 的范围循环，适合遍历容器。`auto &` 表示遍历时拿到元素引用，不复制。

### 8. 智能指针

```cpp
Ptr<GFTTDetector> detector
```

OpenCV 的 `Ptr` 会自动释放对象。C 中如果手动 `malloc` 或创建对象，通常还要手动释放。

### 9. 命名空间

`std::`、`cv::`、`Eigen::` 都是命名空间。它们让不同库可以拥有同名类或函数而不冲突。

### 10. Eigen 矩阵类型

```cpp
Eigen::Matrix2d
Eigen::Vector2d
```

这些是 C++ 模板库提供的固定大小矩阵。C 中通常需要二维数组或手动矩阵库。

## 十一、SLAM 理论解释

### 1. 光流在 SLAM 中解决什么问题

视觉 SLAM 前端需要知道同一个空间点在相邻图像中的位置对应关系。

特征匹配通常是：

```text
检测特征 -> 计算描述子 -> 描述子匹配
```

光流跟踪则是：

```text
上一帧已有点 -> 在下一帧局部区域内直接优化位置
```

它不依赖描述子，速度快，适合视频相邻帧这种运动较小、外观变化不大的场景。

### 2. LK 光流的基本假设

LK 光流主要依赖三个假设：

1. 灰度不变：同一个点在两帧中的亮度近似相同。
2. 小运动：相邻帧位移较小，便于一阶线性化。
3. 局部一致：一个小 patch 内的像素共享同一个二维位移。

代码中的误差就是：

```text
e_i(dx, dy) = I1(x_i, y_i) - I2(x_i + dx, y_i + dy)
```

目标是最小化一个 patch 内所有像素的平方误差：

```text
min_{dx,dy} sum_i e_i(dx, dy)^2
```

### 3. 为什么可以用 Gauss-Newton

误差中包含图像采样函数 `I2(x + dx, y + dy)`，它对 `dx, dy` 是非线性的。

Gauss-Newton 每次在当前估计附近做一阶近似：

```text
e(dx + delta) ≈ e(dx) + J delta
```

然后求一个增量 `delta`，让误差平方和变小。

代码中的：

```cpp
H += J * J.transpose();
b += -error * J;
update = H.ldlt().solve(b);
```

正是 Gauss-Newton 正规方程。

### 4. 为什么角点更适合光流

如果 patch 是纯色区域，往任何方向移动亮度都差不多，无法判断真实位移。

如果 patch 是边缘，只能稳定估计垂直边缘方向的位移，沿边缘方向不稳定，这叫孔径问题。

角点在两个方向都有明显梯度，`H` 更接近满秩，位移估计更稳定。

所以代码用 GFTT 检测角点，再做 LK 跟踪。

### 5. 正向 LK 和反向 LK

正向 LK 使用第二帧当前位置的梯度：

```text
J = -∇I2(x + dx, y + dy)
```

每次 `dx, dy` 更新后，采样位置变化，梯度和 Hessian 也要重算。

反向 LK 使用第一帧模板的梯度：

```text
J = -∇I1(x, y)
```

模板不变，所以 `J` 和 `H` 可以复用。代码通过 `inverse == true` 和 `iter == 0` 实现这一点。

### 6. 图像金字塔为什么有效

LK 的小运动假设要求初值离真实解不太远。如果两帧位移太大，单层 LK 可能收敛到错误位置或直接失败。

图像金字塔把图像逐层缩小。在粗层中，大位移被缩小，优化更容易收敛。得到粗层位移后，再逐层放大到更细层细化。

这就是代码中的 coarse-to-fine：

```text
0.125 倍尺度估计 -> 0.25 倍尺度细化 -> 0.5 倍尺度细化 -> 原图尺度细化
```

### 7. 和 OpenCV `calcOpticalFlowPyrLK` 的关系

手写实现展示了核心数学：

- patch 光度误差
- 图像梯度雅可比
- Gauss-Newton 迭代
- 反向法复用 Hessian
- 金字塔 coarse-to-fine

OpenCV 的 `calcOpticalFlowPyrLK` 是工程化实现，通常包含更多边界处理、终止条件、窗口参数、异常情况处理和优化，因此更稳定。

## 十二、代码中容易注意的细节

1. `GetPixelValue` 假设图像是单通道 `uchar`，所以 `imread(file, 0)` 很重要。
2. `kp2.resize(kp1.size())` 会保证每个输入点都有一个输出槽位。
3. 单层光流默认没有初值，多层光流每层都使用上一层结果作为初值。
4. `success.clear()` 后，`OpticalFlowSingleLevel` 内部会重新 `resize`。
5. 反向法中 `H` 只在第一轮累加，节省了重复计算。
6. 如果 `update` 是 NaN，通常意味着 patch 梯度不足或 Hessian 退化。
7. 可视化中的线段不是三维运动，只是二维图像平面上的像素位移。

## 十三、一句话总结

这份代码用 C++ 和 OpenCV/Eigen 实现了稀疏 LK 光流：先检测适合跟踪的角点，再在局部 patch 上最小化两帧灰度误差，用 Gauss-Newton 求解二维位移，并通过反向法和图像金字塔提升速度与鲁棒性。
