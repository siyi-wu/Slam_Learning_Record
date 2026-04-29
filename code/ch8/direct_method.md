# `direct_method.cpp` 代码精读

本文围绕 `direct_method.cpp` 展开，目标有三条线：

1. 逐行解释代码在做什么。
2. 从 C 和 C++ 的区别出发理解代码中的 C++ 语法。
3. 从 SLAM 理论角度理解稀疏直接法、光度误差、SE(3) 位姿优化和图像金字塔。

## 一、程序整体流程

这份代码实现了一个基于已知深度的稀疏直接法位姿估计。它不先提取和匹配特征描述子，而是随机选择参考帧中的若干像素，利用双目视差图得到这些像素的深度，然后在后续图像中寻找一个相机位姿 `T21`，使参考帧像素块投影到当前帧后的灰度误差最小。

整体流程如下：

1. 读取参考左图 `left.png` 和视差图 `disparity.png`。
2. 在参考图中随机选取 2000 个像素。
3. 用双目公式 `depth = fx * baseline / disparity` 把像素视差转成深度。
4. 对 `000001.png` 到 `000005.png` 逐帧估计相对于参考帧的位姿。
5. 位姿估计使用多层图像金字塔，从粗到细优化。
6. 每一层内部用 Gauss-Newton 最小化光度误差。
7. 每次迭代把所有像素点的雅可比、Hessian 和误差并行累加，再求解 6 维李代数更新量。
8. 把参考像素投影到当前图像并可视化。

SLAM 视角下，这段程序属于视觉里程计前端中的直接法示例。它使用的是“稀疏直接法”：只用一部分像素点，但误差项是灰度误差，而不是特征点重投影误差。

## 二、头文件、命名空间与类型别名

```cpp
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>
```

第 1 行引入 OpenCV 主头文件，提供 `cv::Mat`、`cv::imread`、`cv::resize`、`cv::parallel_for_`、`cv::imshow` 等图像处理和并行工具。

第 2 行引入 Sophus 的 `SE3`，后面用 `Sophus::SE3d` 表示三维刚体变换。SLAM 中相机位姿通常属于李群 SE(3)，它由旋转和平移组成。

第 3 行引入 Boost 的格式化字符串工具。代码中用 `boost::format("./%06d.png")` 生成 `000001.png` 这类文件名。

第 4 行引入 Pangolin。当前代码没有直接使用 Pangolin 的函数，链接中也包含了 Pangolin，可能是本章其他示例或原始工程模板留下来的依赖。

C 语言通常通过 `.h` 头文件提供函数声明，C++ 的头文件还会提供类、模板、命名空间、运算符重载等内容。例如 `Sophus::SE3d` 是一个 C++ 类，不是 C 风格结构体加函数的组合。

```cpp
using namespace std;
```

第 6 行展开标准库命名空间。这样后面可以写 `vector`、`string`、`cout`，不用写 `std::vector`、`std::string`、`std::cout`。

C 语言没有命名空间。C++ 引入命名空间是为了解决大型工程和第三方库中的名字冲突问题。

```cpp
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
```

第 8 行定义类型别名 `VecVector2d`，表示一个存放 `Eigen::Vector2d` 的动态数组。

这里有一个特殊点：`Eigen::aligned_allocator<Eigen::Vector2d>`。Eigen 的某些固定大小向量需要内存对齐，以便使用 SIMD 加速。如果直接写 `vector<Eigen::Vector2d>`，在一些平台上可能出现内存对齐问题。

C 中也有 `typedef`，但这里的类型本身来自 C++ 模板：

```cpp
vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
```

`vector<T>` 是 C++ 标准库模板容器。C 中如果想存动态数组，通常需要手写指针、容量、长度和 `malloc/free` 逻辑。

```cpp
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
double baseline = 0.573;
string left_file = "./left.png";
string disparity_file = "./disparity.png";
boost::format fmt_others("./%06d.png");
```

第 10-17 行定义全局变量。

`fx, fy, cx, cy` 是相机内参。投影模型为：

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

`baseline` 是双目相机基线，用于由视差计算深度：

```text
depth = fx * baseline / disparity
```

`left_file` 和 `disparity_file` 是输入路径。C 中常用 `const char *` 表示字符串；C++ 中 `std::string` 是对象，能自动管理内存。

`fmt_others` 是 Boost 的格式化对象。`"./%06d.png"` 表示整数用 6 位宽度、不足补 0，例如 `1` 变成 `000001`。

```cpp
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
```

第 19-22 行给常用矩阵类型起别名。

- `Matrix6d` 是 6x6 矩阵，用作 Gauss-Newton 的 Hessian 近似矩阵 `H`。
- `Matrix26d` 是 2x6 矩阵，用作像素坐标对李代数扰动的雅可比。
- `Vector6d` 是 6x1 向量，用作右端项 `b` 和位姿更新量。

SLAM 中 SE(3) 位姿的李代数是 6 维，通常前三维表示平移扰动，后三维表示旋转扰动，所以优化变量是 6 维。

## 三、`JacobianAccumulator` 类

```cpp
class JacobianAccumulator {
public:
    ...
private:
    ...
};
```

第 24-71 行定义 `JacobianAccumulator` 类。这个类负责在并行循环中为一批像素点累加 Hessian、右端项和代价。

C 语言中通常用 `struct` 存数据，再写一组函数接收结构体指针。C++ 的 `class` 可以把数据和操作这些数据的函数放在一起，这叫封装。

```cpp
JacobianAccumulator(
    const cv::Mat &img1_,
    const cv::Mat &img2_,
    const VecVector2d &px_ref_,
    const vector<double> depth_ref_,
    Sophus::SE3d &T21_) :
    img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {
    projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
}
```

第 27-35 行是构造函数。构造函数名字和类名相同，没有返回值，在创建对象时自动调用。

参数含义如下：

- `img1_`：参考图像。
- `img2_`：当前图像。
- `px_ref_`：参考图像中选取的像素点。
- `depth_ref_`：这些像素点对应的深度。
- `T21_`：从参考帧到当前帧的位姿。

`const cv::Mat &` 是常量引用。引用可以避免复制整张图像，`const` 表示函数内部不修改它。C 语言通常用指针表达类似语义，例如 `const Mat *img1`，但指针可能为空，引用通常表达“必须绑定到一个对象”。

冒号后面的部分是成员初始化列表：

```cpp
img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_)
```

引用成员必须在初始化列表里绑定，不能先默认构造再赋值。

注意 `depth_ref_` 这里按值传入，成员 `depth_ref` 也按值保存，会复制一份深度数组。更高效的写法可以是 `const vector<double> &depth_ref_`，并让成员也保存引用。

函数体中：

```cpp
projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
```

为每个参考像素准备一个投影点，初始值都是 `(0, 0)`。后面可视化时会用它画出投影位置。

```cpp
void accumulate_jacobian(const cv::Range &range);
```

第 37-38 行声明成员函数。`cv::Range` 表示一个索引区间，OpenCV 的 `parallel_for_` 会把总任务拆成多个区间，并行调用这个函数。

```cpp
Matrix6d hessian() const { return H; }
Vector6d bias() const { return b; }
double cost_func() const { return cost; }
VecVector2d projected_points() const { return projection; }
```

第 40-50 行是几个 getter 函数，用来返回累计结果。

函数末尾的 `const` 表示这个成员函数不会修改对象内部状态。C 没有成员函数，也没有这种对象级别的 `const` 约束。

这些函数直接写在类定义内部，天然适合被编译器内联。

```cpp
void reset() {
    H = Matrix6d::Zero();
    b = Vector6d::Zero();
    cost = 0;
}
```

第 52-57 行把 Hessian、右端项和总代价清零。每次 Gauss-Newton 迭代开始前都要重新累加。

`Matrix6d::Zero()` 是 Eigen 的静态成员函数，返回全零矩阵。C 中如果用数组表示矩阵，通常需要 `memset` 或循环清零。

```cpp
const cv::Mat &img1;
const cv::Mat &img2;
const VecVector2d &px_ref;
const vector<double> depth_ref;
Sophus::SE3d &T21;
VecVector2d projection;
```

第 59-65 行是私有成员。

- `img1` 和 `img2` 是两张图。
- `px_ref` 是参考像素。
- `depth_ref` 是参考像素的深度。
- `T21` 是当前估计的位姿，保存为引用。
- `projection` 保存每个参考点在当前图像中的投影。

`private` 是 C++ 的访问控制。外部不能直接访问这些成员，只能通过类提供的公共函数访问。这有助于保持对象内部状态一致。

```cpp
std::mutex hessian_mutex;
Matrix6d H = Matrix6d::Zero();
Vector6d b = Vector6d::Zero();
double cost = 0;
```

第 67-70 行定义并行累加时共享的数据。

`std::mutex` 是互斥锁。多个线程会同时计算自己的局部 Hessian 和误差，但最终写入共享的 `H`、`b`、`cost` 时必须加锁，否则会产生数据竞争。

`Matrix6d H = Matrix6d::Zero();` 是 C++11 起支持的类内成员初始化。C 语言结构体不能这样直接给成员设置默认值。

## 四、函数声明

```cpp
void DirectPoseEstimationMultiLayer(...);
void DirectPoseEstimationSingleLayer(...);
```

第 73-103 行是函数声明。因为 `main` 在前面调用这些函数，而函数实现放在后面，所以需要先告诉编译器函数签名。

`DirectPoseEstimationSingleLayer` 在一层图像上直接优化位姿。

`DirectPoseEstimationMultiLayer` 构建图像金字塔，并从粗到细调用单层优化。

SLAM 理论上，单层直接法在初值不好或运动较大时容易发散。金字塔能先在低分辨率图像上处理大位移，再逐层细化。

## 五、双线性插值 `GetPixelValue`

```cpp
inline float GetPixelValue(const cv::Mat &img, float x, float y)
```

第 105-106 行定义内联函数，从图像中读取浮点坐标处的灰度值。

直接法投影出来的像素坐标通常不是整数。例如某个三维点投影到当前图像可能是 `(313.42, 180.77)`。图像只在整数网格有采样值，所以需要插值。

`inline` 是 C++ 的内联建议，适合这种频繁调用的小函数。

```cpp
if (x < 0) x = 0;
if (y < 0) y = 0;
if (x >= img.cols) x = img.cols - 1;
if (y >= img.rows) y = img.rows - 1;
```

第 107-111 行做边界截断，把坐标限制在图像范围内。

这里有一个值得注意的小坑：后面会访问 `data[1]`、`data[img.step]` 和 `data[img.step + 1]`，所以双线性插值实际需要右侧和下侧像素。如果 `x` 被截断到 `img.cols - 1` 或 `y` 被截断到 `img.rows - 1`，理论上可能越界访问。更稳妥的写法是把上界限制到 `img.cols - 2` 和 `img.rows - 2`。本程序在调用前通常避开边界，所以多数情况下不会触发。

```cpp
uchar *data = &img.data[int(y) * img.step + int(x)];
```

第 112 行取出左上角整数像素的内存地址。

- `img.data` 是 OpenCV 图像原始数据指针。
- `img.step` 是每一行占用的字节数。
- `int(y) * img.step + int(x)` 定位到第 `y` 行、第 `x` 列。

这行非常接近 C 风格指针操作。C++ 的 OpenCV 也支持更安全的 `img.at<uchar>(y, x)`，但直接指针访问更快。

```cpp
float xx = x - floor(x);
float yy = y - floor(y);
```

第 113-114 行计算小数部分。若 `x = 10.3`，则 `xx = 0.3`。

```cpp
return float(
    (1 - xx) * (1 - yy) * data[0] +
    xx * (1 - yy) * data[1] +
    (1 - xx) * yy * data[img.step] +
    xx * yy * data[img.step + 1]
);
```

第 115-120 行是双线性插值公式。

四个参与插值的像素分别是：

- `data[0]`：左上。
- `data[1]`：右上。
- `data[img.step]`：左下。
- `data[img.step + 1]`：右下。

权重由 `xx` 和 `yy` 决定。距离某个像素越近，该像素权重越大。

## 六、`main` 主函数

```cpp
int main(int argc, char **argv)
```

第 123 行是程序入口。`argc` 和 `argv` 是命令行参数，本程序没有使用它们。

```cpp
cv::Mat left_img = cv::imread(left_file, 0);
cv::Mat disparity_img = cv::imread(disparity_file, 0);
```

第 125-126 行读取参考左图和视差图。第二个参数 `0` 表示按灰度图读取。

`cv::Mat` 是 C++ 类，内部使用引用计数管理图像数据。C 风格图像结构通常需要手动创建和释放内存。

```cpp
cv::RNG rng;
int nPoints = 2000;
int boarder = 20;
VecVector2d pixels_ref;
vector<double> depth_ref;
```

第 128-133 行准备随机采样。

- `cv::RNG rng` 是 OpenCV 随机数生成器对象。
- `nPoints = 2000` 表示随机选 2000 个参考像素。
- `boarder = 20` 应该是 `border`，这里变量名拼错但不影响编译。
- `pixels_ref` 保存参考帧像素坐标。
- `depth_ref` 保存对应深度。

从 C++ 语法看，`VecVector2d pixels_ref;` 会调用默认构造函数创建一个空动态数组。C 中需要手动维护数组容量和长度。

```cpp
for (int i = 0; i < nPoints; i++) {
    int x = rng.uniform(boarder, left_img.cols - boarder);
    int y = rng.uniform(boarder, left_img.rows - boarder);
    int disparity = disparity_img.at<uchar>(y, x);
    double depth = fx * baseline / disparity;
    depth_ref.push_back(depth);
    pixels_ref.push_back(Eigen::Vector2d(x, y));
}
```

第 135-143 行循环随机选点并计算深度。

`rng.uniform(a, b)` 生成 `[a, b)` 范围内的随机整数。代码避免选到边缘像素，因为后面要取 patch 和做插值。

`disparity_img.at<uchar>(y, x)` 读取视差图中一个像素。`uchar` 是无符号 8 位整数。

双目深度公式为：

```text
Z = f * b / d
```

其中 `f` 是焦距，`b` 是基线，`d` 是视差。视差越大，深度越小；视差越小，深度越远。

这里也有一个工程细节：如果 `disparity == 0`，会出现除以 0，得到无穷大深度。更严谨的实现应该跳过视差为 0 或过小的点。

`push_back` 是 `vector` 的成员函数，向动态数组末尾追加元素。C 中如果数组容量不够，需要手动 `realloc`。

```cpp
Sophus::SE3d T_cur_ref;
```

第 145-146 行定义当前帧相对于参考帧的位姿。默认构造的 `Sophus::SE3d` 通常表示单位变换。

变量名 `T_cur_ref` 表示从 reference frame 到 current frame 的变换，也就是代码中函数参数 `T21` 的语义。

```cpp
for (int i = 1; i < 6; i++) {
    cv::Mat img = cv::imread((fmt_others % i).str(), 0);
    // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
}
return 0;
```

第 148-154 行依次读取 `000001.png` 到 `000005.png`，估计它们相对于参考帧的位姿。

`(fmt_others % i).str()` 是 Boost format 的用法。`% i` 把整数填入格式字符串，`.str()` 转成 `std::string`。

被注释的单层函数可以用于对比，多层函数更稳健。

注意 `T_cur_ref` 在循环外定义，并传引用进入函数。每一帧估计结束后，下一帧会沿用上一帧的位姿作为初值。这符合视觉里程计中“相邻帧运动连续”的假设。

## 七、单层直接法位姿估计

```cpp
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21)
```

第 157-162 行实现单层直接法。

参数含义：

- `img1`：参考图像。
- `img2`：当前图像。
- `px_ref`：参考图像中的像素点。
- `depth_ref`：参考像素对应深度。
- `T21`：从参考帧到当前帧的位姿，既是输入初值，也是输出结果。

`Sophus::SE3d &T21` 使用非 const 引用，因为函数内部会更新位姿。C 中一般会传指针来表达可修改输出参数。

```cpp
const int iterations = 10;
double cost = 0, lastCost = 0;
auto t1 = chrono::steady_clock::now();
JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);
```

第 164-167 行设置迭代次数、初始化代价、记录开始时间，并创建雅可比累加器。

`auto` 让编译器自动推导变量类型。这里 `t1` 的真实类型是 `chrono::steady_clock::time_point`。C 语言没有 `auto` 类型推导，现代 C++ 中 `auto` 常用于类型很长但语义清楚的表达式。

```cpp
for (int iter = 0; iter < iterations; iter++) {
```

第 169 行开始 Gauss-Newton 迭代。

```cpp
jaco_accu.reset();
cv::parallel_for_(cv::Range(0, px_ref.size()),
                  std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
```

第 170-172 行清空上一轮累计结果，然后并行计算当前位姿下所有像素点的误差和雅可比。

`std::bind` 把成员函数 `JacobianAccumulator::accumulate_jacobian` 绑定到对象 `jaco_accu` 上，使它可以作为普通可调用对象传给 `parallel_for_`。

`std::placeholders::_1` 表示 `parallel_for_` 传进来的 `cv::Range` 参数会放在这里。

C 语言没有成员函数、函数对象和这种标准库绑定机制。C 中要做类似事情，通常需要函数指针和 `void *userdata`。

```cpp
Matrix6d H = jaco_accu.hessian();
Vector6d b = jaco_accu.bias();
```

第 173-174 行取出累加得到的线性方程：

```text
H * update = b
```

这里 `H = sum(J J^T)`，`b = sum(-e J)`，对应最小化光度误差平方和。

```cpp
Vector6d update = H.ldlt().solve(b);;
T21 = Sophus::SE3d::exp(update) * T21;
cost = jaco_accu.cost_func();
```

第 176-179 行求解更新量并更新位姿。

`H.ldlt().solve(b)` 使用 LDLT 分解求解线性方程。由于 Gauss-Newton 的 Hessian 近似矩阵通常是对称半正定的，LDLT 是常见选择。

`Sophus::SE3d::exp(update)` 把 6 维李代数向量映射到 SE(3) 李群上的位姿增量。更新写成：

```text
T <- exp(update) * T
```

这是左乘扰动。意思是在当前估计的左侧施加一个小的 SE(3) 增量。

第 177 行末尾有两个分号 `;;`，多出来的分号是空语句，不影响程序。

```cpp
if (std::isnan(update[0])) {
    cout << "update is nan" << endl;
    break;
}
```

第 181-185 行检查更新量是否为 NaN。如果 patch 没有纹理，或者 Hessian 奇异，求解可能失败。

直接法需要图像梯度。如果所有点都落在纯色区域，灰度对位姿变化不敏感，优化问题会退化。

```cpp
if (iter > 0 && cost > lastCost) {
    cout << "cost increased: " << cost << ", " << lastCost << endl;
    break;
}
```

第 186-189 行如果代价上升就停止。Gauss-Newton 没有强制保证每一步都下降，尤其在初值较差或线性化误差较大时，代价可能变大。

```cpp
if (update.norm() < 1e-3) {
    break;
}
```

第 190-193 行如果更新量足够小，认为收敛。

`update.norm()` 是 Eigen 向量的二范数。C 中要自己写循环求平方和再开方。

```cpp
lastCost = cost;
cout << "iteration: " << iter << ", cost: " << cost << endl;
```

第 195-196 行记录本轮代价并输出迭代信息。

```cpp
cout << "T21 = \n" << T21.matrix() << endl;
auto t2 = chrono::steady_clock::now();
auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
cout << "direct method for single layer: " << time_used.count() << endl;
```

第 199-202 行输出最终位姿矩阵和耗时。

`T21.matrix()` 把 Sophus 的 SE(3) 对象转成 4x4 齐次变换矩阵。

```cpp
cv::Mat img2_show;
cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR);
VecVector2d projection = jaco_accu.projected_points();
```

第 204-207 行准备可视化。灰度图先转成 BGR 彩色图，这样可以画绿色点和线。

```cpp
for (size_t i = 0; i < px_ref.size(); ++i) {
    auto p_ref = px_ref[i];
    auto p_cur = projection[i];
    if (p_cur[0] > 0 && p_cur[1] > 0) {
        cv::circle(...);
        cv::line(...);
    }
}
```

第 208-216 行遍历所有点，把当前图像中的投影点画出来，并用线连接参考像素坐标和当前投影坐标。

`size_t` 是无符号整数类型，适合表示数组大小和索引。`auto p_ref` 和 `auto p_cur` 让编译器推导为 `Eigen::Vector2d`。

从几何上看，如果位姿估计正确，参考点按照深度和位姿投影到当前帧后，应该落在同一个真实场景点的图像位置附近。

```cpp
cv::imshow("current", img2_show);
cv::waitKey();
```

第 217-218 行显示结果并等待按键。

## 八、雅可比累加：直接法的核心

```cpp
void JacobianAccumulator::accumulate_jacobian(const cv::Range &range)
```

第 221 行实现类成员函数。`JacobianAccumulator::` 表示这个函数属于 `JacobianAccumulator` 类。C 语言没有作用域解析运算符 `::`。

这个函数是整段代码的核心：它把选中的参考像素投影到当前帧，计算光度误差，对位姿求雅可比，并累计正规方程。

```cpp
const int half_patch_size = 1;
int cnt_good = 0;
Matrix6d hessian = Matrix6d::Zero();
Vector6d bias = Vector6d::Zero();
double cost_tmp = 0;
```

第 223-228 行定义局部变量。

`half_patch_size = 1` 表示每个参考点使用一个 `3x3` patch，因为 `x` 和 `y` 会从 `-1` 遍历到 `1`。

`hessian`、`bias` 和 `cost_tmp` 是线程局部累计量。先在局部变量里累计，最后再加锁合并到共享变量，可以减少锁竞争。

```cpp
for (size_t i = range.start; i < range.end; i++) {
```

第 230 行遍历当前线程负责的一段参考点。

```cpp
Eigen::Vector3d point_ref =
    depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
```

第 232-234 行把参考图像中的像素反投影成参考相机坐标系下的三维点。

像素坐标 `(u, v)` 和深度 `Z` 反投影为：

```text
X = Z * (u - cx) / fx
Y = Z * (v - cy) / fy
Z = Z
```

这正是代码中的：

```text
depth * [(u - cx) / fx, (v - cy) / fy, 1]
```

```cpp
Eigen::Vector3d point_cur = T21 * point_ref;
```

第 235 行用当前估计的位姿把三维点从参考相机坐标系变换到当前相机坐标系。

Sophus 重载了 `operator*`，所以可以直接写 `T21 * point_ref`。C 语言没有运算符重载，需要调用类似 `transform_point(T21, point_ref)` 的函数。

```cpp
if (point_cur[2] < 0)
    continue;
```

第 236-237 行如果点在当前相机后方，跳过。相机坐标系中 `Z` 应该为正。

```cpp
float u = fx * point_cur[0] / point_cur[2] + cx, v = fy * point_cur[1] / point_cur[2] + cy;
```

第 239 行把当前相机坐标系下的三维点投影回当前图像。

投影模型是：

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

```cpp
if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size ||
    v > img2.rows - half_patch_size)
    continue;
```

第 240-242 行检查投影点是否离边界太近。因为后面要取 patch 和计算图像梯度，所以投影点附近必须有足够像素。

```cpp
projection[i] = Eigen::Vector2d(u, v);
```

第 244 行保存投影结果，用于最后画图。

```cpp
double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
    Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
cnt_good++;
```

第 245-247 行取出当前帧坐标系下的三维坐标，并预计算 `Z^2`、`1/Z`、`1/Z^2`，避免后面重复计算。

```cpp
for (int x = -half_patch_size; x <= half_patch_size; x++)
    for (int y = -half_patch_size; y <= half_patch_size; y++) {
```

第 249-251 行遍历 patch 内的每个像素。`half_patch_size = 1` 时，一共有 9 个误差项。

这里没有加大括号包住外层 `for` 的循环体，因为 C/C++ 中单条语句可以省略大括号。外层 `for` 的单条语句就是内层 `for`。从可读性角度，工程代码中通常建议加大括号。

```cpp
double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
               GetPixelValue(img2, u + x, v + y);
```

第 253-254 行计算光度误差：

```text
e = I_ref(p_ref + delta) - I_cur(p_cur + delta)
```

直接法的核心假设是光度一致性：同一个三维点在不同帧中的灰度应该相同。由于这里使用 patch，所以不只比较中心像素，还比较周围 `3x3` 小块。

```cpp
Matrix26d J_pixel_xi;
Eigen::Vector2d J_img_pixel;
```

第 255-256 行定义两个雅可比。

- `J_pixel_xi`：像素坐标 `(u, v)` 对 SE(3) 李代数扰动 `xi` 的导数，尺寸为 `2x6`。
- `J_img_pixel`：图像灰度对像素坐标 `(u, v)` 的导数，也就是图像梯度，尺寸为 `2x1`。

链式法则为：

```text
de/dxi = - dI_cur/duv * duv/dxi
```

负号来自误差定义 `e = I_ref - I_cur`。

```cpp
J_pixel_xi(0, 0) = fx * Z_inv;
J_pixel_xi(0, 1) = 0;
J_pixel_xi(0, 2) = -fx * X * Z2_inv;
J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
J_pixel_xi(0, 5) = -fx * Y * Z_inv;
```

第 258-263 行填写像素横坐标 `u` 对 6 维位姿扰动的导数。

```cpp
J_pixel_xi(1, 0) = 0;
J_pixel_xi(1, 1) = fy * Z_inv;
J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
J_pixel_xi(1, 5) = fy * X * Z_inv;
```

第 265-270 行填写像素纵坐标 `v` 对 6 维位姿扰动的导数。

整体矩阵为：

```text
duv/dxi =
[ fx/Z,     0, -fx*X/Z^2, -fx*X*Y/Z^2,  fx + fx*X^2/Z^2, -fx*Y/Z ]
[    0,  fy/Z, -fy*Y/Z^2, -fy - fy*Y^2/Z^2, fy*X*Y/Z^2,  fy*X/Z ]
```

这来自两部分链式求导：

1. 三维点在相机坐标系中受到 SE(3) 小扰动。
2. 三维点通过针孔相机模型投影到像素坐标。

```cpp
J_img_pixel = Eigen::Vector2d(
    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y))
);
```

第 272-275 行用中心差分计算当前图像在投影点附近的灰度梯度：

```text
dI/du = 0.5 * (I(u+1, v) - I(u-1, v))
dI/dv = 0.5 * (I(u, v+1) - I(u, v-1))
```

直接法依赖图像梯度。如果梯度很小，像素移动不会引起灰度变化，位姿就难以从光度误差中估计出来。

```cpp
Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();
```

第 277-278 行计算总雅可比。

`J_img_pixel.transpose()` 是 `1x2`，`J_pixel_xi` 是 `2x6`，乘积是 `1x6`，再转置成 `6x1`。

因为误差为：

```text
e = I_ref - I_cur
```

所以对位姿求导时有负号：

```text
J = de/dxi = - dI_cur/duv * duv/dxi
```

```cpp
hessian += J * J.transpose();
bias += -error * J;
cost_tmp += error * error;
```

第 280-282 行累加 Gauss-Newton 正规方程。

如果把误差线性化为：

```text
e(x + dx) ≈ e + J^T dx
```

最小化平方和可得到：

```text
H dx = b
H = sum(J J^T)
b = sum(-e J)
```

这和代码完全对应。

```cpp
if (cnt_good) {
    unique_lock<mutex> lck(hessian_mutex);
    H += hessian;
    b += bias;
    cost += cost_tmp / cnt_good;
}
```

第 286-292 行把局部累计结果合并到类成员中。

`unique_lock<mutex> lck(hessian_mutex);` 创建锁对象并加锁。离开作用域时，`lck` 析构并自动解锁。这是 C++ 的 RAII 思想：资源获取即初始化，资源释放交给对象析构函数自动完成。

C 中使用互斥锁通常需要手动调用 `lock` 和 `unlock`，如果中途 `return` 或异常，容易忘记释放。

## 九、多层金字塔直接法

```cpp
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21)
```

第 295-300 行实现多层直接法。它本身不改变优化目标，只是用不同分辨率的图像从粗到细优化同一个位姿。

```cpp
int pyramids = 4;
double pyramid_scale = 0.5;
double scales[] = {1.0, 0.5, 0.25, 0.125};
```

第 302-305 行设置金字塔参数。总共 4 层，每层缩小为上一层的一半。

第 0 层是原图，第 1 层是半尺寸，第 2 层是四分之一尺寸，第 3 层是八分之一尺寸。

```cpp
vector<cv::Mat> pyr1, pyr2;
for (int i = 0; i < pyramids; i++) {
    if (i == 0) {
        pyr1.push_back(img1);
        pyr2.push_back(img2);
    } else {
        cv::Mat img1_pyr, img2_pyr;
        cv::resize(...);
        cv::resize(...);
        pyr1.push_back(img1_pyr);
        pyr2.push_back(img2_pyr);
    }
}
```

第 307-322 行创建两张图各自的图像金字塔。

`pyr1` 保存参考图金字塔，`pyr2` 保存当前图金字塔。第 0 层直接使用原图，后续层由上一层缩放得到。

`cv::resize` 是 OpenCV 图像缩放函数。这里用默认插值方式。

SLAM 理论上，金字塔的作用是扩大收敛域。原图中的大位移，在低分辨率图像中会变成较小位移，Gauss-Newton 更容易收敛。

```cpp
double fxG = fx, fyG = fy, cxG = cx, cyG = cy;
```

第 324 行备份原始相机内参。

图像缩放后，像素坐标系也缩放，所以内参也要等比例缩放：

```text
fx_l = scale_l * fx
fy_l = scale_l * fy
cx_l = scale_l * cx
cy_l = scale_l * cy
```

```cpp
for (int level = pyramids - 1; level >= 0; level--) {
```

第 325 行从最粗层开始，逐层回到原图层。也就是先用 `0.125` 尺度估计粗略位姿，再用更高分辨率细化。

```cpp
VecVector2d px_ref_pyr;
for (auto &px: px_ref) {
    px_ref_pyr.push_back(scales[level] * px);
}
```

第 326-329 行把参考像素坐标缩放到当前金字塔层。

`for (auto &px: px_ref)` 是 C++11 范围 for 循环。C 中通常写成基于索引的循环。

`scales[level] * px` 能直接让标量乘 Eigen 向量，这是 Eigen 提供的运算符重载。

```cpp
fx = fxG * scales[level];
fy = fyG * scales[level];
cx = cxG * scales[level];
cy = cyG * scales[level];
DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
```

第 331-336 行缩放相机内参，并在当前层调用单层直接法。

`T21` 在各层之间连续传递。粗层优化后的位姿会作为细层初值。

一个工程细节：函数结束后没有把全局内参恢复为 `fxG, fyG, cxG, cyG`。由于最后一层 `level = 0` 的 scale 是 `1.0`，所以最终会恢复到原始数值。如果 `scales` 或循环顺序改了，就要注意这个隐含假设。

## 十、SLAM 公式与代码对应

这段代码优化的问题可以写成：

```text
min_T sum_i sum_delta || I_ref(p_i + delta) - I_cur(project(T * P_i) + delta) ||^2
```

其中：

- `p_i` 是参考帧像素。
- `depth_i` 是由双目视差得到的深度。
- `P_i = depth_i * K^{-1} p_i` 是参考帧三维点。
- `T` 是从参考帧到当前帧的 SE(3) 变换。
- `project(.)` 是针孔相机投影。
- `delta` 是 patch 内的偏移。

对应代码关系如下：

- `P_i = depth_i * K^{-1} p_i`：第 233-234 行。
- `P_cur = T21 * P_i`：第 235 行。
- `project(P_cur)`：第 239 行。
- 光度误差 `e`：第 253-254 行。
- 图像梯度 `dI/duv`：第 272-275 行。
- 投影对位姿的雅可比 `duv/dxi`：第 258-270 行。
- 总雅可比 `de/dxi`：第 278 行。
- Hessian 和右端项：第 280-281 行。
- 求解增量：第 177 行。
- SE(3) 左乘更新：第 178 行。

## 十一、直接法和特征法的区别

特征法通常流程是：

1. 检测特征点。
2. 计算描述子。
3. 匹配特征。
4. 用匹配点做 PnP、ICP 或对极几何估计位姿。

直接法的流程更接近这份代码：

1. 选择有深度的像素点。
2. 根据当前位姿把三维点投影到另一帧。
3. 比较投影位置的灰度差。
4. 直接优化相机位姿，使灰度误差最小。

直接法的优点：

- 不需要描述子匹配。
- 可以利用没有明显角点但有梯度的像素。
- 理论上能达到亚像素精度。

直接法的限制：

- 依赖光度一致性，曝光变化、反光、运动模糊会影响结果。
- 需要较好的初值，否则投影位置偏太远，线性化容易失败。
- 对相机标定、深度精度和图像梯度较敏感。

## 十二、从 C 到 C++：本文件出现的关键语法

这份代码集中展示了很多 C++ 相比 C 的核心能力：

- 命名空间：`std::`、`cv::`、`Eigen::`、`Sophus::` 用来避免名字冲突。
- 类：`JacobianAccumulator` 把数据和行为封装在一起。
- 构造函数：创建对象时自动初始化成员。
- 成员初始化列表：引用成员和复杂对象可以直接初始化。
- 访问控制：`public` 和 `private` 控制外部能访问什么。
- 引用：`const cv::Mat &` 避免复制，同时表达只读。
- 非 const 引用：`Sophus::SE3d &T21` 表达函数会修改传入对象。
- 模板：`vector<T>`、`Eigen::Matrix<double, 6, 6>` 都是模板实例。
- 运算符重载：`T21 * point_ref`、`scales[level] * px` 写起来像数学公式。
- RAII：`unique_lock<mutex>` 自动加锁和解锁。
- 自动类型推导：`auto t1`、`auto p_ref` 减少冗长类型书写。
- 范围 for：`for (auto &px: px_ref)` 直接遍历容器。
- 标准库函数对象：`std::bind` 和 `std::placeholders::_1` 把成员函数变成可并行调用的任务。

如果用 C 写同样的程序，通常会看到更多裸指针、手动内存管理、显式结构体初始化、函数指针和手写矩阵运算。C++ 的优势是可以把数学对象、图像对象、位姿对象和优化过程表达得更接近问题本身。

## 十三、代码中的几个实践注意点

1. `disparity == 0` 时深度会无效，真实工程中应该跳过这类点。
2. `GetPixelValue` 的边界截断最好改成 `img.cols - 2` 和 `img.rows - 2`，避免双线性插值访问右侧或下侧像素时越界。
3. `depth_ref` 在多个函数中按值传递，会复制数组；可以改成 `const vector<double> &depth_ref` 提升效率。
4. `boarder` 应该拼作 `border`，这是命名问题，不影响运行。
5. 代码随机选点，没有筛选梯度强的像素；如果选到大量低纹理点，Hessian 会更容易退化。
6. 该实现没有鲁棒核函数，动态物体、遮挡和光照变化都会直接影响平方误差。

## 十四、一句话总结

`direct_method.cpp` 用双目视差提供参考帧深度，用 SE(3) 表示相机运动，用光度误差构造优化目标，用图像梯度和投影雅可比通过链式法则得到位姿雅可比，最后通过 Gauss-Newton 在图像金字塔上从粗到细估计当前帧相对于参考帧的相机位姿。
