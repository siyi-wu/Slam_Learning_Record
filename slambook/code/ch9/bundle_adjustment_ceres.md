# `bundle_adjustment_ceres.cpp` 代码精读

本文围绕 `bundle_adjustment_ceres.cpp` 展开，目标有三条线：

1. 逐行解释代码在做什么。
2. 从 C 和 C++ 的区别出发理解代码中的 C++ 语法。
3. 从 SLAM 理论角度理解 Ceres 中的 Bundle Adjustment、残差块、自动求导、鲁棒核和 Schur 求解。

## 一、程序整体流程

这份代码用 Ceres Solver 求解 BAL 数据集上的 Bundle Adjustment 问题。它和 `bundle_adjustment_g2o.cpp` 优化的是同一个目标：同时调整相机参数和三维点，使所有观测的重投影误差尽可能小。

整体流程如下：

1. 从命令行读取 BAL 数据文件。
2. 用 `BALProblem` 读入相机、三维点和二维观测。
3. 对数据做归一化，并扰动初值。
4. 保存优化前点云 `initial.ply`。
5. 在 `SolveBA` 中创建 `ceres::Problem`。
6. 每条二维观测创建一个 residual block。
7. 每个 residual block 连接一个 9 维相机参数块和一个 3 维点参数块。
8. 残差由 `SnavelyReprojectionError` 定义，并由 Ceres 自动求导。
9. 使用 Huber loss 抑制外点。
10. 使用 `SPARSE_SCHUR` 求解 BA 的稀疏结构。
11. 保存优化后点云 `final.ply`。

和 g2o 版本相比，Ceres 版本不需要显式定义顶点和边类，而是围绕“参数块 + 残差块”建模。

## 二、头文件与命名空间

```cpp
#include <iostream>
#include <ceres/ceres.h>
#include "common.h"
#include "SnavelyReprojectionError.h"
```

第 1 行引入 C++ 输入输出流。

第 2 行引入 Ceres 主头文件，提供 `ceres::Problem`、`ceres::CostFunction`、`ceres::LossFunction`、`ceres::Solve` 等接口。

第 3 行引入 `BALProblem`，负责读写 BAL 数据。

第 4 行引入 `SnavelyReprojectionError`，它定义了带径向畸变的重投影残差。

```cpp
using namespace std;
```

第 6 行展开标准库命名空间。这样可以写 `cout` 和 `endl`，不用写 `std::cout` 和 `std::endl`。

C 语言没有命名空间。C++ 用命名空间管理大型库中的名字。

```cpp
void SolveBA(BALProblem &bal_problem);
```

第 8 行声明 `SolveBA`。因为 `main` 在函数实现之前调用它，所以需要提前声明。

## 三、主函数

```cpp
int main(int argc, char **argv) {
```

第 10 行是程序入口。`argc` 是命令行参数数量，`argv` 是命令行参数数组。

```cpp
if (argc != 2) {
    cout << "usage: bundle_adjustment_ceres bal_data.txt" << endl;
    return 1;
}
```

第 11-14 行检查用户是否传入一个 BAL 数据文件。如果参数数量不对，输出用法并返回 `1`。

```cpp
BALProblem bal_problem(argv[1]);
```

第 16 行构造 `BALProblem` 对象，从 `argv[1]` 指向的文件中读取数据。

C++ 构造函数把“创建对象”和“初始化对象”绑定在一起。C 中通常会先定义结构体变量，再调用初始化函数。

```cpp
bal_problem.Normalize();
bal_problem.Perturb(0.1, 0.5, 0.5);
```

第 17-18 行先归一化数据，再给初值加噪声。

`Normalize()` 让点云和相机中心处在更合适的尺度，改善数值稳定性。

`Perturb(0.1, 0.5, 0.5)` 分别扰动旋转、平移和三维点，用于模拟不完美初值。

```cpp
bal_problem.WriteToPLYFile("initial.ply");
SolveBA(bal_problem);
bal_problem.WriteToPLYFile("final.ply");
```

第 19-21 行保存优化前点云，运行 BA，再保存优化后点云。

由于 `SolveBA` 接收的是非 const 引用，优化会直接修改 `bal_problem` 内部的相机和点参数。

```cpp
return 0;
```

第 23 行表示程序正常结束。

## 四、`SolveBA`：创建 Ceres 优化问题

```cpp
void SolveBA(BALProblem &bal_problem) {
```

第 26 行定义 BA 求解函数。参数是引用，避免复制整个问题，并允许函数内部修改它。

C 中一般用指针传入可修改对象；C++ 引用语义更自然，也避免空指针含义。

```cpp
const int point_block_size = bal_problem.point_block_size();
const int camera_block_size = bal_problem.camera_block_size();
double *points = bal_problem.mutable_points();
double *cameras = bal_problem.mutable_cameras();
```

第 27-30 行取出参数块大小和可修改的底层数组。

BAL 默认相机参数块大小是 9：

```text
[0,1,2] angle-axis rotation
[3,4,5] translation
[6] focal
[7] k1
[8] k2
```

三维点参数块大小是 3：

```text
[X, Y, Z]
```

`double *` 是 C 风格指针。Ceres 为了高效和通用，参数块直接使用连续内存地址。

```cpp
const double *observations = bal_problem.observations();
ceres::Problem problem;
```

第 32-36 行取出观测数组，并创建 Ceres 优化问题对象。

每条观测有两个数，表示图像平面上的二维坐标。`ceres::Problem` 是 C++ 类，负责管理残差块、参数块和求解配置。

```cpp
for (int i = 0; i < bal_problem.num_observations(); ++i) {
```

第 38 行遍历所有观测。Ceres 中通常一条观测对应一个 residual block。

```cpp
ceres::CostFunction *cost_function;
```

第 39 行声明代价函数指针。每个代价函数会输入一个相机参数块和一个点参数块，输出 2 维残差。

```cpp
cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);
```

第 41-43 行创建当前观测对应的重投影误差函数。

`observations[2 * i + 0]` 是观测的 x 坐标，`observations[2 * i + 1]` 是 y 坐标。

`SnavelyReprojectionError::Create(...)` 返回一个 Ceres `CostFunction`，内部使用自动求导。

```cpp
ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
```

第 45-46 行创建 Huber 鲁棒核。

普通最小二乘对外点非常敏感。Huber loss 在误差小时像平方误差，在误差大时增长变慢，从而降低错误观测的影响。

```cpp
double *camera = cameras + camera_block_size * bal_problem.camera_index()[i];
double *point = points + point_block_size * bal_problem.point_index()[i];
```

第 48-52 行找到当前观测对应的相机参数块和三维点参数块。

`camera_index()[i]` 表示第 `i` 条观测来自哪个相机。

`point_index()[i]` 表示第 `i` 条观测看到了哪个三维点。

通过指针偏移找到参数块起始地址：

```text
第 k 个相机地址 = cameras + 9 * k
第 j 个点地址 = points + 3 * j
```

```cpp
problem.AddResidualBlock(cost_function, loss_function, camera, point);
```

第 54 行把残差块加入 Ceres 问题。

这一行表达了 BA 的核心图结构：

```text
residual_i = reprojection_error(camera_k, point_j)
```

Ceres 会自动把同一个 `camera` 地址识别为同一个参数块，把同一个 `point` 地址识别为同一个参数块。你不需要像 g2o 那样手动创建顶点 ID。

```cpp
std::cout << "bal problem file loaded..." << std::endl;
std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
          << bal_problem.num_points() << " points. " << std::endl;
std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;
```

第 57-61 行输出问题规模，包括相机数、三维点数和观测数。

这里显式写 `std::cout`，虽然前面已经 `using namespace std;`。两种写法都可以。

```cpp
std::cout << "Solving ceres BA ... " << endl;
ceres::Solver::Options options;
```

第 63-64 行输出提示，并创建 Ceres 求解器配置对象。

```cpp
options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
```

第 65 行选择 `SPARSE_SCHUR` 线性求解器。

BA 的 Hessian 有典型的相机-点块稀疏结构。Schur complement 可以先消去三维点变量，得到只关于相机变量的系统，适合大规模 BA。

```cpp
options.minimizer_progress_to_stdout = true;
```

第 66 行让 Ceres 把每轮优化的进度打印到终端。

```cpp
ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);
std::cout << summary.FullReport() << "\n";
```

第 67-69 行创建求解结果摘要，调用 `ceres::Solve` 开始优化，最后输出完整报告。

Ceres 会根据 residual block 自动构建雅可比、正规方程，选择线性求解器并更新参数块。因为参数块直接指向 `BALProblem` 内部数组，所以优化结束后 `bal_problem` 已经包含最新结果。

## 五、残差模型 `SnavelyReprojectionError`

虽然主文件很短，真正的误差定义在 `SnavelyReprojectionError.h` 中。

```cpp
class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observation_x, double observation_y)
        : observed_x(observation_x), observed_y(observation_y) {}
```

这个类保存一条观测的二维坐标。冒号后面是成员初始化列表，比在函数体内赋值更直接。

```cpp
template<typename T>
bool operator()(const T *const camera,
                const T *const point,
                T *residuals) const {
    T predictions[2];
    CamProjectionWithDistortion(camera, point, predictions);
    residuals[0] = predictions[0] - T(observed_x);
    residuals[1] = predictions[1] - T(observed_y);
    return true;
}
```

这是 Ceres 自动求导的核心。

`operator()` 让对象可以像函数一样被调用。C 中没有运算符重载，通常要写普通函数指针。

`template<typename T>` 很关键：自动求导时，Ceres 会把 `T` 换成一种 Jet 类型，它携带数值和导数。只要误差函数用模板类型 `T` 正常计算，Ceres 就能自动得到雅可比。

残差定义为：

```text
residual = predicted_observation - measured_observation
```

```cpp
static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions)
```

这个静态函数实现带径向畸变的投影。

```cpp
AngleAxisRotatePoint(camera, point, p);
p[0] += camera[3];
p[1] += camera[4];
p[2] += camera[5];
```

先用相机的旋转向量旋转三维点，再加平移：

```text
P_c = R * P_w + t
```

```cpp
T xp = -p[0] / p[2];
T yp = -p[1] / p[2];
```

得到归一化平面坐标。这里的负号来自 BAL/Snavely 数据集的相机坐标约定。

```cpp
const T &l1 = camera[7];
const T &l2 = camera[8];
T r2 = xp * xp + yp * yp;
T distortion = T(1.0) + r2 * (l1 + l2 * r2);
```

计算径向畸变：

```text
d = 1 + k1*r^2 + k2*r^4
```

```cpp
const T &focal = camera[6];
predictions[0] = focal * distortion * xp;
predictions[1] = focal * distortion * yp;
```

乘以焦距，得到最终预测观测。

```cpp
static ceres::CostFunction *Create(const double observed_x, const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
        new SnavelyReprojectionError(observed_x, observed_y)));
}
```

`Create` 创建 Ceres 自动求导代价函数。

模板参数含义为：

- `SnavelyReprojectionError`：误差函数类。
- `2`：残差维度是 2。
- `9`：第一个参数块，也就是相机，维度是 9。
- `3`：第二个参数块，也就是三维点，维度是 3。

这正好对应 BA 中每条观测连接一个相机和一个三维点。

## 六、SLAM 理论对应

BA 优化目标是：

```text
min_{cameras, points} sum_ij rho( || project(camera_i, point_j) - observation_ij ||^2 )
```

在 Ceres 代码中：

- `camera` 参数块表示一个相机的旋转、平移、焦距和畸变。
- `point` 参数块表示一个三维路标点。
- `SnavelyReprojectionError` 表示单条观测的重投影误差。
- `problem.AddResidualBlock(...)` 把一条观测加入总优化问题。
- `HuberLoss` 是鲁棒核函数。
- `SPARSE_SCHUR` 利用 BA 的相机-点稀疏结构。

每个 residual block 只依赖一个相机和一个点，所以整个雅可比非常稀疏。Ceres 根据这种结构构建线性系统，并用 Schur complement 高效求解。

## 七、Ceres 和 g2o 建模方式对比

g2o 更像“手动搭图”：

```text
定义相机顶点
定义点顶点
定义投影边
把顶点和边加入优化器
```

Ceres 更像“声明残差函数”：

```text
定义一个残差计算 functor
对每条观测 AddResidualBlock
让 Ceres 管理参数块和求导
```

所以 Ceres 版本主文件更短。它把大部分复杂性藏在 `SnavelyReprojectionError` 和 Ceres 自动求导机制里。

## 八、从 C 到 C++：本文件的关键语法

- 类对象：`ceres::Problem problem`、`ceres::Solver::Options options` 都是带行为的对象。
- 引用参数：`BALProblem &bal_problem` 避免复制，并允许修改原对象。
- 指针参数块：`double *camera`、`double *point` 直接指向待优化内存。
- 构造函数：`BALProblem bal_problem(argv[1])` 读文件并初始化对象。
- 成员函数：`bal_problem.Normalize()`、`problem.AddResidualBlock(...)` 把数据和行为封装在类中。
- 命名空间：`ceres::`、`std::` 避免名字冲突。
- `new`：动态创建 `HuberLoss` 和自动求导代价函数。
- 模板：`AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>` 在编译期指定残差和参数块维度。
- 函数对象：`operator()` 让误差类对象像函数一样被 Ceres 调用。
- `const`：`const double *observations` 表示只读观测数组。

如果用 C 写类似程序，通常要手动设计结构体、函数指针、残差回调、雅可比回调和内存管理。Ceres 的 C++ 接口让“误差怎么计算”和“优化器怎么调用它”之间的连接更自然。

## 九、实践注意点

1. Ceres 版本没有手写雅可比，而是通过 `AutoDiffCostFunction` 自动求导，代码短且不容易写错。
2. 参数块直接指向 `BALProblem` 的内部数组，所以优化结束后不需要再显式写回。
3. Huber loss 的尺度设为 `1.0`，不同数据尺度下可能需要调整。
4. `SPARSE_SCHUR` 是 BA 的常用选择，比普通稠密求解器更适合大规模相机-点问题。
5. 旋转使用 angle-axis 表示，不是四元数；这个版本的相机参数块维度是 9。

## 十、一句话总结

`bundle_adjustment_ceres.cpp` 用 Ceres 把 BA 表达成一组残差块：每条二维观测对应一个重投影残差，残差依赖一个相机参数块和一个三维点参数块，Ceres 通过自动求导、Huber 鲁棒核和 `SPARSE_SCHUR` 求解器完成相机与地图点的联合优化。
