# `bundle_adjustment_g2o.cpp` 代码精读

本文围绕 `bundle_adjustment_g2o.cpp` 展开，目标有三条线：

1. 逐行解释代码在做什么。
2. 从 C 和 C++ 的区别出发理解代码中的 C++ 语法。
3. 从 SLAM 理论角度理解 Bundle Adjustment、重投影误差、图优化和 Schur 结构。

## 一、程序整体流程

这份代码用 g2o 求解 BAL 数据集上的 Bundle Adjustment 问题。BAL 问题中有若干相机、若干三维点和若干二维观测。每条观测表示“某个相机看到了某个d三维点，并在图像平面上观测到二维坐标”。

整体流程如下：

1. 从命令行读取 BAL 数据文件路径。
2. 用 `BALProblem` 读入相机参数、三维点和二维观测。
3. 对数据做归一化，并人为扰动初值。
4. 保存优化前点云 `initial.ply`。
5. 调用 `SolveBA` 构建 g2o 图优化问题。
6. 每个相机是一个 9 维顶点：旋转、平移、焦距、两个径向畸变参数。
7. 每个地图点是一个 3 维顶点。
8. 每条观测是一条二元边，连接一个相机顶点和一个点顶点。
9. 边的误差是投影值减去观测值。
10. 使用 Levenberg-Marquardt 和 CSparse 求解。
11. 把优化后的顶点写回 `BALProblem`，保存 `final.ply`。

SLAM 后端中，BA 的目标是同时优化相机位姿和地图点，使所有观测的重投影误差平方和最小。

## 二、头文件与命名空间

```cpp
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>
```

第 1-6 行引入 g2o 的核心组件。

- `base_vertex.h`：自定义优化顶点的基类。
- `base_binary_edge.h`：自定义二元边的基类。
- `block_solver.h`：块结构求解器，适合 BA 中相机块和路标点块。
- `optimization_algorithm_levenberg.h`：Levenberg-Marquardt 优化算法。
- `linear_solver_csparse.h`：CSparse 稀疏线性求解器。
- `robust_kernel_impl.h`：Huber 等鲁棒核函数。

第 7 行引入 C++ 标准输入输出流。C 中常用 `printf` 和 `fprintf`，C++ 中更常用 `std::cout`。

```cpp
#include "common.h"
#include "sophus/se3.hpp"
```

第 9 行引入本章的 `BALProblem`，负责读写 BAL 数据。

第 10 行引入 Sophus。代码主要用 `SO3d` 表示旋转，用李代数指数映射更新旋转。

```cpp
using namespace Sophus;
using namespace Eigen;
using namespace std;
```

第 12-14 行展开命名空间。这样后面可以直接写 `SO3d`、`Vector3d`、`cout`，不用写完整前缀。

C 语言没有命名空间。C++ 通过命名空间避免不同库之间的名字冲突。

## 三、相机参数结构 `PoseAndIntrinsics`

```cpp
struct PoseAndIntrinsics {
```

第 16-17 行定义结构体，表示一个相机的外参和内参。

C 也有 `struct`，但 C++ 的 `struct` 可以有构造函数、成员函数、默认成员初始化，几乎和 `class` 一样；主要区别是 `struct` 默认成员是 `public`，`class` 默认成员是 `private`。

```cpp
PoseAndIntrinsics() {}
```

第 18 行是默认构造函数。创建空对象时调用。

```cpp
explicit PoseAndIntrinsics(double *data_addr) {
    rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2]));
    translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);
    focal = data_addr[6];
    k1 = data_addr[7];
    k2 = data_addr[8];
}
```

第 20-27 行从一段连续内存中读取相机参数。

BAL 数据中每个相机有 9 个参数：

```text
[0,1,2] angle-axis rotation
[3,4,5] translation
[6] focal
[7] k1
[8] k2
```

`SO3d::exp(...)` 把 3 维旋转向量映射成 SO(3) 旋转。SLAM 中常用李代数表示旋转扰动，但真正的旋转属于李群 SO(3)。

`explicit` 防止编译器进行隐式类型转换。没有它时，某些需要 `PoseAndIntrinsics` 的地方可能自动把 `double *` 转成对象，容易产生不清晰的代码。

```cpp
void set_to(double *data_addr) {
    auto r = rotation.log();
    for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
    for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
    data_addr[6] = focal;
    data_addr[7] = k1;
    data_addr[8] = k2;
}
```

第 29-37 行把优化后的结构体写回连续数组。

`rotation.log()` 把 SO(3) 旋转转回 3 维李代数向量。`auto` 让编译器自动推导变量类型。C 语言没有这种现代类型推导。

```cpp
SO3d rotation;
Vector3d translation = Vector3d::Zero();
double focal = 0;
double k1 = 0, k2 = 0;
```

第 39-42 行是成员变量。

- `rotation`：相机旋转。
- `translation`：相机平移。
- `focal`：焦距。
- `k1, k2`：二阶和四阶径向畸变系数。

`Vector3d::Zero()` 和 `double focal = 0` 是 C++ 成员默认初始化。C 结构体本身不能在定义里这样初始化成员。

## 四、相机顶点 `VertexPoseAndIntrinsics`

```cpp
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> {
```

第 45-46 行定义相机顶点类，继承自 g2o 的 `BaseVertex`。

`BaseVertex<9, PoseAndIntrinsics>` 的含义是：

- 优化变量维度是 9。
- 估计值类型是 `PoseAndIntrinsics`。

C 语言没有类继承和模板。这里的 g2o 通过 C++ 模板让顶点维度和估计类型在编译期确定。

```cpp
EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
```

第 48 行是 Eigen 的内存对齐宏。类中含有 Eigen 固定大小向量或 Sophus 对象时，动态分配对象需要保证内存对齐。

```cpp
VertexPoseAndIntrinsics() {}
```

第 50 行是默认构造函数。

```cpp
virtual void setToOriginImpl() override {
    _estimate = PoseAndIntrinsics();
}
```

第 52-54 行重写 g2o 顶点的“设为原点”函数。`_estimate` 是 g2o 基类保存的当前估计。

`virtual` 表示虚函数，允许通过基类指针调用子类实现。`override` 是 C++11 语法，告诉编译器这里必须覆盖父类虚函数；如果函数签名写错，编译器会报错。C 中没有虚函数和多态。

```cpp
virtual void oplusImpl(const double *update) override {
    _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
    _estimate.translation += Vector3d(update[3], update[4], update[5]);
    _estimate.focal += update[6];
    _estimate.k1 += update[7];
    _estimate.k2 += update[8];
}
```

第 56-62 行定义顶点如何应用优化增量。

9 维增量含义为：

```text
[0,1,2] rotation update
[3,4,5] translation update
[6] focal update
[7] k1 update
[8] k2 update
```

旋转不能简单相加，因为旋转属于 SO(3) 流形。代码使用：

```text
R <- exp(delta_so3) * R
```

这是左乘扰动。平移、焦距和畸变参数属于欧氏空间，所以直接相加。

```cpp
Vector2d project(const Vector3d &point) {
    Vector3d pc = _estimate.rotation * point + _estimate.translation;
    pc = -pc / pc[2];
    double r2 = pc.squaredNorm();
    double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
    return Vector2d(_estimate.focal * distortion * pc[0],
                    _estimate.focal * distortion * pc[1]);
}
```

第 64-72 行根据当前相机参数投影一个三维点。

首先把世界点变到相机坐标：

```text
P_c = R * P_w + t
```

然后归一化成像平面坐标：

```text
x = -X / Z
y = -Y / Z
```

这里的负号来自 BAL/Snavely 数据集采用的相机坐标约定。

径向畸变模型为：

```text
r^2 = x^2 + y^2
d = 1 + r^2 * (k1 + k2 * r^2)
u = f * d * x
v = f * d * y
```

返回值是预测的二维观测。

```cpp
virtual bool read(istream &in) {}
virtual bool write(ostream &out) const {}
```

第 74-76 行是 g2o 顶点读写接口。这里没有实现实际读写，因为本程序手动构建图，而不是从 g2o 文件读取。

严格说，这两个函数声明返回 `bool`，但函数体没有 `return`。部分编译器会警告。更严谨写法是 `return false;`。

## 五、三维点顶点 `VertexPoint`

```cpp
class VertexPoint : public g2o::BaseVertex<3, Vector3d> {
```

第 79 行定义路标点顶点。`BaseVertex<3, Vector3d>` 表示优化变量维度是 3，估计值类型是 Eigen 的三维向量。

```cpp
EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
VertexPoint() {}
```

第 81-83 行同样处理 Eigen 内存对齐，并定义默认构造函数。

```cpp
virtual void setToOriginImpl() override {
    _estimate = Vector3d(0, 0, 0);
}
```

第 85-87 行把点设为原点。

```cpp
virtual void oplusImpl(const double *update) override {
    _estimate += Vector3d(update[0], update[1], update[2]);
}
```

第 89-91 行应用三维点增量。三维点在欧氏空间中，可以直接加法更新。

```cpp
virtual bool read(istream &in) {}
virtual bool write(ostream &out) const {}
```

第 93-95 行同样是 g2o 的读写接口占位。

## 六、投影边 `EdgeProjection`

```cpp
class EdgeProjection :
    public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint> {
```

第 98-99 行定义二元边。模板参数含义是：

- 残差维度为 2。
- 观测值类型是 `Vector2d`。
- 第一个顶点类型是 `VertexPoseAndIntrinsics`。
- 第二个顶点类型是 `VertexPoint`。

在 BA 中，一条观测连接一个相机和一个三维点，所以它正好是一条二元边。

```cpp
virtual void computeError() override {
    auto v0 = (VertexPoseAndIntrinsics *) _vertices[0];
    auto v1 = (VertexPoint *) _vertices[1];
    auto proj = v0->project(v1->estimate());
    _error = proj - _measurement;
}
```

第 103-108 行计算边的误差。

`_vertices` 是 g2o 基类保存的顶点指针数组。这里把第 0 个顶点转成相机顶点，把第 1 个顶点转成地图点顶点。

`v1->estimate()` 取出三维点坐标，`v0->project(...)` 用相机模型投影，得到预测观测。

误差定义为：

```text
e = projected_point - observed_point
```

这就是 BA 的重投影误差。优化目标是所有边的误差平方和。

`auto` 简化了类型书写。强制类型转换 `(VertexPoseAndIntrinsics *)` 是 C 风格转换，在现代 C++ 中也可以写成 `static_cast<VertexPoseAndIntrinsics *>(...)`，语义更清楚。

```cpp
// use numeric derivatives
virtual bool read(istream &in) {}
virtual bool write(ostream &out) const {}
```

第 110-113 行说明这里没有手写雅可比，g2o 会使用数值求导。手写雅可比通常更快，但更容易写错；数值求导适合教学或原型验证。

## 七、主函数

```cpp
void SolveBA(BALProblem &bal_problem);
```

第 117 行声明函数。因为 `main` 会先调用 `SolveBA`，而函数实现放在后面，所以需要提前声明。

```cpp
int main(int argc, char **argv) {
```

第 119 行是程序入口。`argc` 是命令行参数数量，`argv` 是参数字符串数组。

```cpp
if (argc != 2) {
    cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;
    return 1;
}
```

第 121-124 行检查用户是否提供了 BAL 数据文件路径。如果没有，输出用法并返回错误码。

```cpp
BALProblem bal_problem(argv[1]);
```

第 126 行从文件构造 `BALProblem` 对象。构造函数会读入相机、点和观测。

C 中通常会写一个初始化函数，例如 `BALProblemInit(&problem, argv[1])`；C++ 构造函数让对象创建和初始化绑定在一起。

```cpp
bal_problem.Normalize();
bal_problem.Perturb(0.1, 0.5, 0.5);
```

第 127-128 行先归一化数据，再人为扰动初值。

`Normalize()` 把点云和相机中心移动、缩放到合适尺度，有助于数值稳定。

`Perturb(0.1, 0.5, 0.5)` 给旋转、平移和三维点增加噪声，用于模拟初值不准的情况。

```cpp
bal_problem.WriteToPLYFile("initial.ply");
SolveBA(bal_problem);
bal_problem.WriteToPLYFile("final.ply");
```

第 129-131 行保存优化前点云，运行 BA，再保存优化后点云。可以用 MeshLab 或 CloudCompare 查看相机和点云变化。

```cpp
return 0;
```

第 133 行程序正常结束。

## 八、构建并求解 g2o BA 问题

```cpp
void SolveBA(BALProblem &bal_problem) {
```

第 136 行定义求解函数。参数是非 const 引用，因为函数会把优化后的结果写回 `bal_problem`。

```cpp
const int point_block_size = bal_problem.point_block_size();
const int camera_block_size = bal_problem.camera_block_size();
double *points = bal_problem.mutable_points();
double *cameras = bal_problem.mutable_cameras();
```

第 137-140 行取出数据布局。

BAL 中一个三维点有 3 个参数，一个相机默认有 9 个参数。`mutable_points()` 和 `mutable_cameras()` 返回可修改的原始数组指针。

这几行保留了 C 风格数组操作：`double *` 指向连续内存。C++ 在上层用类管理数据，但底层仍然可以高效地使用指针。

```cpp
typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;
typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
```

第 142-144 行定义求解器类型。

`BlockSolverTraits<9, 3>` 表示 BA 的块结构：

- 相机块维度是 9。
- 路标点块维度是 3。

这种块结构能利用 BA 的稀疏性，尤其是 Schur 消元结构。

```cpp
auto solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
```

第 145-147 行创建 Levenberg-Marquardt 优化算法。

内部嵌套关系是：

```text
OptimizationAlgorithmLevenberg
    -> BlockSolver
        -> LinearSolverCSparse
```

`new` 动态分配对象。C 中类似 `malloc`，但 C++ 的 `new` 会调用构造函数。`g2o::make_unique` 创建 `unique_ptr`，用于表达独占所有权。

```cpp
g2o::SparseOptimizer optimizer;
optimizer.setAlgorithm(solver);
optimizer.setVerbose(true);
```

第 148-150 行创建稀疏优化器，设置求解算法，并打开日志输出。

```cpp
const double *observations = bal_problem.observations();
```

第 152-153 行取出观测数组。每个观测有两个数：二维坐标 `x, y`。

```cpp
vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
vector<VertexPoint *> vertex_points;
```

第 154-156 行创建两个数组，保存后面生成的顶点指针。

`vector` 是 C++ 动态数组，会自动扩容。C 中需要手动维护容量和内存。

```cpp
for (int i = 0; i < bal_problem.num_cameras(); ++i) {
    VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();
    double *camera = cameras + camera_block_size * i;
    v->setId(i);
    v->setEstimate(PoseAndIntrinsics(camera));
    optimizer.addVertex(v);
    vertex_pose_intrinsics.push_back(v);
}
```

第 157-164 行为每个相机创建一个顶点。

`cameras + camera_block_size * i` 通过指针偏移找到第 `i` 个相机的参数起始地址。

`setId(i)` 设置 g2o 顶点 ID。相机顶点 ID 从 `0` 开始。

`setEstimate(PoseAndIntrinsics(camera))` 用 BAL 数组初始化顶点估计。

`optimizer.addVertex(v)` 把顶点加入图优化器。

`vertex_pose_intrinsics.push_back(v)` 保存指针，方便建边时按相机索引找到顶点。

```cpp
for (int i = 0; i < bal_problem.num_points(); ++i) {
    VertexPoint *v = new VertexPoint();
    double *point = points + point_block_size * i;
    v->setId(i + bal_problem.num_cameras());
    v->setEstimate(Vector3d(point[0], point[1], point[2]));
    v->setMarginalized(true);
    optimizer.addVertex(v);
    vertex_points.push_back(v);
}
```

第 165-174 行为每个三维点创建顶点。

点顶点 ID 从 `num_cameras` 开始，避免和相机顶点 ID 冲突。

`setMarginalized(true)` 告诉 g2o 这些点在 BA 中可以被边缘化。BA 常用 Schur complement 先消去三维点，只求相机变量的较小系统，再回代点变量。这能显著提升求解效率。

```cpp
for (int i = 0; i < bal_problem.num_observations(); ++i) {
    EdgeProjection *edge = new EdgeProjection;
    edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);
    edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);
    edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1]));
    edge->setInformation(Matrix2d::Identity());
    edge->setRobustKernel(new g2o::RobustKernelHuber());
    optimizer.addEdge(edge);
}
```

第 176-185 行为每条观测创建一条投影边。

`camera_index()[i]` 表示第 `i` 条观测来自哪个相机。

`point_index()[i]` 表示第 `i` 条观测看到哪个三维点。

`setMeasurement(...)` 设置观测到的二维像素坐标。

`setInformation(Matrix2d::Identity())` 设置信息矩阵。信息矩阵是协方差矩阵的逆。这里用单位阵，表示两个方向权重相同、噪声方差为 1。

`setRobustKernel(new g2o::RobustKernelHuber())` 使用 Huber 鲁棒核，降低离群观测对优化的影响。

```cpp
optimizer.initializeOptimization();
optimizer.optimize(40);
```

第 187-188 行初始化优化，并迭代最多 40 次。

LM 会反复线性化重投影误差，构造正规方程，求解增量，更新顶点估计，直到收敛或达到迭代次数。

```cpp
for (int i = 0; i < bal_problem.num_cameras(); ++i) {
    double *camera = cameras + camera_block_size * i;
    auto vertex = vertex_pose_intrinsics[i];
    auto estimate = vertex->estimate();
    estimate.set_to(camera);
}
```

第 190-196 行把优化后的相机顶点写回 BAL 参数数组。

`vertex->estimate()` 返回 `PoseAndIntrinsics`，再调用 `set_to(camera)` 写回连续内存。

```cpp
for (int i = 0; i < bal_problem.num_points(); ++i) {
    double *point = points + point_block_size * i;
    auto vertex = vertex_points[i];
    for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
}
```

第 197-201 行把优化后的三维点写回 BAL 参数数组。

## 九、SLAM 理论对应

BA 优化目标可以写成：

```text
min_{cameras, points} sum_ij rho( || project(camera_i, point_j) - observation_ij ||^2 )
```

其中：

- `camera_i` 是第 `i` 个相机的旋转、平移、焦距和畸变参数。
- `point_j` 是第 `j` 个三维点。
- `observation_ij` 是第 `i` 个相机对第 `j` 个点的二维观测。
- `project` 是带径向畸变的相机投影模型。
- `rho` 是 Huber 鲁棒核。

在 g2o 图优化中：

- 相机参数是 `VertexPoseAndIntrinsics`。
- 三维点是 `VertexPoint`。
- 观测是 `EdgeProjection`。
- 重投影误差在 `computeError()` 中计算。

BA 的稀疏性来自这样一个事实：每条边只连接一个相机和一个点。因此 Hessian 具有块稀疏结构。Schur complement 可以先消去点变量，得到只关于相机变量的系统，求完相机后再恢复点。

## 十、从 C 到 C++：本文件的关键语法

- 类和继承：`VertexPoseAndIntrinsics` 继承 g2o 顶点基类，`EdgeProjection` 继承 g2o 边基类。
- 模板：`BaseVertex<9, PoseAndIntrinsics>`、`BlockSolverTraits<9, 3>` 在编译期指定维度和类型。
- 虚函数和多态：g2o 通过 `setToOriginImpl`、`oplusImpl`、`computeError` 调用用户自定义行为。
- `override`：让编译器检查函数是否真的覆盖父类虚函数。
- 构造函数：`PoseAndIntrinsics(double *data_addr)` 把原始数组封装成结构化对象。
- `explicit`：避免不清晰的隐式转换。
- 引用：`const Vector3d &point` 避免复制三维点。
- 运算符重载：`_estimate.rotation * point` 看起来像数学公式。
- `auto`：简化复杂类型书写。
- `vector`：替代 C 中手动管理的动态数组。
- `new` 和对象构造：动态创建顶点和边，同时调用构造函数。

## 十一、实践注意点

1. `read()` 和 `write()` 返回 `bool`，但函数体没有 `return`，严格编译设置下会有警告。
2. 本实现没有手写雅可比，g2o 会用数值导数，教学上简单，但速度和精度通常不如解析雅可比。
3. 顶点和边用 `new` 创建后交给 g2o 管理；阅读这类代码时要注意对象所有权。
4. 相机模型使用 BAL/Snavely 的负号约定：`x = -X/Z`，不要和一般针孔模型混淆。
5. `setMarginalized(true)` 对 BA 很关键，它让求解器能利用点变量可被 Schur 消元的结构。

## 十二、一句话总结

`bundle_adjustment_g2o.cpp` 把 BA 建模成一个 g2o 图优化问题：相机和三维点是顶点，二维观测是边，边误差是带畸变相机模型下的重投影误差，最后用 LM、稀疏线性求解器和 Schur 结构同时优化相机参数与地图点。
