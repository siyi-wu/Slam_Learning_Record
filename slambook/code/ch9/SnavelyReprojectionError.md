# `SnavelyReprojectionError.h` 代码精读

本文围绕 `SnavelyReprojectionError.h` 展开，目标有三条线：

1. 逐行解释代码在做什么。
2. 从 C 和 C++ 的区别出发理解代码中的 C++ 语法。
3. 从 SLAM 理论角度理解重投影误差、相机模型、径向畸变和 Ceres 自动求导。

这份头文件是 `bundle_adjustment_ceres.cpp` 的核心：它定义了一条观测边的残差，也就是 Bundle Adjustment 中最重要的误差项。

## 一、它在 BA 中的位置

在 `bundle_adjustment_ceres.cpp` 里，每一条二维观测都会创建一个残差块：

```cpp
cost_function = SnavelyReprojectionError::Create(
    observations[2 * i + 0],
    observations[2 * i + 1]
);
```

然后加入 Ceres 问题：

```cpp
problem.AddResidualBlock(cost_function, loss_function, camera, point);
```

所以 `SnavelyReprojectionError` 表达的是：

```text
给定一个相机参数 camera 和一个三维点 point，
计算它投影到图像平面上的预测位置，
再和真实观测位置相减，
得到 2 维重投影误差。
```

数学形式是：

```math
e_{ij} =
\pi(c_i, X_j) - u_{ij}
```

其中：

- `c_i` 是第 `i` 个相机参数。
- `X_j` 是第 `j` 个三维点。
- `u_ij` 是数据文件中给出的二维观测。
- `π(c_i, X_j)` 是相机投影函数。
- `e_ij` 是二维残差。

整体 BA 优化目标是：

```math
\min_{\{c_i\}, \{X_j\}}
\frac{1}{2}
\sum_{(i,j)\in\mathcal{O}}
\rho\left(\|e_{ij}\|^2\right)
```

其中 `ρ` 是 Huber 鲁棒核，由 `bundle_adjustment_ceres.cpp` 里的 `ceres::HuberLoss(1.0)` 提供。

## 二、头文件保护

```cpp
#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H
```

这两行是头文件保护，防止同一个头文件被重复包含。

如果没有头文件保护，多个 `.cpp` 或 `.h` 间接包含同一个头文件时，编译器可能会看到重复的类定义，从而报错。

C 和 C++ 都常用这种写法。现代 C++ 里也常见：

```cpp
#pragma once
```

但这份代码使用的是传统、兼容性更强的宏保护方式。

## 三、包含头文件

```cpp
#include <iostream>
```

引入 C++ 标准输入输出流。这个文件里实际上没有直接使用 `std::cout` 或 `std::cerr`，所以这一行不是必须的。

C 语言中常用：

```c
#include <stdio.h>
```

C++ 更推荐使用：

```cpp
#include <iostream>
```

区别是：

- C 的 `stdio.h` 使用 `printf`、`scanf` 这类函数。
- C++ 的 `iostream` 使用 `std::cout`、`std::cin` 这类流对象。

```cpp
#include "ceres/ceres.h"
```

引入 Ceres Solver 的主头文件。

这份代码会用到：

```cpp
ceres::CostFunction
ceres::AutoDiffCostFunction
```

`ceres::` 是命名空间。C 语言没有命名空间，所有函数名基本在全局空间中。C++ 用命名空间避免大型库之间发生名字冲突。

```cpp
#include "rotation.h"
```

引入角轴旋转相关函数。这里用到的是：

```cpp
AngleAxisRotatePoint(camera, point, p);
```

它的作用是把三维点 `point` 按照相机参数里的角轴旋转量进行旋转。

从 SLAM 角度看，这一行对应：

```math
P' = R P
```

其中 `R` 由角轴向量转换而来。

## 四、类定义

```cpp
class SnavelyReprojectionError {
```

定义一个 C++ 类。

在 C 语言中，通常会用 `struct` 加函数来组织数据和行为，例如：

```c
typedef struct {
    double observed_x;
    double observed_y;
} SnavelyReprojectionError;
```

然后再写普通函数处理它。

C++ 的 `class` 可以把数据和函数放在一起，这更适合表达“一个残差模型对象”。

从 Ceres 角度看，这个类是一个 functor，也就是“可以像函数一样被调用的对象”。

## 五、public 访问区

```cpp
public:
```

表示下面的成员可以从类外访问。

C++ 类默认成员是 `private`，而 C++ `struct` 默认成员是 `public`。

这和 C 不同。C 的 `struct` 没有访问控制，所有字段都可以直接访问。

这里把构造函数、`operator()` 和 `Create()` 放在 `public` 下，是因为 Ceres 和外部代码需要调用它们。

## 六、构造函数

```cpp
SnavelyReprojectionError(double observation_x, double observation_y)
    : observed_x(observation_x),
      observed_y(observation_y) {}
```

这是构造函数。它和类名相同，没有返回值。

它的作用是保存当前这条观测的真实二维坐标：

```text
observed_x = observation_x
observed_y = observation_y
```

冒号后面的部分叫成员初始化列表：

```cpp
: observed_x(observation_x),
  observed_y(observation_y)
```

它等价于在构造函数体里写：

```cpp
observed_x = observation_x;
observed_y = observation_y;
```

但 C++ 更推荐成员初始化列表，因为它是在对象构造时直接初始化成员，语义更清楚，也可能更高效。

从 SLAM 角度看，`observed_x` 和 `observed_y` 就是固定观测值：

```math
u_{ij} =
\begin{bmatrix}
u_{ij}^x \\
u_{ij}^y
\end{bmatrix}
```

优化过程中它们不会被改变。被改变的是相机参数和三维点。

## 七、模板函数 `operator()`

```cpp
template<typename T>
```

这表示下面的函数是一个模板函数。

`T` 可以是 `double`，也可以是 Ceres 自动求导使用的 Jet 类型。

这是理解 Ceres 自动求导的关键。

如果 `T = double`，函数只计算残差数值。

如果 `T = ceres::Jet<double, N>`，函数不仅计算残差，还能携带导数信息，从而自动得到雅可比。

C 语言没有模板。C 里如果想让同一段逻辑支持不同类型，通常要写多份函数，或者用宏、`void*`。C++ 模板可以在编译期生成类型安全的代码。

```cpp
bool operator()(const T *const camera,
                const T *const point,
                T *residuals) const {
```

这是函数调用运算符重载。

定义了 `operator()` 后，对象就能像函数一样被调用：

```cpp
SnavelyReprojectionError error(x, y);
error(camera, point, residuals);
```

Ceres 的 `AutoDiffCostFunction` 正是通过这个接口调用残差函数。

参数含义：

```cpp
const T *const camera
```

`camera` 是相机参数数组指针。两个 `const` 含义不同：

- `const T *` 表示指针指向的数据不能被修改。
- `*const camera` 表示指针本身不能改指向别处。

所以 `const T *const camera` 表示：

```text
不能修改 camera 指向的相机参数；
也不能让 camera 指向别的地址。
```

```cpp
const T *const point
```

同理，`point` 是三维点参数数组，函数内部只读。

```cpp
T *residuals
```

`residuals` 是输出数组。这个函数要把 2 维重投影误差写进去，所以不能是 `const`。

最后的：

```cpp
) const
```

表示这个成员函数不会修改当前对象的成员变量，也就是不会修改 `observed_x` 和 `observed_y`。

从 C++ 习惯看，凡是只读成员函数，都应该尽量加 `const`。

## 八、计算预测投影

```cpp
// camera[0,1,2] are the angle-axis rotation
```

注释说明相机参数数组的前 3 个元素是角轴旋转。

完整相机参数是 9 维：

```text
camera[0], camera[1], camera[2]  角轴旋转 r
camera[3], camera[4], camera[5]  平移 t
camera[6]                        焦距 f
camera[7]                        径向畸变 k1
camera[8]                        径向畸变 k2
```

从 SLAM 角度看，这个相机模型同时优化外参和部分内参：

```math
c = [r, t, f, k_1, k_2]
```

```cpp
T predictions[2];
```

创建一个长度为 2 的数组，用来保存预测的二维投影坐标。

因为 `T` 是模板类型，所以这里不是固定的 `double predictions[2]`。

如果 Ceres 正在自动求导，`T` 可能是 Jet 类型。这样投影计算中的每一步都会自动传播导数。

```cpp
CamProjectionWithDistortion(camera, point, predictions);
```

调用静态函数，计算带径向畸变的相机投影。

数学上就是：

```math
\hat{u}_{ij} = \pi(c_i, X_j)
```

其中 `hat` 表示预测值。

```cpp
residuals[0] = predictions[0] - T(observed_x);
residuals[1] = predictions[1] - T(observed_y);
```

计算残差：

```math
e_x = \hat{x} - x_{\text{obs}}
```

```math
e_y = \hat{y} - y_{\text{obs}}
```

也就是：

```math
e =
\begin{bmatrix}
\hat{x} - x_{\text{obs}} \\
\hat{y} - y_{\text{obs}}
\end{bmatrix}
```

这里写 `T(observed_x)` 是为了把 `double` 类型的观测值转换成模板类型 `T`。

如果 `T` 是 `double`，这个转换很普通。

如果 `T` 是 Jet，`T(observed_x)` 会构造一个 Jet 常量，使它能和 `predictions[0]` 做运算。

```cpp
return true;
```

告诉 Ceres 残差计算成功。

如果这里返回 `false`，Ceres 会认为当前残差块计算失败。

## 九、相机投影函数

```cpp
// camera : 9 dims array
// [0-2] : angle-axis rotation
// [3-5] : translateion
// [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
// point : 3D location.
// predictions : 2D predictions with center of the image plane.
```

这段注释解释参数布局。

其中 `translateion` 是拼写错误，应为 `translation`。

`second and forth order radial distortion` 指的是二阶和四阶径向畸变：

```math
1 + k_1r^2 + k_2r^4
```

这里不是说图像坐标有二阶向量，而是畸变模型里包含 `r^2` 和 `r^4` 两项。

```cpp
template<typename T>
static inline bool CamProjectionWithDistortion(
    const T *camera,
    const T *point,
    T *predictions) {
```

这也是模板函数，原因和 `operator()` 一样：它必须兼容 Ceres 自动求导。

`static` 表示这个函数属于类本身，不依赖某个具体对象。

也就是说，可以这样调用：

```cpp
SnavelyReprojectionError::CamProjectionWithDistortion(...)
```

不需要先创建一个 `SnavelyReprojectionError` 对象。

`inline` 表示建议编译器把函数体内联展开，减少函数调用开销。它也适合放在头文件里，避免多重定义问题。

从 C 的角度看，C 也有 `static` 和 `inline`，但 C++ 的 `static` 成员函数是类作用域里的函数，语义更偏向“这个函数属于这个类”。

## 十、角轴旋转

```cpp
// Rodrigues' formula
T p[3];
AngleAxisRotatePoint(camera, point, p);
```

`p` 是旋转后的三维点。

`AngleAxisRotatePoint(camera, point, p)` 使用 `camera[0..2]` 作为角轴旋转向量，把世界坐标中的三维点旋转到相机坐标系的方向上。

角轴向量记为：

```math
\omega =
\begin{bmatrix}
\omega_x \\
\omega_y \\
\omega_z
\end{bmatrix}
```

它的方向表示旋转轴，模长表示旋转角：

```math
\theta = \|\omega\|
```

通过 Rodrigues 公式可以得到旋转矩阵：

```math
R =
I + \sin\theta K + (1-\cos\theta)K^2
```

其中 `K` 是旋转轴对应的反对称矩阵。

这一步对应：

```math
P' = R X
```

## 十一、加平移

```cpp
// camera[3,4,5] are the translation
p[0] += camera[3];
p[1] += camera[4];
p[2] += camera[5];
```

把平移加到旋转后的点上。

这一步得到相机坐标系下的三维点：

```math
P_c = R X + t
```

写成分量：

```math
P_c =
\begin{bmatrix}
X_c \\
Y_c \\
Z_c
\end{bmatrix}
```

在 SLAM 中，这就是从世界坐标系到相机坐标系的变换。

注意这里的 `t` 不是相机中心 `C`。如果相机位姿写成：

```math
P_c = R(P_w - C)
```

那么：

```math
t = -RC
```

所以 `camera[3..5]` 是外参中的平移项，而不是直接的相机中心坐标。

## 十二、透视投影

```cpp
// Compute the center fo distortion
T xp = -p[0] / p[2];
T yp = -p[1] / p[2];
```

这里有一个注释拼写错误：`fo` 应该是 `of`。

代码把相机坐标系下的三维点投影到归一化成像平面：

```math
x = -\frac{X_c}{Z_c}
```

```math
y = -\frac{Y_c}{Z_c}
```

一般针孔相机模型常写成：

```math
x = \frac{X_c}{Z_c}, \quad y = \frac{Y_c}{Z_c}
```

这里多了负号，是 BAL/Snavely 数据集使用的相机约定。很多 SfM 数据集中相机看向负 z 方向，所以投影时使用负号。

不要把这个负号简单理解成错误，它是坐标系约定的一部分。

## 十三、径向畸变参数

```cpp
// Apply second and fourth order radial distortion
const T &l1 = camera[7];
const T &l2 = camera[8];
```

这里用引用 `const T &` 给 `camera[7]` 和 `camera[8]` 起别名。

`l1` 对应 `k1`，`l2` 对应 `k2`。

C++ 引用可以理解为“变量别名”。C 语言没有引用，只能用指针表达类似效果。

例如 C 中可能写：

```c
const double *l1 = &camera[7];
```

然后使用时写 `*l1`。

C++ 引用用起来更像普通变量：

```cpp
l1
```

这里加 `const` 是因为不希望修改相机参数。

## 十四、计算畸变系数

```cpp
T r2 = xp * xp + yp * yp;
```

计算归一化平面坐标到光心的平方距离：

```math
r^2 = x^2 + y^2
```

```cpp
T distortion = T(1.0) + r2 * (l1 + l2 * r2);
```

计算径向畸变系数：

```math
d = 1 + r^2(k_1 + k_2r^2)
```

展开就是：

```math
d = 1 + k_1r^2 + k_2r^4
```

径向畸变描述的是图像点离中心越远，畸变通常越明显。

这里只考虑径向畸变，没有考虑切向畸变。

`T(1.0)` 和前面的 `T(observed_x)` 类似，是为了兼容自动求导类型。

## 十五、乘焦距得到预测观测

```cpp
const T &focal = camera[6];
```

读取焦距。

这里的相机内参模型很简化，只优化一个焦距 `f`，没有显式优化主点坐标 `cx, cy`。

注释中也写了：

```text
predictions : 2D predictions with center of the image plane.
```

意思是图像坐标以成像平面中心为原点。

```cpp
predictions[0] = focal * distortion * xp;
predictions[1] = focal * distortion * yp;
```

得到最终预测坐标：

```math
\hat{x} = f d x
```

```math
\hat{y} = f d y
```

完整投影过程可以写成：

```math
\begin{aligned}
P_c &= RX + t \\
x &= -X_c/Z_c \\
y &= -Y_c/Z_c \\
r^2 &= x^2 + y^2 \\
d &= 1 + k_1r^2 + k_2r^4 \\
\hat{u} &=
\begin{bmatrix}
fdx \\
fdy
\end{bmatrix}
\end{aligned}
```

然后残差为：

```math
e =
\begin{bmatrix}
fdx - u_x \\
fdy - u_y
\end{bmatrix}
```

```cpp
return true;
```

表示投影计算成功。

## 十六、创建 Ceres 代价函数

```cpp
static ceres::CostFunction *Create(
    const double observed_x,
    const double observed_y) {
```

这是一个静态工厂函数，用来创建 Ceres 可以识别的代价函数对象。

函数返回：

```cpp
ceres::CostFunction *
```

这是 Ceres 的基类指针。C++ 允许用基类指针指向派生类对象，这是多态的基础。

C 语言没有类继承和虚函数。C 里如果要模拟多态，通常需要结构体里放函数指针。

```cpp
return (new ceres::AutoDiffCostFunction<
    SnavelyReprojectionError, 2, 9, 3>(
        new SnavelyReprojectionError(observed_x, observed_y)));
```

这一行信息非常密集。

`new SnavelyReprojectionError(observed_x, observed_y)` 创建一个残差 functor 对象，里面保存当前观测值。

外层的：

```cpp
ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>
```

表示用 Ceres 自动求导包装这个 functor。

模板参数含义是：

```text
SnavelyReprojectionError  残差 functor 类型
2                         残差维度
9                         第一个参数块维度，也就是 camera
3                         第二个参数块维度，也就是 point
```

因此 Ceres 知道：

```text
operator()(camera, point, residuals)

camera    是 9 维
point     是 3 维
residuals 是 2 维
```

从图优化角度看，这一条边连接：

```text
9 维相机顶点 + 3 维路标点顶点 -> 2 维重投影误差
```

对应雅可比维度：

```math
J_c = \frac{\partial e}{\partial c} \in \mathbb{R}^{2\times9}
```

```math
J_p = \frac{\partial e}{\partial X} \in \mathbb{R}^{2\times3}
```

Ceres 自动求导会根据模板类型 `T` 自动生成这两个雅可比，不需要手写。

## 十七、private 成员

```cpp
private:
    double observed_x;
    double observed_y;
```

`private` 表示这两个成员变量只能在类内部访问。

它们保存当前观测的真实二维位置。

把它们设为私有成员，是 C++ 封装思想：外部不应该随意修改一条残差边的观测值。

在 C 中，结构体字段通常可以随便访问：

```c
error.observed_x = 100.0;
```

C++ 通过 `private` 限制这种直接修改。

从 SLAM 理论上说，观测值是测量数据，是固定量；优化变量是相机参数和三维点。

所以这里的设计是合理的：

```text
observed_x, observed_y  固定
camera, point           优化
residuals               输出误差
```

## 十八、结束头文件保护

```cpp
};
```

结束类定义。

注意 C++ 类定义末尾必须有分号。C 的 `struct` 定义末尾也需要分号。

```cpp
#endif // SnavelyReprojection.h
```

结束头文件保护。

严格说，前面的宏名是：

```cpp
SnavelyReprojection_H
```

而注释写的是：

```cpp
SnavelyReprojection.h
```

这只是注释，不影响编译。

## 十九、把代码翻译成数学公式

这份代码实现的残差模型是：

```math
e =
\begin{bmatrix}
e_x \\
e_y
\end{bmatrix}
=
\begin{bmatrix}
\hat{x} - x_{\text{obs}} \\
\hat{y} - y_{\text{obs}}
\end{bmatrix}
```

其中：

```math
P_c = R(\omega)X + t
```

```math
x = -\frac{P_{c,x}}{P_{c,z}},
\quad
y = -\frac{P_{c,y}}{P_{c,z}}
```

```math
r^2 = x^2 + y^2
```

```math
d = 1 + k_1r^2 + k_2r^4
```

```math
\hat{x} = fdx,
\quad
\hat{y} = fdy
```

最终：

```math
e =
\begin{bmatrix}
fdx - x_{\text{obs}} \\
fdy - y_{\text{obs}}
\end{bmatrix}
```

BA 就是让所有观测的这个误差平方和尽量小：

```math
\min
\sum_{(i,j)}
\left\|
\pi(c_i, X_j) - u_{ij}
\right\|^2
```

加入 Huber 核后就是鲁棒最小二乘：

```math
\min
\sum_{(i,j)}
\rho\left(
\left\|
\pi(c_i, X_j) - u_{ij}
\right\|^2
\right)
```

## 二十、为什么这段代码能让 Ceres 自动优化

Ceres 需要知道三件事：

1. 优化变量是什么。
2. 残差怎么计算。
3. 每个参数块和残差的维度是多少。

这份头文件提供第 2 和第 3 件事：

```cpp
AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>
```

告诉 Ceres：

```text
残差函数：SnavelyReprojectionError::operator()
残差维度：2
相机参数维度：9
三维点参数维度：3
```

而 `bundle_adjustment_ceres.cpp` 里的：

```cpp
problem.AddResidualBlock(cost_function, loss_function, camera, point);
```

告诉 Ceres：

```text
这条边连接哪个 camera 参数块和哪个 point 参数块。
```

之后 Ceres 会自动完成：

1. 用当前参数计算所有残差。
2. 用自动求导计算雅可比。
3. 构造线性化系统。
4. 用 LM 或信赖域方法求解增量。
5. 更新相机和三维点。
6. 重复直到收敛。

所以这份代码虽然短，但它定义了 BA 的核心物理意义：  
**一个三维点经过相机外参、内参和畸变模型投影后，应该尽量落在真实观测位置上。**

