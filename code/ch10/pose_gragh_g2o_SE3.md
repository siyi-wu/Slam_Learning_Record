# `pose_graph_g2o_SE3.cpp` 代码精读

本文围绕 `pose_graph_g2o_SE3.cpp` 展开，目标有两条线：

1. 逐行解释代码在做什么。
2. 从 SLAM 理论角度理解位姿图优化、SE3 约束、信息矩阵和 g2o 求解过程。

这个程序完成的任务是：

```text
从 sphere.g2o 读取一个位姿图，把每个位姿作为顶点，把相对位姿观测作为边，
然后用 g2o 对整张图做非线性优化，最后保存优化后的 result.g2o。
```

## 一、位姿图优化是什么

位姿图优化，英文是 Pose Graph Optimization。

它和第 9 章的 BA 都属于图优化，但优化对象不同：

```text
BA:
顶点 = 相机位姿 + 三维地图点
边   = 图像中的二维观测
误差 = 重投影误差

Pose Graph:
顶点 = 机器人或相机的位姿
边   = 两个位姿之间的相对运动观测
误差 = 相对位姿误差
```

在这个程序中，每个顶点是一个 SE3 位姿：

```math
T_i =
\begin{bmatrix}
R_i & t_i \\
0 & 1
\end{bmatrix}
\in SE(3)
```

每条边是两个位姿之间的相对位姿观测：

```math
Z_{ij}
```

如果当前估计的两个位姿是 `T_i` 和 `T_j`，那么根据当前估计得到的相对变换是：

```math
\hat{Z}_{ij} = T_i^{-1}T_j
```

它应该接近传感器、里程计、匹配或回环检测给出的观测：

```math
Z_{ij}
```

所以位姿图优化的目标是：

```math
\min_{\{T_i\}}
\sum_{(i,j)}
e_{ij}^T \Omega_{ij} e_{ij}
```

其中：

- `e_ij` 是相对位姿误差。
- `Ω_ij` 是信息矩阵。
- 信息矩阵越大，表示这条约束越可信。

在 SE3 上，常见误差写法是：

```math
e_{ij} =
\log
\left(
Z_{ij}^{-1} T_i^{-1}T_j
\right)
\in \mathbb{R}^6
```

这个文件使用 g2o 已经提供好的 `VertexSE3` 和 `EdgeSE3`，所以误差计算细节由 g2o 内部封装。

## 二、头文件

```cpp
#include <iostream>
#include <fstream>
#include <string>
```

这三行引入标准库功能：

- `iostream` 用于终端输出，例如 `cout`。
- `fstream` 用于读取 `.g2o` 文件，例如 `ifstream`。
- `string` 用于保存每一行开头的标签，例如 `VERTEX_SE3:QUAT` 和 `EDGE_SE3:QUAT`。

```cpp
#include <g2o/types/slam3d/types_slam3d.h>
```

引入 g2o 已经定义好的 3D SLAM 类型。

这个程序主要用到：

```cpp
g2o::VertexSE3
g2o::EdgeSE3
```

`VertexSE3` 表示 SE3 位姿顶点。

`EdgeSE3` 表示两个 SE3 位姿之间的相对约束。

```cpp
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
```

这三行引入 g2o 优化器需要的组件：

- `block_solver.h`：块求解器，利用图优化中的块状稀疏结构。
- `optimization_algorithm_levenberg.h`：Levenberg-Marquardt 非线性优化算法。
- `linear_solver_eigen.h`：基于 Eigen 的线性方程求解器。

图优化每次迭代都会线性化，并求解类似下面的线性系统：

```math
H\Delta x = -b
```

这里的线性求解器负责求 `Δx`。

## 三、命名空间与程序说明

```cpp
using namespace std;
```

后面可以直接写 `cout`、`ifstream`、`string`，不用每次写 `std::cout`、`std::ifstream`、`std::string`。

```cpp
/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 这里使用g2o/types/slam3d/中的SE3表示位姿，它实质上是四元数而非李代数.
 * **********************************************/
```

这段注释说明了程序目的。

重点有三个：

1. 使用 g2o 做位姿图优化。
2. 输入文件是 `sphere.g2o`。
3. 位姿使用 g2o 内置的 SE3 类型，读写格式是四元数形式。

`.g2o` 文件中主要有两类记录：

```text
VERTEX_SE3:QUAT ...
EDGE_SE3:QUAT ...
```

`VERTEX_SE3:QUAT` 表示一个位姿顶点，常见格式是：

```text
VERTEX_SE3:QUAT id tx ty tz qx qy qz qw
```

`EDGE_SE3:QUAT` 表示两个位姿之间的一条边，常见格式是：

```text
EDGE_SE3:QUAT id1 id2 tx ty tz qx qy qz qw information_matrix_upper_triangle
```

其中 `tx ty tz qx qy qz qw` 表示相对位姿观测，后面的信息矩阵表示这条约束的置信度。

## 四、main 函数与参数检查

```cpp
int main(int argc, char **argv) {
```

程序入口。

`argc` 是命令行参数数量，`argv` 是命令行参数数组。

如果运行：

```bash
./pose_graph_g2o_SE3 sphere.g2o
```

那么：

```text
argc = 2
argv[0] = "./pose_graph_g2o_SE3"
argv[1] = "sphere.g2o"
```

```cpp
if (argc != 2) {
    cout << "Usage: pose_graph_g2o_SE3 sphere.g2o" << endl;
    return 1;
}
```

检查用户是否传入了一个 `.g2o` 文件。

如果参数数量不对，就打印使用方式并退出。

返回 `1` 表示程序异常结束。

## 五、打开输入文件

```cpp
ifstream fin(argv[1]);
```

创建输入文件流，打开用户传入的 `.g2o` 文件。

```cpp
if (!fin) {
    cout << "file " << argv[1] << " does not exist." << endl;
    return 1;
}
```

检查文件是否成功打开。

如果文件不存在或打开失败，就输出错误并退出。

## 六、创建 g2o 求解器

```cpp
typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
```

定义块求解器类型。

`BlockSolverTraits<6, 6>` 表示该优化问题使用 6 维块结构。

虽然 `.g2o` 文件里的位姿用：

```text
tx ty tz qx qy qz qw
```

一共 7 个数保存，但 SE3 的自由度是 6：

```text
3 维平移 + 3 维旋转
```

四元数用 4 个数表示旋转，但有单位长度约束：

```math
q_x^2 + q_y^2 + q_z^2 + q_w^2 = 1
```

所以旋转自由度仍然是 3。

```cpp
typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
```

定义线性求解器类型。

每次非线性优化迭代都会把问题线性化，得到：

```math
H\Delta x = -b
```

`LinearSolverType` 负责求解这个线性系统。

```cpp
auto solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
```

创建 Levenberg-Marquardt 优化算法。

求解器结构可以理解为：

```text
OptimizationAlgorithmLevenberg
    -> BlockSolver
        -> LinearSolverEigen
```

LM 方法在高斯牛顿和梯度下降之间做折中。

对于非线性最小二乘问题，普通高斯牛顿线性化后求：

```math
H\Delta x = -b
```

LM 会加入阻尼项：

```math
(H + \lambda I)\Delta x = -b
```

当当前估计比较差时，阻尼能让优化更稳定；当接近最优解时，LM 会更像高斯牛顿。

```cpp
g2o::SparseOptimizer optimizer;     // 图模型
```

创建稀疏优化器。

位姿图通常是稀疏图：每个关键帧只和少数相邻关键帧或回环关键帧有关。

所以 Hessian 矩阵也是稀疏的，`SparseOptimizer` 会利用这种结构。

```cpp
optimizer.setAlgorithm(solver);   // 设置求解器
optimizer.setVerbose(true);       // 打开调试输出
```

把 LM 求解器设置给优化器，并打开优化日志输出。

优化日志可以帮助观察误差是否下降、优化是否收敛。

## 七、循环读取 `.g2o` 文件

```cpp
int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量
```

记录读取到的顶点数量和边数量。

```cpp
while (!fin.eof()) {
```

循环读取文件，直到文件结束。

```cpp
string name;
fin >> name;
```

读取当前记录的类型名。

可能读到：

```text
VERTEX_SE3:QUAT
EDGE_SE3:QUAT
```

## 八、读取位姿顶点

```cpp
if (name == "VERTEX_SE3:QUAT") {
```

如果当前记录是 SE3 顶点，就创建位姿顶点。

```cpp
g2o::VertexSE3 *v = new g2o::VertexSE3();
```

创建一个 g2o 内置 SE3 顶点。

从 SLAM 角度看，这个	顶点可以表示：

- 某一帧相机位姿。
- 某一时刻机器人位姿。
- 轨迹中的一个关键帧位姿。

```cpp
int index = 0;
fin >> index;
```

读取顶点 ID。

顶点行通常类似：

```text
VERTEX_SE3:QUAT 0 x y z qx qy qz qw
```

这里先读出 `0` 这个 ID。

```cpp
v->setId(index);
```

设置顶点 ID。

g2o 用这个 ID 在边中找到对应顶点。

```cpp
v->read(fin);
```

让 `VertexSE3` 从文件流中读取剩下的数据。

剩下的数据通常是：

```text
tx ty tz qx qy qz qw
```

也就是平移加四元数。

```cpp
optimizer.addVertex(v);
```

把顶点加入优化器。

```cpp
vertexCnt++;
```

顶点计数加一。

```cpp
if (index == 0)
    v->setFixed(true);
```

如果当前顶点 ID 是 0，就固定这个顶点。

这一步非常重要。

位姿图只有相对约束，因此存在全局坐标系自由度。也就是说，所有位姿一起左乘一个固定变换，任意两个位姿之间的相对变换并不会改变：

```math
T_i' = S T_i
```

```math
(T_i')^{-1}T_j'
=
(S T_i)^{-1}(S T_j)
=
T_i^{-1}T_j
```

所以必须固定一个顶点，或者加入一个先验约束，才能确定全局坐标系。

这里固定第 0 个顶点，相当于把第一帧作为世界坐标系参考。

## 九、读取相对位姿边

```cpp
} else if (name == "EDGE_SE3:QUAT") {
```

如果当前记录是 SE3 边，就创建相对位姿约束。

```cpp
g2o::EdgeSE3 *e = new g2o::EdgeSE3();
```

创建一条 SE3 边。

`EdgeSE3` 连接两个 `VertexSE3`。

从 SLAM 角度看，它可以来自：

- 里程计相邻帧约束。
- 视觉前端估计出的相邻关键帧运动。
- ICP/NDT 等点云匹配得到的相对位姿。
- 回环检测得到的历史关键帧约束。

```cpp
int idx1, idx2;     // 关联的两个顶点
fin >> idx1 >> idx2;
```

读取这条边连接的两个顶点 ID。

边行通常类似：

```text
EDGE_SE3:QUAT 0 1 measurement information
```

这里 `0` 和 `1` 表示这条边连接顶点 0 和顶点 1。

```cpp
e->setId(edgeCnt++);
```

设置边的 ID，并让边计数加一。

```cpp
e->setVertex(0, optimizer.vertices()[idx1]);
e->setVertex(1, optimizer.vertices()[idx2]);
```

把边连接到两个顶点上。

这一步建立图结构：

```text
vertex idx1  ---- edge ----  vertex idx2
```

```cpp
e->read(fin);
```

让 `EdgeSE3` 从文件流读取剩下的边数据。

边数据包括：

```text
相对位姿测量 tx ty tz qx qy qz qw
信息矩阵上三角元素
```

对于 SE3 约束，误差是 6 维，所以信息矩阵是：

```math
\Omega_{ij} \in \mathbb{R}^{6\times6}
```

信息矩阵是协方差矩阵的逆：

```math
\Omega = \Sigma^{-1}
```

协方差越小，说明测量越可信，对应信息矩阵越大。

```cpp
optimizer.addEdge(e);
```

把边加入优化器。

加入边以后，g2o 就知道这两个位姿之间存在一个相对运动约束。

## 十、读取状态检查

```cpp
}
if (!fin.good()) break;
```

如果文件流状态不好，就退出循环。

可能原因包括：

- 已经读到文件末尾。
- 文件格式错误。
- 某个数字读取失败。

```cpp
}
```

结束读取循环。

此时所有顶点和边已经加入 `optimizer`。

## 十一、输出读取结果

```cpp
cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;
```

输出读取到的顶点数和边数。

这一步用于确认 `.g2o` 文件是否被完整读取。

## 十二、执行优化

```cpp
cout << "optimizing ..." << endl;
```

输出优化提示。

```cpp
optimizer.initializeOptimization();
```

初始化优化。

g2o 会整理图结构，准备顶点、边、雅可比和线性系统。

```cpp
optimizer.optimize(30);
```

执行最多 30 次迭代。

每一轮大致执行：

1. 根据当前位姿估计计算每条边的误差。
2. 对误差函数线性化，得到雅可比。
3. 构造正规方程。
4. 求解位姿增量。
5. 根据 LM 策略决定是否接受更新。
6. 更新位姿顶点。

整体目标函数是：

```math
F(T) =
\sum_{(i,j)}
e_{ij}^T \Omega_{ij} e_{ij}
```

线性化：

```math
e(x+\Delta x)
\approx
e(x) + J\Delta x
```

局部最小二乘问题：

```math
\min_{\Delta x}
\left\|
e + J\Delta x
\right\|^2
```

带信息矩阵时，对应正规方程：

```math
J^T \Omega J \Delta x
=
-J^T \Omega e
```

记：

```math
H = J^T \Omega J
```

```math
b = J^T \Omega e
```

则：

```math
H\Delta x = -b
```

LM 方法加入阻尼项：

```math
(H + \lambda I)\Delta x = -b
```

## 十三、保存结果

```cpp
cout << "saving optimization results ..." << endl;
```

输出保存提示。

```cpp
optimizer.save("result.g2o");
```

把优化后的图保存为 `result.g2o`。

因为这里使用的是 g2o 内置的 `VertexSE3` 和 `EdgeSE3`，所以可以直接调用 `optimizer.save()`。

保存后的文件可以用 `g2o_viewer` 打开，和原始 `sphere.g2o` 对比优化前后的轨迹形状。

```cpp
return 0;
```

程序正常结束。

```cpp
}
```

结束 `main` 函数。

## 十四、这份代码对应的图优化模型

这份程序的图模型是：

```text
顶点:
T_0, T_1, T_2, ..., T_n

边:
Z_ij, 连接 T_i 和 T_j
```

每个顶点：

```math
T_i \in SE(3)
```

每条边观测：

```math
Z_{ij} \in SE(3)
```

预测相对位姿：

```math
\hat{Z}_{ij} = T_i^{-1}T_j
```

误差：

```math
e_{ij} =
\log
\left(
Z_{ij}^{-1}\hat{Z}_{ij}
\right)
=
\log
\left(
Z_{ij}^{-1}T_i^{-1}T_j
\right)
```

其中 `log` 表示从 SE3 群映射到 se3 李代数：

```math
\log: SE(3) \rightarrow \mathbb{R}^6
```

最终优化目标：

```math
\min_{\{T_i\}}
\sum_{(i,j)}
e_{ij}^T \Omega_{ij} e_{ij}
```

这就是位姿图优化的核心。

## 十五、SE3 四元数表示和李代数表示

这个程序使用的是：

```cpp
g2o::VertexSE3
g2o::EdgeSE3
```

它读取 `.g2o` 文件中的格式：

```text
VERTEX_SE3:QUAT
EDGE_SE3:QUAT
```

这里的 `QUAT` 表示旋转用四元数存储。

一个位姿在文件中通常有 7 个数：

```text
tx ty tz qx qy qz qw
```

但是 SE3 的自由度是 6，不是 7。

原因是四元数有单位长度约束：

```math
q_x^2 + q_y^2 + q_z^2 + q_w^2 = 1
```

所以四元数虽然用 4 个数存储旋转，但只有 3 个旋转自由度。

g2o 内部优化时会在切空间里更新：

```math
\delta \xi \in \mathbb{R}^6
```

然后把增量作用到当前位姿上。

和 `pose_graph_g2o_lie_algebra.cpp` 相比：

```text
pose_graph_g2o_SE3.cpp:
使用 g2o 内置 VertexSE3 / EdgeSE3
读写和误差计算由 g2o 封装
代码短，适合快速使用

pose_graph_g2o_lie_algebra.cpp:
自定义顶点和边
自己写 read、write、computeError、linearizeOplus
代码长，但更适合理解数学细节
```

## 十六、为什么要固定第一个顶点

如果不固定任何顶点，位姿图只有相对约束。

所有位姿同时乘一个全局变换：

```math
T_i' = S T_i
```

任意相对位姿仍然不变：

```math
(T_i')^{-1}T_j'
=
T_i^{-1}T_j
```

因此优化问题无法确定全局坐标系。

这会导致 Hessian 矩阵奇异或接近奇异。

代码里：

```cpp
if (index == 0)
    v->setFixed(true);
```

就是把第一个位姿固定为世界坐标系参考，消除全局自由度。

## 十七、整体流程总结

程序流程可以浓缩为：

```text
1. 检查命令行参数
2. 打开 .g2o 文件
3. 创建 g2o 优化器
4. 设置 LM + BlockSolver + Eigen 线性求解器
5. 逐行读取 .g2o 文件
6. 遇到 VERTEX_SE3:QUAT 就创建位姿顶点
7. 遇到 EDGE_SE3:QUAT 就创建相对位姿约束边
8. 固定第一个顶点
9. 初始化优化
10. 迭代优化 30 次
11. 保存 result.g2o
```

从 SLAM 后端角度看，它完成的是：

```text
前端或回环检测提供相对位姿约束；
后端把这些约束组织成一张图；
优化器调整所有关键帧位姿；
让轨迹同时满足局部里程计约束和全局回环约束。
```

第 9 章 BA 优化的是相机位姿和三维地图点，误差是重投影误差。

第 10 章 Pose Graph 优化的是位姿本身，误差是两个位姿之间的相对变换误差。

所以 Pose Graph 比 BA 更轻量，常用于回环之后的全局轨迹校正。

