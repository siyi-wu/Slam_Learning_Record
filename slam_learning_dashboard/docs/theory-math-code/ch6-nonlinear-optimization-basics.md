# 第6章理论：非线性优化基础

## 1. 章节核心

SLAM 后端本质是在“噪声观测下估计最可能状态”。  
工程里通常转成非线性最小二乘并迭代求解。

## 2. 从 MAP 到最小二乘

在高斯噪声假设下，最大后验估计可转成：

$$
\min_{\mathbf{x}} \sum_i \|\mathbf{e}_i(\mathbf{x})\|_{\mathbf{\Omega}_i}^2
=
\min_{\mathbf{x}} \sum_i \mathbf{e}_i^\top \mathbf{\Omega}_i \mathbf{e}_i
$$

其中 $\mathbf{\Omega}_i$ 是信息矩阵（协方差的逆）。

## 3. Gauss-Newton 与 LM

线性化后，Gauss-Newton 法方程：

$$
\mathbf{H}\Delta\mathbf{x}=\mathbf{b},\quad
\mathbf{H}=\sum_i\mathbf{J}_i^\top\mathbf{\Omega}_i\mathbf{J}_i,\quad
\mathbf{b}=-\sum_i\mathbf{J}_i^\top\mathbf{\Omega}_i\mathbf{e}_i
$$

Levenberg-Marquardt 在病态或远离最优时更稳：

$$
(\mathbf{H}+\lambda\mathbf{I})\Delta\mathbf{x}=\mathbf{b}
$$

理解这两步后，后续 BA、位姿图优化、回环优化的数学形式都能串起来。

## 4. 代码映射

1. `code/ch6/gaussNewton.cpp`
2. `code/ch6/CMakeLists.txt`
3. `code/ch6/README.md`

对应关系：
1. `gaussNewton.cpp`：手写“构建残差/Jacobian -> 组装法方程 -> 迭代更新”。
2. `CMakeLists.txt`：引入优化相关依赖并构建可执行程序。

## 5. 实战提醒

1. 优化不收敛先查初值、外点、尺度，再调求解器参数。
2. 观察每次迭代的代价函数变化，避免“看似跑完但结果无效”。
3. 大系统优先利用稀疏结构，避免把问题稠密化。

## 6. 深入阅读

1. [非线性优化（专题版）](./nonlinear-optimization.md)

## 7. 网络资料（精选）

1. Ceres 非线性最小二乘教程（官方）：[Ceres NNLS Tutorial](https://ceres-solver.org/nnls_tutorial.html)
2. g2o 官方仓库与说明：[RainerKuemmerle/g2o](https://github.com/RainerKuemmerle/g2o)
3. 《视觉SLAM十四讲》第一版代码（ch6 对应 Ceres/g2o）：[gaoxiang12/slambook](https://github.com/gaoxiang12/slambook)
4. 《视觉SLAM十四讲》第二版代码（ch6 目录）：[gaoxiang12/slambook2](https://github.com/gaoxiang12/slambook2)
