# 李群与李代数：为什么位姿优化要在切空间里做

## 1. 物理概念

你在优化中需要“微调姿态”。  
但旋转矩阵必须保持正交约束，不能直接做欧式加法：

$$
\mathbf{R}_{k+1} \neq \mathbf{R}_k + \Delta
$$

正确做法是：在李代数（切空间）中优化增量，再映射回李群。

## 2. 数学公式

SO(3) 更新：

$$
\mathbf{R}_{k+1} = \exp(\delta\boldsymbol{\phi}^\wedge)\mathbf{R}_k
$$

SE(3) 更新：

$$
\mathbf{T}_{k+1} = \exp(\delta\boldsymbol{\xi}^\wedge)\mathbf{T}_k,
\quad \delta\boldsymbol{\xi}\in \mathbb{R}^6
$$

其中 $\wedge$ 表示 hat 算子，$\vee$ 表示 vee 算子。

## 3. 代码映射

本地示例文件：
- `code/ch4/useSophus.cpp`

关键映射：
1. `log`：李群到李代数
- 公式：$\boldsymbol{\phi}=\log(\mathbf{R})$
- 代码：第 26 行 `Vector3d so3 = SO3_R.log();`

2. hat/vee
- 公式：$\boldsymbol{\phi}^\wedge$ 与逆运算 $\vee$
- 代码：第 28 行 `SO3d::hat`，第 30 行 `SO3d::vee`。

3. SO(3) 小量更新
- 公式：$\mathbf{R}'=\exp(\delta\boldsymbol{\phi}^\wedge)\mathbf{R}$
- 代码：第 32-34 行。

4. SE(3) 小量更新
- 公式：$\mathbf{T}'=\exp(\delta\boldsymbol{\xi}^\wedge)\mathbf{T}$
- 代码：第 53-57 行。

## 4. 与后端优化的连接

你在 g2o 顶点更新里看到的：

- `code/ch7/pose_estimation_3d2d.cpp` 第 256-260 行

本质就是把优化变量 `update` 转成 $\delta\boldsymbol{\xi}$，再执行指数映射左乘更新。

## 5. 实战提醒

1. 一定先统一“左乘更新”还是“右乘更新”，否则雅可比符号会错。
2. 调试时可先把更新量设小（如 `1e-4`）观察误差是否单调下降。
