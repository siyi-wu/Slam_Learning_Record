# 第8章理论：光流与直接法

## 1. 物理概念

第8章把“特征点几何”扩展到“像素强度优化”。
直接法不显式提取特征，而是直接最小化灰度误差。

## 2. 数学公式

光度误差：

$$
e = I_1(\mathbf{u}) - I_2(\pi(\mathbf{T}\mathbf{P}))
$$

链式求导：

$$
\frac{\partial e}{\partial \boldsymbol{\xi}}=
\frac{\partial e}{\partial I}
\frac{\partial I}{\partial \mathbf{u}}
\frac{\partial \mathbf{u}}{\partial \boldsymbol{\xi}}
$$

## 3. 代码映射

- `code/ch8/optical_flow.cpp`
- `code/ch8/direct_method.cpp`

## 4. 实战提醒

1. 直接法对光照变化敏感，建议做金字塔与鲁棒核处理。
2. 像素梯度质量会直接影响收敛稳定性。
