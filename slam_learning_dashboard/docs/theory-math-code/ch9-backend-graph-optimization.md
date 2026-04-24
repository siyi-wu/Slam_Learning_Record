# 第9章理论：后端图优化

## 1. 物理概念

前端给出“观测”，后端要在全局范围内找到一组最一致的状态。  
图优化把每个状态看成顶点，把观测约束看成边。

## 2. 数学公式

统一目标函数：

$$
\min_{\mathbf{x}} \sum_{k} \rho_k\left(\mathbf{e}_k(\mathbf{x})^\top \mathbf{\Omega}_k \mathbf{e}_k(\mathbf{x})\right)
$$

线性化后得到法方程：

$$
\mathbf{H}\Delta \mathbf{x} = \mathbf{b}, \quad
\mathbf{H}=\sum_k \mathbf{J}_k^\top\mathbf{\Omega}_k\mathbf{J}_k,\quad
\mathbf{b}=-\sum_k \mathbf{J}_k^\top\mathbf{\Omega}_k\mathbf{e}_k
$$

## 3. 代码映射

文件：
- `code/ch9/bundle_adjustment_g2o.cpp`
- `code/ch9/bundle_adjustment_ceres.cpp`
- `code/ch9/SnavelyReprojectionError.h`

对应关系：
1. 误差项 `reprojection error` 对应每条观测边。
2. Jacobian 在 g2o/Ceres 中由边或残差块自动组织。
3. 鲁棒核（Huber/Cauchy）对应目标函数中的 $\rho(\cdot)$。

## 4. 实战提醒

1. 调后端先看残差分布，再看轨迹误差。
2. 若不收敛，优先排查外点、初值和尺度。
3. 稀疏问题尽量保持块结构，避免稠密化。
