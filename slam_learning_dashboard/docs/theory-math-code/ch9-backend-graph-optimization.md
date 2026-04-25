# 第9章理论：后端图优化

## 1. 章节主线

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

## 3. BA与图优化统一视角

1. BA 是图优化在“相机位姿 + 地图点”上的特例。
2. 位姿图优化常只优化位姿节点，计算量更可控。
3. 鲁棒核函数 $\rho(\cdot)$ 用于削弱外点影响，是工程稳定性的关键。

## 4. 代码映射

文件：
- `code/ch9/bundle_adjustment_g2o.cpp`
- `code/ch9/bundle_adjustment_ceres.cpp`
- `code/ch9/SnavelyReprojectionError.h`

对应关系：
1. 误差项 `reprojection error` 对应每条观测边。
2. Jacobian 在 g2o/Ceres 中由边或残差块自动组织。
3. 鲁棒核（Huber/Cauchy）对应目标函数中的 $\rho(\cdot)$。

## 5. 实战提醒

1. 调后端先看残差分布，再看轨迹误差。
2. 若不收敛，优先排查外点、初值和尺度。
3. 稀疏问题尽量保持块结构，避免稠密化。

## 6. 网络资料（精选）

1. g2o 官方仓库与介绍：[RainerKuemmerle/g2o](https://github.com/RainerKuemmerle/g2o)
2. Ceres 非线性最小二乘教程（官方）：[Ceres NNLS Tutorial](https://ceres-solver.org/nnls_tutorial.html)
3. g2o 论文 DOI（ICRA 2011）：[10.1109/ICRA.2011.5979949](https://doi.org/10.1109/ICRA.2011.5979949)
4. 《视觉SLAM十四讲》第二版代码（ch9）：[gaoxiang12/slambook2](https://github.com/gaoxiang12/slambook2)
