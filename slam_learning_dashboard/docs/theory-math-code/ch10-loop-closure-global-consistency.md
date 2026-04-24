# 第10章理论：回环检测与全局一致性

## 1. 物理概念

视觉里程计会积累漂移，回环用于识别“回到旧地点”的时刻，并加入远距离约束修正全局地图。

## 2. 数学公式

回环约束残差（SE(3) 形式）：

$$
\mathbf{r}_{ij}=\log\left((\mathbf{T}_{ij}^{\text{meas}})^{-1}\mathbf{T}_i^{-1}\mathbf{T}_j\right)
$$

位姿图优化目标：

$$
\min_{\{\mathbf{T}_k\}} \sum_{(i,j)\in\mathcal{E}} \|\mathbf{r}_{ij}\|^2_{\mathbf{\Omega}_{ij}}
$$

## 3. 代码映射

文件：
- `code/ch10/pose_graph_g2o_SE3.cpp`
- `code/ch10/pose_graph_g2o_lie_algebra.cpp`

对应关系：
1. 回环边与里程计边共同组成图的边集合 $\mathcal{E}$。
2. 每次优化都在“当前漂移估计”基础上做增量修正。
3. Lie Algebra 版本和 SE3 版本本质一致，只是状态参数化不同。

## 4. 实战提醒

1. 回环候选必须配几何验证，防止误闭环。
2. 先做局部窗口筛选，再做全局图优化更稳定。
3. 回环触发后要同步更新可视化和地图索引，避免前后端状态不一致。
