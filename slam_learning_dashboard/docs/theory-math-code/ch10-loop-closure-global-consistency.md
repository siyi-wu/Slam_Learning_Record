# 第10章理论：回环检测与全局一致性

## 1. 章节主线

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

## 3. 回环检测的工程流程

1. 候选检索：基于词袋/检索索引快速找到历史关键帧候选。
2. 几何验证：通过特征匹配、位姿验证剔除误闭环。
3. 图优化：把回环边加入位姿图，全局优化分发修正量。

## 4. 代码映射

文件：
- `code/ch10/pose_graph_g2o_SE3.cpp`
- `code/ch10/pose_graph_g2o_lie_algebra.cpp`

对应关系：
1. 回环边与里程计边共同组成图的边集合 $\mathcal{E}$。
2. 每次优化都在“当前漂移估计”基础上做增量修正。
3. Lie Algebra 版本和 SE3 版本本质一致，只是状态参数化不同。

## 5. 实战提醒

1. 回环候选必须配几何验证，防止误闭环。
2. 先做局部窗口筛选，再做全局图优化更稳定。
3. 回环触发后要同步更新可视化和地图索引，避免前后端状态不一致。

## 6. 网络资料（精选）

1. ORB-SLAM2 仓库（含回环与重定位模块）：[raulmur/ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2)
2. ORB-SLAM2 论文 DOI：[10.1109/TRO.2017.2705103](https://doi.org/10.1109/TRO.2017.2705103)
3. DBoW3 词袋库（开源实现）：[rmsalinas/DBoW3](https://github.com/rmsalinas/DBow3)
4. g2o 官方仓库：[RainerKuemmerle/g2o](https://github.com/RainerKuemmerle/g2o)
