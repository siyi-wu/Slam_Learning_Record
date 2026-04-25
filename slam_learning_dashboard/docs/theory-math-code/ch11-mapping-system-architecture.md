# 第11章理论：建图系统与数据关联

## 1. 章节主线

SLAM 的“图”不只是一张点云，而是一组随时间演化的数据结构：关键帧、地图点、观测关系与线程状态。

## 2. 数学公式

单点重投影约束：

$$
\mathbf{e}_{ij} = \mathbf{u}_{ij} - \pi(\mathbf{T}_{cw,i}\mathbf{P}_{w,j})
$$

地图点常用质量指标可写为：

$$
s_j = f(n_j, \bar{\theta}_j, \sigma_{reproj,j})
$$

其中 $n_j$ 是观测次数，$\bar{\theta}_j$ 是平均视角基线，$\sigma_{reproj,j}$ 是重投影误差统计量。

## 3. 系统架构关注点

1. 关键帧管理：何时插入、何时剔除，决定地图规模与实时性。
2. 地图点管理：可见性、重投影误差、视角基线共同决定点质量。
3. 数据关联：候选检索 + 几何验证，决定系统鲁棒性上限。
4. 线程协同：Tracking / LocalMapping / LoopClosing 的共享状态一致性。

## 4. 代码映射

文件：
- `code/ch11/loop_closure.cpp`
- `code/ch11/feature_training.cpp`
- `code/ch11/gen_vocab_large.cpp`

对应关系：
1. 特征词袋是“快速候选检索”的数据索引层。
2. 回环检测线程产出候选，几何模块再决定是否入图。
3. 关键帧和地图点生命周期管理决定系统稳定性上限。

## 5. 实战提醒

1. 建图问题常见瓶颈不是优化器，而是数据结构一致性。
2. 地图点剔除规则要和前端跟踪策略协同设计。
3. 多线程下优先保证状态可重复和可回放。

## 6. 网络资料（精选）

1. ORB-SLAM2 仓库（关键帧/地图点/词袋架构）：[raulmur/ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2)
2. ORB-SLAM2 论文 DOI：[10.1109/TRO.2017.2705103](https://doi.org/10.1109/TRO.2017.2705103)
3. DBoW3 词袋库：[rmsalinas/DBoW3](https://github.com/rmsalinas/DBow3)
4. GTSAM 因子图教程（架构建模参考）：[Factor Graphs and GTSAM](https://gtsam.org/tutorials/intro.html)
