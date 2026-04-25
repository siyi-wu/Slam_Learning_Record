# 第14章理论：工程评估与实验设计

## 1. 章节主线

系统能跑通只是起点，工程化目标是“可复现、可比较、可迭代”。  
评估体系要同时覆盖精度、鲁棒性、实时性与资源开销。

## 2. 数学公式

ATE（绝对轨迹误差）：

$$
\text{ATE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\left\|\text{trans}\left(\mathbf{T}_{gt,i}^{-1}\mathbf{T}_{est,i}\right)\right\|^2}
$$

RPE（相对位姿误差）：

$$
\mathbf{E}_i = \left(\mathbf{T}_{gt,i}^{-1}\mathbf{T}_{gt,i+\Delta}\right)^{-1}
\left(\mathbf{T}_{est,i}^{-1}\mathbf{T}_{est,i+\Delta}\right)
$$

## 3. 评估维度建议

1. 精度：ATE / RPE 的均值、RMSE、分位数。
2. 鲁棒性：失败率、重定位成功率、长序列漂移。
3. 实时性：前端帧率、后端平均优化耗时、峰值延迟。
4. 资源：CPU/GPU 占用、内存、模型大小与地图大小。

## 4. 代码映射

文件：
- `code/ch14/README.md`

建议工程落点：
1. `configs/` 保存实验参数模板。
2. `logs/` 保存运行日志与指标快照。
3. `scripts/eval/` 统一输出 ATE/RPE/耗时统计。

## 5. 实战提醒

1. 每次实验固定随机种子和依赖版本。
2. 结论必须基于多次重复实验，不看单次结果。
3. 建立回归基线，防止后续改动破坏已有性能。

## 6. 网络资料（精选）

1. TUM RGB-D Benchmark（官方）：[RGB-D Dataset and Benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
2. TUM 官方评测工具说明（ATE/RPE）：[tools](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools)
3. KITTI 里程计评测（官方）：[KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
4. `evo` 轨迹评估工具（常用开源）：[MichaelGrupp/evo](https://github.com/MichaelGrupp/evo)
5. 轨迹评估教程工具箱（RPG）：[rpg_trajectory_evaluation](https://github.com/uzh-rpg/rpg_trajectory_evaluation)
