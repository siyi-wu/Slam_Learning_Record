# 第14章理论：工程评估与实验设计

## 1. 物理概念

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

## 3. 代码映射

文件：
- `code/ch14/README.md`

建议工程落点：
1. `configs/` 保存实验参数模板。
2. `logs/` 保存运行日志与指标快照。
3. `scripts/eval/` 统一输出 ATE/RPE/耗时统计。

## 4. 实战提醒

1. 每次实验固定随机种子和依赖版本。
2. 结论必须基于多次重复实验，不看单次结果。
3. 建立回归基线，防止后续改动破坏已有性能。
