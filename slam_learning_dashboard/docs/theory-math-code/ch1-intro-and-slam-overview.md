# 第1章理论：SLAM问题定义与系统全景

## 1. 物理概念

SLAM 的核心任务是“边定位、边建图”。
机器人在未知环境中运动时，需要同时估计自身位姿与地图结构。

## 2. 数学表达

系统状态可抽象为：

$$
\mathbf{x}_k = [\mathbf{T}_{wk}, \mathcal{M}_k]
$$

观测模型可写为：

$$
\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{n}_k
$$

其中 $\mathbf{T}_{wk}$ 是位姿，$\mathcal{M}_k$ 是地图，$\mathbf{n}_k$ 是噪声。

## 3. 代码映射

本章偏系统认知，建议结合以下入口建立全局图：
- `docs/code-map/slambook2-chapter-map.md`
- `code/README.md`

## 4. 实战提醒

1. 先建立“前端-后端-回环-建图”模块图，再深入单算法。
2. 学每段代码时都明确它属于状态估计链路的哪一环。
