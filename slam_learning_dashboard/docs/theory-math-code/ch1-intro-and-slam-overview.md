# 第1章理论：SLAM问题定义与系统全景

## 1. 核心问题

SLAM（Simultaneous Localization and Mapping）解决的是：
1. 机器人当前“在什么位置、什么姿态”（Localization）。
2. 周围环境“长什么样”（Mapping）。

视觉 SLAM 中，传感器主要是相机。系统通常拆成四块：
1. 前端视觉里程计（Tracking / VO）：估计相邻帧运动。
2. 后端优化（Backend）：做全局一致性优化。
3. 回环检测（Loop Closing）：发现“走回老地方”并校正累计漂移。
4. 建图（Mapping）：维护稀疏或稠密地图。

## 2. 概率建模主线

离散时刻的状态与观测可写成：

$$
\mathbf{x}_k = [\mathbf{T}_{wk}, \mathcal{M}_k],\quad
\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{n}_k
$$

更完整地看，SLAM是状态估计问题：

$$
p(\mathbf{x}_{0:k}, \mathcal{M}\mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k})
$$

其中 $\mathbf{u}_{1:k}$ 是运动先验（如 IMU / 里程计），$\mathbf{z}_{1:k}$ 是视觉观测。后续章节的几何、优化都在为这个后验估计服务。

## 3. 视觉SLAM系统认知重点

1. 单目存在尺度不确定性；双目和 RGB-D 可直接观测尺度。
2. 前端追踪稳定性决定“能不能跑起来”，后端与回环决定“能跑多远而不飘”。
3. 实际工程里，SLAM常以“局部地图实时+全局图低频优化”的方式平衡实时性和精度。

## 4. 代码映射

本章主要做总览认知，建议从以下入口建立全局图：
1. `docs/code-map/slambook2-chapter-map.md`
2. `docs/theory-math-code/index.md`
3. `code/README.md`

## 5. 实战提醒

1. 先画自己的模块数据流：输入传感器、输出轨迹、地图更新、优化触发条件。
2. 每学习一个算法，都定位它是“前端约束生成”还是“后端状态求解”。

## 6. 网络资料（精选）

1. 《视觉SLAM十四讲》代码仓（作者维护）：[gaoxiang12/slambook2](https://github.com/gaoxiang12/slambook2)
2. 《视觉SLAM十四讲》第一版代码与章节目录说明：[gaoxiang12/slambook](https://github.com/gaoxiang12/slambook)
3. 英文版讲义与源码（便于对照术语）：[gaoxiang12/slambook-en](https://github.com/gaoxiang12/slambook-en)
4. SLAM经典综述（Part I，IEEE 2006）：[DOI:10.1109/MRA.2006.1638022](https://doi.org/10.1109/MRA.2006.1638022)
