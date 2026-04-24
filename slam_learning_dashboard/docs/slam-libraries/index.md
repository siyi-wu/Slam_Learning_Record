# SLAM 核心第三方库解析（全书导向）

本模块覆盖《视觉SLAM十四讲》中最关键的六个库：
- Eigen
- Sophus
- OpenCV
- Ceres
- g2o
- PCL

统一讲解框架：
1. 核心数据结构（你在代码里最常碰到的类型）。
2. 适用范围（这个库在 SLAM 流程里负责什么）。
3. 典型 API 调用范例（可直接复用到你后续代码中）。

## 库使用分布

- 前端与几何基础：Eigen / Sophus / OpenCV
- 后端优化核心：g2o / Ceres
- 稠密建图与点云：PCL

你的本地代码根目录：`code`

## 建议阅读顺序

1. [Eigen：数值计算底座](./eigen.md)
2. [Sophus：SE(3)/SO(3) 李群表示](./sophus.md)
3. [OpenCV：视觉前端与图像处理](./opencv.md)
4. [g2o：图优化后端](./g2o.md)
5. [Ceres：非线性最小二乘优化](./ceres.md)
6. [PCL：点云处理与三维重建](./pcl.md)
