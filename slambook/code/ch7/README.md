# ch7 学习脚手架：特征法前端与几何估计

## 章节目标

1. 掌握特征提取、匹配与筛选流程。
2. 理解对极几何、PnP、三角化与 BA 的关系。
3. 能对比 OpenCV 求解与手写/g2o 优化实现。

## 文件入口

- `code/ch7/orb_cv.cpp`
- `code/ch7/pose_estimation_2d2d.cpp`
- `code/ch7/triangulation.cpp`
- `code/ch7/pose_estimation_3d2d.cpp`
- `code/ch7/pose_estimation_3d3d.cpp`

## 学习检查点

1. 你能解释 `E = [t]_x R` 与 `recoverPose` 的关系。
2. 你能说明 3D-2D 与 3D-3D 两种位姿估计问题的差异。
3. 你能读懂 g2o 顶点与边的残差、雅可比实现。

## 建议实践

1. 对比 `solvePnP`、手写 GN、g2o BA 的结果与耗时。
2. 在三角化结果上增加重投影误差筛选。
