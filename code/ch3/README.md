# ch3 学习脚手架：三维刚体运动与 Eigen 基础

## 章节目标

1. 掌握 Eigen 的矩阵/向量基本操作。
2. 理解旋转、平移与齐次变换的表示方式。
3. 能在不同坐标系之间进行点变换。

## 文件入口

- `code/ch3/useEigen/eigenMatrix.cpp`
- `code/ch3/useGeometry/useGeometry.cpp`
- `code/ch3/examples/coordinateTransform.cpp`
- `code/ch3/examples/plotTrajectory.cpp`

## 学习检查点

1. 你能说明 `Matrix3d`、`Vector3d`、`Isometry3d` 的用途。
2. 你能写出 `p2 = T21 * p1` 的代码并解释物理意义。
3. 你能读懂轨迹可视化示例中的坐标系约定。

## 建议实践

1. 在 `useGeometry.cpp` 中更换旋转轴和角度，验证输出变化。
2. 在 `coordinateTransform.cpp` 中加入自定义测试点并手算对照。
