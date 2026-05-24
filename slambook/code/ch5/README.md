# ch5 学习脚手架：相机模型、图像处理与 RGB-D/双目

## 章节目标

1. 掌握 OpenCV 图像读取、访问与基础操作。
2. 理解相机内参与像素坐标到相机坐标的映射。
3. 能完成双目视差与 RGB-D 点云拼接的基础流程。

## 文件入口

- `code/ch5/imageBasics/imageBasics.cpp`
- `code/ch5/imageBasics/undistortImage.cpp`
- `code/ch5/stereoVIsion/stereoVision.cpp`
- `code/ch5/rgbd/joinMap.cpp`

## 学习检查点

1. 你能解释去畸变中像素映射的几何含义。
2. 你能说明视差、焦距、基线与深度的关系。
3. 你能追踪 `joinMap.cpp` 中从深度图到三维点云的流程。

## 建议实践

1. 修改畸变参数并观察 `undistortImage` 输出差异。
2. 在 `stereoVision.cpp` 中调整 SGBM 参数，比较视差图质量。
