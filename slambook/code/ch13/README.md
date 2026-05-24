Frame：一帧双目图像
Feature：图像里的二维特征点
MapPoint：由特征点三角化得到的三维地图点

一帧 Frame
 ├── 左图 left_img_
 ├── 右图 right_img_
 ├── 位姿 pose_
 └── 很多 Feature

一个 Feature
 ├── 在图像上的 2D 位置
 └── 可能关联一个 MapPoint

一个 MapPoint
 ├── 在世界坐标系下的 3D 位置
 └── 被多个 Feature 观测到

map关键帧保留：
1. 不要保留太相似的关键帧
2. 也不要让局部地图离当前帧太远
3. 让 active_keyframes_ 围绕当前帧，形成一个局部窗口

frontend：
接收新图像帧
初始化地图
跟踪相机运动
估计当前帧位姿
判断跟踪质量
插入关键帧
生成新的地图点
通知后端优化
通知 viewer 可视化

