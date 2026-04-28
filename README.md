# Slam Learning Record

这是一个围绕《视觉SLAM十四讲》构建的个人学习仓库，目标是把**代码实践、理论笔记与章节索引**统一起来，形成可持续迭代的学习闭环。

## 仓库定位

1. 按章节维护 `code/chX` 代码与学习脚手架。
2. 记录理论推导、实验过程与复盘笔记。
3. 通过根目录 README 与各章 README 串联章节学习路径。

## 当前目录结构

```text
Slam_Learning_Record/
├── code/                      # 第2~14章代码与学习脚手架
│   ├── ch2/ ... ch14/
│   └── README.md              # code 总学习路线
├── Notes_SLAM/                # 视觉SLAM笔记
├── Notes_MVS/                 # MVS/双目/多视图学习笔记
├── build/                     # 本地构建产物
└── README.md
```

## 代码学习入口（`code/`）

- 统一入口：`code/README.md`
- 每章入口：`code/chX/README.md`（`X=2...14`）

当前各章 README 已建立，便于按章学习：

- `ch2`：CMake 与最小 C++ 工程
- `ch3`：Eigen 与刚体运动
- `ch4`：Sophus 与李群李代数
- `ch5`：相机模型 / 双目 / RGB-D
- `ch6`：Gauss-Newton 非线性优化
- `ch7`：特征法前端与几何估计
- `ch8`：光流法与直接法
- `ch9`：BA 与后端优化（g2o / Ceres）
- `ch10`：位姿图优化
- `ch11`：回环检测
- `ch12`：稠密重建与点云处理
- `ch13`：工程化 VO 系统结构
- `ch14`：评估与工程收尾

## 笔记模块

- `Notes_SLAM/`：视觉SLAM十四讲主笔记
- `Notes_MVS/`：MVS/多视图相关扩展学习

## 依赖与环境建议

常见依赖（不同章节会有差异）：

- CMake
- C++17 编译器（g++ / clang++）
- Eigen3
- OpenCV
- Sophus
- g2o
- Ceres
- PCL
- Pangolin
- Boost

## 参考资料

- 官方代码仓库：<https://github.com/gaoxiang12/slambook2>
