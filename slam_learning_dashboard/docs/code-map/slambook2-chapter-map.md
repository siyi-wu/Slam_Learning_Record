# 《视觉SLAM十四讲》章节映射总表

> 说明：本表按“全书一体化”组织，便于连续学习与扩展。

| 章节 | 主题 | 对应目录/文档入口 |
|---|---|---|
| 第1章 | 预备知识与课程导览 | `docs/index.md` |
| 第2章 | 初识 SLAM 与 CMake | `code/ch2`，`code/ch2/README.md` |
| 第3章 | 三维空间刚体运动（Eigen） | `code/ch3`，`code/ch3/README.md`，`theory-math-code/rigid-body-motion.md` |
| 第4章 | 李群李代数（Sophus） | `code/ch4`，`code/ch4/README.md`，`theory-math-code/lie-group-lie-algebra.md` |
| 第5章 | 相机模型与 RGB-D/双目 | `code/ch5`，`code/ch5/README.md` |
| 第6章 | 非线性优化基础（GN） | `code/ch6`，`code/ch6/README.md`，`theory-math-code/nonlinear-optimization.md` |
| 第7章 | 视觉里程计前端与几何估计 | `code/ch7`，`code/ch7/README.md`，`theory-math-code/epipolar-geometry.md` |
| 第8章 | 光流与直接法 | `code/ch8`，`code/ch8/README.md`，`theory-math-code/nonlinear-optimization.md` |
| 第9章 | 后端优化与图优化扩展 | `code/ch9`，`code/ch9/README.md`，`notes/chapters/ch9-backend-optimization.md` |
| 第10章 | 回环检测与位姿图 | `code/ch10`，`code/ch10/README.md`，`notes/chapters/ch10-loop-closure.md` |
| 第11章 | 建图与工程系统组织 | `code/ch11`，`code/ch11/README.md`，`notes/chapters/ch11-mapping-system.md` |
| 第12章 | 稠密重建与点云处理 | `code/ch12`，`code/ch12/README.md`，`notes/chapters/ch12-dense-reconstruction.md` |
| 第13章 | 视觉惯导/多传感器融合扩展 | `code/ch13`，`code/ch13/README.md`，`notes/chapters/ch13-vio-fusion.md` |
| 第14章 | 总结与工程实践路线 | `code/ch14`，`code/ch14/README.md`，`notes/chapters/ch14-engineering-practice.md` |

## 维护规则

1. 每章都维护“代码入口 + 文档入口”两个链接位。
2. 新增示例优先放在 `code/chX`，并在对应章节文档补上说明。
3. 章节扩展时只增内容，不改变主线结构。
