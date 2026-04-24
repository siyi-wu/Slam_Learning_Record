---
layout: home

hero:
  name: "视觉SLAM学习仪表盘"
  text: "从物理直觉到工程代码的全书学习路径"
  tagline: "面向《视觉SLAM十四讲》：C->C++过渡、核心库拆解、数学公式到代码逐行映射"
  actions:
    - theme: brand
      text: 开始学习
      link: ./cpp-for-c-programmers/
    - theme: alt
      text: 查看章节映射
      link: ./code-map/slambook2-chapter-map

features:
  - title: C 程序员友好迁移
    details: 只讲 C 和 C++差异点，聚焦你在 SLAM 代码中高频遇到的语法与工程模式。
  - title: 六大核心库贯通
    details: Eigen、Sophus、OpenCV、Ceres、g2o、PCL，统一讲清“数据结构-适用场景-API范式”。
  - title: 公式到代码映射
    details: 刚体运动、李群李代数、非线性优化、对极几何，全部给出 LaTeX 推导与代码落点。
---

## 学习路线

1. C++ 过渡指南：[cpp-for-c-programmers](./cpp-for-c-programmers/)
2. 核心库解析：[slam-libraries](./slam-libraries/)
3. 理论深度映射：[theory-math-code](./theory-math-code/)
4. 全书章节索引：[code-map/slambook2-chapter-map](./code-map/slambook2-chapter-map)

## 扩展专题预留

- 双目匹配
- MVS（多视图几何）
- 回环检测
- VIO

入口目录：`docs/notes/chapters/`
