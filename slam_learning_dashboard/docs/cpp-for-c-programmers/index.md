# 写给 C 程序员的 SLAM C++ 指南（全书版）

> 目标读者：有扎实 C 语言基础，正在学习《视觉SLAM十四讲》，希望快速读懂并改写 `slambook2` 风格 C++代码。
>
> 编写策略：只讲 **C 和 C++差异**，跳过你已经熟悉的控制流、基础变量定义。

## 1. 本指南范围与使用方式

- 你的本地代码位于 `code`。
- 本指南统一面向《十四讲》全书的 C++工程写法。
- 章节中的示例分为两类：
  - 直接可运行的本地示例（来自 `code` 目录）。
  - 全书通用的工程范式（用于后续章节扩展）。

## 2. 先建立一张“C -> C++”迁移地图

1. 函数参数传递：`指针 + 手工判空` -> `const 引用/引用`。
2. 数据组织：`struct + 函数散落` -> `class + 成员函数 + 封装`。
3. 多态扩展：`函数指针表` -> `virtual + override`。
4. 泛型复用：`宏/void*` -> `template`。
5. 容器与算法：`手写数组/链表` -> `std::vector/std::map + 算法库`。
6. 资源管理：`malloc/free, new/delete` -> `RAII + 智能指针`。

## 3. 与你本地代码强相关的入口

- 引用、`const` 传参、STL 容器：
  - `code/ch7/pose_estimation_3d2d.cpp`
  - `code/ch7/triangulation.cpp`
- 类、继承、多态（g2o 顶点与边）：
  - `code/ch7/pose_estimation_3d3d.cpp`
  - `code/ch7/pose_estimation_3d2d.cpp`
- 面向对象封装 + 并行调用：
  - `code/ch8/optical_flow.cpp`
  - `code/ch8/direct_method.cpp`

## 4. 阅读顺序（建议）

1. [引用、OOP 与多态](./references-oop.md)
2. [模板与 STL（SLAM 高频）](./template-stl.md)
3. [智能指针与 RAII](./smart-pointer-raii.md)

## 5. 学习目标（完成本模块后）

- 你可以把 C 风格函数改造成更安全的 C++接口。
- 你能读懂 g2o/Ceres 中“继承 + 重写”的优化建模代码。
- 你能判断在 SLAM 工程里什么时候该用 `vector/map`、什么时候用智能指针。
- 你能识别“会泄漏/会悬垂/会越界”的旧写法，并替换为 RAII 风格。
