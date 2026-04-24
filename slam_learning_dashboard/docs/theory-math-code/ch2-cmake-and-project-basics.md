# 第2章理论：工程组织与构建基础

## 1. 物理概念

第2章不是几何推导章，但它决定后续实验是否可复现。
核心是把“算法”变成“可稳定编译与运行的工程”。

## 2. 结构模型

可以把项目抽象成：

$$
\text{Project} = (\text{Source},\ \text{Dependencies},\ \text{BuildConfig})
$$

构建输出是将源代码和依赖映射为可执行文件：

$$
f_{build}: (S, D, C) \rightarrow Binaries
$$

## 3. 代码映射

- `code/ch2/CMakeLists.txt`
- `code/ch2/usehello.cpp`
- `code/ch2/libtest.cpp`

对应关系：
1. `add_executable` 定义可执行目标。
2. `target_link_libraries` 管理模块依赖关系。

## 4. 实战提醒

1. 从第2章开始统一编译选项，避免后续章节环境漂移。
2. 每章保留最小可运行样例，便于回归测试。
