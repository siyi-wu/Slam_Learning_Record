# 第2章理论：工程组织与构建基础

## 1. 为什么这一章很关键

第2章看似“工程杂项”，但它是后面所有实验可复现的前提。  
SLAM 项目常见失败不是算法错，而是构建环境不一致、依赖版本漂移、目标链接不完整。

## 2. 构建系统抽象

可以把项目写成三元组：

$$
\text{Project} = (\text{Source},\ \text{Dependencies},\ \text{BuildConfig})
$$

构建过程是映射：

$$
f_{build}: (S,D,C) \rightarrow \text{Binaries}
$$

其中：
1. `Source`：源码与头文件组织。
2. `Dependencies`：Eigen / OpenCV / Sophus / Ceres / g2o 等外部库。
3. `BuildConfig`：编译器、标准、优化等级、链接规则。

## 3. CMake中的核心对象

1. 目标（Target）：`add_executable` / `add_library`。
2. 依赖关系：`target_link_libraries`。
3. 头文件可见性：`target_include_directories`。
4. 编译特性：`target_compile_features` / `set(CMAKE_CXX_STANDARD ...)`。

理解“以目标为中心”后，项目规模扩大也不容易失控。

## 4. 代码映射

1. `code/ch2/CMakeLists.txt`
2. `code/ch2/usehello.cpp`
3. `code/ch2/libtest.cpp`
4. `code/ch2/libtest.h`

对应关系：
1. `CMakeLists.txt` 定义了“源码 -> 目标 -> 链接关系”。
2. `usehello.cpp + libtest.cpp` 展示了最小的多文件工程组织。

## 5. 实战提醒

1. 尽早固定 C++ 标准和第三方版本，避免章节间行为变化。
2. 每章保持一个最小可运行 demo，后续改动都可回归验证。
3. 将“编译通过”与“结果正确”分开检查，别把二者混为一谈。

## 6. 网络资料（精选）

1. CMake 官方教程（最新版本）：[CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
2. 《视觉SLAM十四讲》第一版代码仓（含 ch2）：[gaoxiang12/slambook](https://github.com/gaoxiang12/slambook)
3. 《视觉SLAM十四讲》第二版代码仓（按章节组织）：[gaoxiang12/slambook2](https://github.com/gaoxiang12/slambook2)
