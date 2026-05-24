# ch2 学习脚手架：CMake 与最小 C++工程

## 章节目标

1. 跑通最小 C++工程编译链。
2. 理解头文件、源文件与库链接关系。
3. 建立后续章节统一的构建习惯。

## 文件入口

- `code/ch2/CMakeLists.txt`
- `code/ch2/libtest.h`
- `code/ch2/libtest.cpp`
- `code/ch2/usehello.cpp`

## 学习检查点

1. 你能解释 `add_library` 与 `add_executable` 的关系。
2. 你能把 `libtest` 中的函数扩展并在 `usehello.cpp` 调用。
3. 你能独立新建一个最小 CMake 子项目。

## 建议实践

1. 新增一个 `print_version()` 函数并完成编译运行。
2. 尝试把 `libtest` 改为静态库和动态库，观察产物差异。
