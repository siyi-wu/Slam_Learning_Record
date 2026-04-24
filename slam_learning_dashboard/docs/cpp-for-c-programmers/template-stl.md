# 模板与 STL（SLAM 代码的“基础设施”）

## 1. 模板（Template）：告别 `void*` 和重复代码

### 1.1 C 与 C++差异

C 里做泛型常见两种方式：
- `void*` + 强制类型转换（类型不安全）。
- 宏展开（调试困难，可读性差）。

C++ 里：
- 模板在编译期生成类型安全代码。
- SLAM 常见用途：固定维度矩阵、容器别名、泛型工具函数。

### 1.2 你本地代码可见的模板痕迹

`code/ch7/pose_estimation_3d2d.cpp`：

```cpp
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
```

这本质上是模板实例化：
- `std::vector<T, Allocator>` 的 `T` 和分配器都是模板参数。
- `Eigen::aligned_allocator` 处理 SSE/AVX 对齐需求，避免未对齐访问问题。

### 1.3 你需要立刻掌握的最小模板写法

```cpp
template <typename Derived>
double SquaredNorm(const Eigen::MatrixBase<Derived>& v) {
    // 统一处理 Vector2d/Vector3d/动态维向量
    return v.squaredNorm();
}
```

意义：
- 一个函数适配多种向量类型。
- 编译器会在编译期展开，通常没有运行时额外开销。

---

## 2. STL 容器：在 SLAM 中最常用的四个

## 2.1 `std::vector`：默认首选

典型场景：
- 特征点集合、匹配集合、轨迹序列、残差块列表。

你本地代码：
- `code/ch7/triangulation.cpp`
- `code/ch8/optical_flow.cpp`

使用建议：
1. 已知规模时先 `reserve(n)`，减少扩容开销。
2. 遍历时优先 `const auto&`，避免拷贝。
3. 与 OpenCV/Eigen 交互时留意元素类型与对齐。

示例：
```cpp
std::vector<cv::KeyPoint> keypoints;
keypoints.reserve(2000);  // 预分配，减少重复内存搬移

for (const auto& kp : keypoints) {
    // 只读遍历，不拷贝 KeyPoint
}
```

## 2.2 `std::map` / `std::unordered_map`：做索引与关联

全书常见场景：
- 以 `frame_id` 查帧对象。
- 以 `landmark_id` 查路标点对象。

选择经验：
- 需要有序遍历或范围查询：`std::map`。
- 主要是高频查找：`std::unordered_map`。

示例：
```cpp
std::unordered_map<unsigned long, Eigen::Vector3d> landmarks;
landmarks.emplace(42UL, Eigen::Vector3d(1.0, 2.0, 3.0));

auto it = landmarks.find(42UL);
if (it != landmarks.end()) {
    const Eigen::Vector3d& pw = it->second;
    // 使用 pw 做投影或误差计算
}
```

## 2.3 `std::array`：固定小尺寸数据

在常量维度、栈上存储的小块数据上可用（例如小窗口像素偏移）。

```cpp
std::array<int, 8> dx = {-1, 0, 1, -1, 1, -1, 0, 1};
std::array<int, 8> dy = {-1, -1, -1, 0, 0, 1, 1, 1};
```

## 2.4 `std::deque`：双端队列（滑窗缓存常见）

全书后续常见：
- 滑动窗口中按时间推进帧，头删尾插频繁。

```cpp
std::deque<double> imu_timestamps;
imu_timestamps.push_back(12.34);
imu_timestamps.pop_front();
```

---

## 3. STL 算法：把“手写循环”变成“语义化表达”

C 风格常见：手写 for 循环筛选、统计。  
C++推荐：优先标准算法。

```cpp
std::vector<cv::DMatch> good_matches;
good_matches.reserve(matches.size());

std::copy_if(matches.begin(), matches.end(), std::back_inserter(good_matches),
             [](const cv::DMatch& m) {
                 return m.distance < 30.0;  // 筛掉差匹配
             });
```

收益：
- 逻辑表达更直接（“筛选”而不是“循环细节”）。
- 更不容易写出越界或索引错误。

---

## 4. Eigen 模板与 STL 混用的一个关键坑

当容器里存 Eigen 定长向量/矩阵（如 `Vector3d`）时，需要考虑对齐：

```cpp
using VecVector3d = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
```

如果遗漏对齐分配器，可能在特定平台/编译选项下触发崩溃或性能退化。

---

## 5. 本章小结

1. 模板是“类型安全 + 零成本抽象”的核心，不是语法炫技。  
2. `vector` 是 SLAM 数据流默认容器；`map/unordered_map` 负责对象索引。  
3. STL 算法能减少样板循环，降低 C 迁移时期的细节 bug。
