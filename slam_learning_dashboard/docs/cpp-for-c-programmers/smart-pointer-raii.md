# 智能指针与 RAII：把资源错误扼杀在编译期和作用域内

## 1. 先建立直觉：RAII 是 C++工程安全的底座

RAII（Resource Acquisition Is Initialization）核心思想：
- 资源在对象构造时获取。
- 资源在对象析构时自动释放。

对 C 程序员最关键的变化：
- 不再靠“记得 free/delete”来保证正确性。
- 即使中途 `return` 或异常，析构也会执行，资源能自动释放。

---

## 2. 为什么 SLAM 工程尤其需要 RAII

SLAM 代码常见资源：
- 堆对象（优化图节点、地图点、帧对象）
- 文件句柄、日志流
- 线程锁（`std::mutex`）
- OpenCV/Eigen/g2o 中的大对象与上下文

如果沿用 C 风格手工释放，随着模块增加（前端/后端/回环/建图），泄漏和悬垂指针风险会快速放大。

---

## 3. 你本地代码中的风险点

在 `code/ch7/pose_estimation_3d3d.cpp` 能看到：

```cpp
auto solver = new g2o::OptimizationAlgorithmLevenberg(...);
VertexPose *pose = new VertexPose();
EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(...);
```

这类写法在教程代码里便于讲解，但在工程化版本中有隐患：
- 控制流复杂后，容易漏掉对应 `delete`。
- 所有权关系不直观（谁负责释放？何时释放？）。

---

## 4. 智能指针三兄弟：在 SLAM 里的职责分工

## 4.1 `std::unique_ptr`（唯一所有权，默认首选）

适合：
- 明确只有一个拥有者的对象（例如某个模块内部私有求解器）。

```cpp
auto solver = std::make_unique<MySolver>();
solver->Solve();
// 离开作用域自动析构，无需 delete
```

## 4.2 `std::shared_ptr`（共享所有权）

适合：
- 一个对象被多个模块共享生命周期（典型：Frame/MapPoint 被地图与跟踪线程共同持有）。

```cpp
using FramePtr = std::shared_ptr<Frame>;
FramePtr frame = std::make_shared<Frame>(/* 构造参数 */);
```

注意：
- 引用计数有开销。
- 设计不当会形成循环引用（需要 `std::weak_ptr` 打断环）。

## 4.3 `std::weak_ptr`（弱引用，观察者）

适合：
- “我想访问对象，但不拥有它”的关系（例如地图点反向引用关键帧）。

```cpp
std::weak_ptr<Frame> host_frame;
if (auto f = host_frame.lock()) {
    // 对象仍存活，可以安全访问
}
```

---

## 5. 把 C 风格手工资源管理改成 RAII（示例）

## 5.1 传统写法（风险高）

```cpp
// 不推荐：容易在早退分支中泄漏
Optimizer* opt = new Optimizer();
if (!Init(opt)) {
    return;  // 这里泄漏
}
Run(opt);
delete opt;
```

## 5.2 RAII 写法（推荐）

```cpp
// 推荐：作用域结束自动释放
auto opt = std::make_unique<Optimizer>();
if (!Init(*opt)) {
    return;  // 不泄漏
}
Run(*opt);
```

这段代码的教学重点：
- `Init` 和 `Run` 使用引用，表达“必须存在对象”。
- 不需要在多个分支手动维护 `delete`。

---

## 6. 锁管理也要 RAII（并发模块非常关键）

你在 `code/ch8/direct_method.cpp` 里已有 `std::mutex`。  
推荐统一用 `std::lock_guard` / `std::unique_lock` 管理临界区：

```cpp
std::mutex mtx;

double SafeReadAndUpdate(double& value, double delta) {
    std::lock_guard<std::mutex> lock(mtx); // 构造时加锁，析构时解锁
    value += delta;
    return value;
}
```

收益：
- 早退、异常都不会忘记解锁。
- 比手写 `lock()/unlock()` 更稳健。

---

## 7. 给你的工程化迁移清单（从现在到全书）

1. 新增代码禁止裸 `new/delete`（除非被第三方接口强制）。
2. 模块私有对象优先 `unique_ptr`。
3. 跨模块共享对象使用 `shared_ptr + weak_ptr` 组合。
4. 所有互斥锁使用 RAII 锁封装。
5. 每个类在注释里写明“所有权关系”（谁持有谁）。

---

## 8. 本章小结

1. RAII 不是语法技巧，而是 C++工程正确性的核心机制。  
2. 智能指针让对象生命周期可见、可控、可推理。  
3. 你后续学习第9章以后时，越早坚持 RAII，后面调试成本越低。
