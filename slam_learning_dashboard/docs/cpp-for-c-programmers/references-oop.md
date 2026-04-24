# 引用、面向对象与多态（C 程序员高频痛点）

## 1. 引用与 `const`：从“能跑”到“可维护”

### 1.1 C 与 C++差异

C 里你常见：
```c
void foo(Point* p);   // 需要判空，调用方容易传错
```

C++ 在 SLAM 中更常见：
```cpp
void foo(const Point& p);   // 不可为空，且不拷贝
```

核心收益：
- 语义明确：这个参数“必须存在”。
- 减少拷贝：大对象（如 `Eigen::Matrix`、`std::vector`）按引用传递。
- 可读性高：`const` 明确“只读输入”。

### 1.2 你本地代码中的真实模式

在 `code/ch7/pose_estimation_3d2d.cpp` 中：

```cpp
void bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose
);
```

解释：
- `const VecVector3d &points_3d`：只读输入，避免拷贝大量 3D 点。
- `Sophus::SE3d &pose`：输出参数，函数内部直接更新位姿。

### 1.3 C 思维迁移提醒

1. `const T&` 优先于 `T*`（只读、非空语义更强）。
2. 只有“可能为空”或“可选参数”才考虑指针。
3. 输出参数建议显式命名，如 `pose_out`（工程代码更清楚）。

---

## 2. 类（Class）封装：把“散函数 + 全局变量”收拢到对象

### 2.1 C 与 C++差异

C 常见组织：
- 一堆 `struct` + 一堆函数 + 若干全局配置。

SLAM C++常见组织：
- 一个类持有状态（图像、点、相机参数），成员函数执行计算。

### 2.2 你本地代码中的真实模式

`code/ch8/optical_flow.cpp`：

```cpp
class OpticalFlowTracker {
public:
    OpticalFlowTracker(
        const Mat &img1_,
        const Mat &img2_,
        const vector<KeyPoint> &kp1_,
        vector<KeyPoint> &kp2_,
        vector<bool> &success_,
        bool inverse_ = true,
        bool has_initial_ = false)
        : img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_),
          success(success_), inverse(inverse_), has_initial(has_initial_) {}

    void calculateOpticalFlow(const Range &range);

private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};
```

关键点：
- 构造函数初始化列表：在对象创建时绑定依赖，避免后续“半初始化状态”。
- `private` 成员：把中间状态封装，减少外部误改。
- 类函数 `calculateOpticalFlow`：明确“这个算法属于这个对象”。

---

## 3. 继承与多态：为什么 g2o 代码看起来“像框架”

### 3.1 C 与 C++差异

C 的“多态替代方案”通常是函数指针 + 上下文指针。  
C++ 在 SLAM 里大量使用：`基类接口 + 虚函数重写`。

### 3.2 你本地代码中的真实模式

`code/ch7/pose_estimation_3d3d.cpp`：

```cpp
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override {
    _estimate = Sophus::SE3d();
  }

  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }
};
```

理解方式：
- `BaseVertex<6, Sophus::SE3d>` 是“接口模板 + 维度约束”。
- 你实现 `setToOriginImpl` 与 `oplusImpl`，等于告诉优化器“我的状态怎么初始化、怎么更新”。
- `override` 能帮你在编译期检查签名是否匹配，避免“以为重写了其实没有”的隐藏 bug。

### 3.3 为什么这对 SLAM 很关键

SLAM 后端会有多种残差块/顶点类型：
- 重投影误差边
- IMU 预积分误差边
- 位姿图约束边

都可以复用统一优化框架，只换你自己写的派生类逻辑。

---

## 4. 一段“C 风格 -> C++风格”改写示例

```cpp
// C++ 改写版：用引用和类封装状态
class PoseRefiner {
public:
    // 输入点云只读，位姿可写
    void refine(const std::vector<Eigen::Vector3d>& pts_world,
                const std::vector<Eigen::Vector2d>& pts_img,
                Sophus::SE3d& pose) {
        // 1) 构建误差项
        // 2) 计算雅可比
        // 3) 更新 pose（这里省略具体优化细节）
        // 注：将优化循环封装在成员函数里，调用方不需要接触内部临时变量。
    }
};
```

对 C 程序员的直觉翻译：
- 把“输入输出指针协议”升级为“类型系统表达语义”。
- 把“靠注释约定”升级为“靠 `const`、访问控制、override 约束”。

---

## 5. 本章小结

1. 在 SLAM 代码中，`const T&` 是默认输入形态。  
2. 类不是“为了面向对象而面向对象”，而是为了管理算法状态和依赖。  
3. g2o/Ceres 风格代码难点不在语法，而在“通过继承把数学模型接入框架”。
