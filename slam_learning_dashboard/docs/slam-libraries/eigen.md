# Eigen：数值计算底座

## 对齐级别
- ：你的本地 `code` 已大量使用。

## 1. 核心数据结构

1. 固定维矩阵/向量：
- `Eigen::Matrix3d`
- `Eigen::Vector3d`
- `Eigen::Matrix<double, 6, 6>`

2. 动态维矩阵：
- `Eigen::MatrixXd`
- `Eigen::VectorXd`

3. 几何相关：
- `Eigen::Quaterniond`
- `Eigen::AngleAxisd`
- `Eigen::Isometry3d`

## 2. 适用范围

在 SLAM 中，Eigen 几乎无处不在：
- 前端：坐标变换、归一化坐标计算、三角化。
- 后端：构建雅可比 `J`、海森矩阵 `H = J^T J`、求解线性方程。
- 轨迹与位姿：旋转、平移、齐次变换。

你的本地典型文件：
- `code/ch3/useEigen/eigenMatrix.cpp`
- `code/ch3/useGeometry/useGeometry.cpp`
- `code/ch8/direct_method.cpp`

## 3. 典型 API 范例

### 3.1 构造矩阵、求解线性系统

```cpp
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

int main() {
    // H 是 6x6 海森矩阵，b 是 6x1 向量
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 6, 1> b;
    b << 1, 2, 3, 4, 5, 6;

    // SLAM 中常用 LDLT 分解求解 H * dx = b
    Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);

    std::cout << "dx = " << dx.transpose() << std::endl;
    return 0;
}
```

### 3.2 旋转与位姿变换

```cpp
#include <Eigen/Core>
#include <Eigen/Geometry>

int main() {
    // 绕 Z 轴旋转 45 度
    Eigen::AngleAxisd aa(M_PI / 4.0, Eigen::Vector3d(0, 0, 1));

    // 转旋转矩阵
    Eigen::Matrix3d R = aa.toRotationMatrix();

    // 构造 SE(3) 变换
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(R);
    T.pretranslate(Eigen::Vector3d(1, 2, 3));

    // 变换点坐标
    Eigen::Vector3d p(1, 0, 0);
    Eigen::Vector3d p_world = T * p;
    return 0;
}
```

## 4. 常见坑与建议

1. 容器存 `Eigen::Vector3d` 时考虑对齐分配器：
```cpp
using VecVector3d = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
```
2. 浮点比较不要用 `==`，用阈值判断。
3. 优先使用固定维矩阵（编译期优化更充分）。
