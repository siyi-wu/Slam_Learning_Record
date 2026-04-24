# Sophus：SE(3)/SO(3) 李群表示

## 对齐级别
- ：你的本地 `code` 已在多处使用。

## 1. 核心数据结构

1. 旋转群与李代数：
- `Sophus::SO3d`
- `SO3::log()` / `SO3::exp()`
- `SO3::hat()` / `SO3::vee()`

2. 位姿群与李代数：
- `Sophus::SE3d`
- `SE3::log()` / `SE3::exp()`
- `SE3::hat()` / `SE3::vee()`

## 2. 适用范围

Sophus 主要解决“如何稳定表示和更新位姿”：
- 优化中对位姿做小量更新：`T <- exp(\delta\xi) T`
- 轨迹误差计算（对数映射到李代数做度量）
- 图优化顶点状态更新（g2o/Ceres 自定义残差都常用）

你的本地典型文件：
- `code/ch4/useSophus.cpp`
- `code/ch7/pose_estimation_3d2d.cpp`
- `code/ch8/direct_method.cpp`

## 3. 典型 API 范例

### 3.1 SO(3) 与 so(3) 互转

```cpp
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

int main() {
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 6.0, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

    Sophus::SO3d so3(R);

    // 李群 -> 李代数
    Eigen::Vector3d omega = so3.log();

    // 李代数 -> 李群（微小扰动更新）
    Eigen::Vector3d d_omega(1e-4, 0, 0);
    Sophus::SO3d so3_new = Sophus::SO3d::exp(d_omega) * so3;
    return 0;
}
```

### 3.2 SE(3) 位姿更新（优化核心写法）

```cpp
#include <Eigen/Core>
#include <sophus/se3.hpp>

int main() {
    Sophus::SE3d T;  // 单位位姿

    // 6维增量：前3平移，后3旋转（按该 Sophus 版本约定）
    Eigen::Matrix<double, 6, 1> dx;
    dx << 1e-3, 0, 0, 1e-4, 2e-4, -1e-4;

    // 左乘更新
    T = Sophus::SE3d::exp(dx) * T;
    return 0;
}
```

## 4. 常见坑与建议

1. 不要直接对旋转矩阵逐元素加减；应在李代数空间更新。
2. 注意 `SE3` 六维向量顺序，统一团队约定（平移在前/旋转在前）。
3. 与 g2o/Ceres 联动时，把更新规则写在注释中，避免左右乘混淆。
