# Ceres：非线性最小二乘优化

## 对齐级别
- ：全书后续高频库，你本地 `code` 目前暂未出现。

## 1. 核心数据结构

1. 问题与残差块：
- `ceres::Problem`
- `AddResidualBlock(...)`

2. 代价函数：
- 自动求导：`ceres::AutoDiffCostFunction`
- 解析雅可比：`ceres::SizedCostFunction`

3. 求解配置：
- `ceres::Solver::Options`
- `ceres::Solver::Summary`

4. 鲁棒核：
- `ceres::HuberLoss`
- `ceres::CauchyLoss`

## 2. 适用范围

Ceres 适合：
- BA 与位姿优化
- 标定问题（相机/IMU）
- 任何可写成残差平方和的非线性优化问题

相对 g2o，Ceres 的特点：
- 自动求导使用门槛低，上手快。
- 对参数块局部参数化支持成熟（旋转、四元数等）。

## 3. 典型 API 范例

### 3.1 自动求导残差

```cpp
#include <ceres/ceres.h>
#include <Eigen/Core>

struct ReprojectionError {
    ReprojectionError(const Eigen::Vector3d& pw, const Eigen::Vector2d& uv, double fx, double fy, double cx, double cy)
        : pw_(pw), uv_(uv), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    template <typename T>
    bool operator()(const T* const se3, T* residual) const {
        // 示例中用 se3[0..5] 表示位姿增量参数，仅演示接口形式。
        // 实际工程中应结合 Sophus/Quaternion 局部参数化。
        T X = T(pw_.x()) + se3[0];
        T Y = T(pw_.y()) + se3[1];
        T Z = T(pw_.z()) + se3[2] + T(1e-6);

        T u = T(fx_) * X / Z + T(cx_);
        T v = T(fy_) * Y / Z + T(cy_);

        residual[0] = T(uv_.x()) - u;
        residual[1] = T(uv_.y()) - v;
        return true;
    }

    Eigen::Vector3d pw_;
    Eigen::Vector2d uv_;
    double fx_, fy_, cx_, cy_;
};
```

### 3.2 组装与求解

```cpp
#include <ceres/ceres.h>

int main() {
    double pose[6] = {0, 0, 0, 0, 0, 0}; // 待优化参数
    ceres::Problem problem;

    // 伪代码：遍历观测，添加残差块
    // for (...) {
    //   auto* cost = new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6>(
    //       new ReprojectionError(pw, uv, fx, fy, cx, cy));
    //   problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), pose);
    // }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 20;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    return 0;
}
```

## 4. 常见坑与建议

1. 自动求导很方便，但性能敏感场景建议手写雅可比对比。
2. 始终配合鲁棒核处理外点。
3. 参数尺度差异大时要做归一化，否则优化病态。
