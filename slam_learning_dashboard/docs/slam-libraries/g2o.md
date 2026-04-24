# g2o：图优化后端

## 对齐级别
- ：你的本地 `code` 已有完整示例。

## 1. 核心数据结构

1. 优化器与求解器：
- `g2o::SparseOptimizer`
- `g2o::BlockSolver<...>`
- `g2o::OptimizationAlgorithmGaussNewton`
- `g2o::OptimizationAlgorithmLevenberg`

2. 图元素：
- 顶点：`g2o::BaseVertex<Dim, EstimateType>`
- 边：`g2o::BaseUnaryEdge / BaseBinaryEdge`

3. 信息矩阵：
- `setInformation(...)` 定义观测置信度。

## 2. 适用范围

g2o 在 SLAM 中主要负责：
- BA（Bundle Adjustment）
- 位姿图优化（Pose Graph）
- 各类几何约束的非线性最小二乘

你的本地典型文件：
- `code/ch7/pose_estimation_3d2d.cpp`
- `code/ch7/pose_estimation_3d3d.cpp`

## 3. 典型 API 范例

### 3.1 自定义顶点与边

```cpp
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <sophus/se3.hpp>

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    void oplusImpl(const double* update) override {
        Eigen::Matrix<double, 6, 1> dx;
        dx << update[0], update[1], update[2], update[3], update[4], update[5];
        // 左乘更新，和李群优化公式对应
        _estimate = Sophus::SE3d::exp(dx) * _estimate;
    }

    bool read(std::istream&) override { return true; }
    bool write(std::ostream&) const override { return true; }
};
```

### 3.2 组装优化问题并求解

```cpp
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

int main() {
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
    );

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // 1) 添加顶点
    // 2) 添加边
    // 3) 设置信息矩阵
    // 4) initializeOptimization + optimize

    optimizer.initializeOptimization();
    optimizer.optimize(10);
    return 0;
}
```

## 4. 常见坑与建议

1. `Edge::linearizeOplus()` 的雅可比符号非常容易写反，先做数值梯度对拍。
2. 信息矩阵尺度要匹配噪声模型，否则会出现收敛慢或偏置。
3. 顶点更新规则（左乘/右乘）要与残差定义保持一致。
