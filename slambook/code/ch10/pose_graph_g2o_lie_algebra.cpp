#include <iostream> // 标准输入输出，用于打印运行信息。
#include <fstream>  // 文件流，用于读取和写入 .g2o 文件。
#include <string>   // 字符串类型，用于读取每一行记录的标签。
#include <memory>   // 智能指针工具，供 make_unique 创建求解器组件。
#include <Eigen/Core> // Eigen 核心矩阵和向量类型。

#include <g2o/core/base_vertex.h>                    // 自定义 g2o 顶点需要继承的基类。
#include <g2o/core/base_binary_edge.h>               // 自定义二元边需要继承的基类。
#include <g2o/core/block_solver.h>                   // 块求解器，用于构造位姿图优化问题。
#include <g2o/core/optimization_algorithm_levenberg.h> // Levenberg-Marquardt 优化算法。
#include <g2o/solvers/eigen/linear_solver_eigen.h>   // 基于 Eigen 的线性方程求解器。

#include <sophus/se3.hpp> // Sophus 的 SE3/SO3 李群李代数实现。

using namespace std; // 直接使用 std 命名空间中的 cout、ifstream、vector 等。
using namespace Eigen; // 直接使用 Eigen 命名空间中的矩阵、向量和四元数类型。
using Sophus::SE3d; // 使用 Sophus 的双精度 SE3 类型表示三维位姿。
using Sophus::SO3d; // 使用 Sophus 的双精度 SO3 类型表示三维旋转。

/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 本节使用李代数表达位姿图，节点和边的方式为自定义
 * **********************************************/

typedef Matrix<double, 6, 6> Matrix6d; // 定义 6x6 矩阵类型，对应 SE3 李代数的 6 个自由度。

// 给定误差求J_R^{-1}的近似
Matrix6d JRInv(const SE3d &e) { // 根据当前误差 SE3 计算右雅可比逆的近似。
    Matrix6d J; // 创建 6x6 雅可比矩阵。
    J.block(0, 0, 3, 3) = SO3d::hat(e.so3().log()); // 左上块由旋转误差的反对称矩阵构成。
    J.block(0, 3, 3, 3) = SO3d::hat(e.translation()); // 右上块由平移误差的反对称矩阵构成。
    J.block(3, 0, 3, 3) = Matrix3d::Zero(3, 3); // 左下块置零。
    J.block(3, 3, 3, 3) = SO3d::hat(e.so3().log()); // 右下块同样由旋转误差的反对称矩阵构成。
    // J = J * 0.5 + Matrix6d::Identity();
    J = Matrix6d::Identity();    // 实际代码中把右雅可比逆近似为单位矩阵，简化雅可比计算。
    return J; // 返回近似后的右雅可比逆矩阵。
}

// 李代数顶点
typedef Matrix<double, 6, 1> Vector6d; // 定义 6 维向量类型，对应 se(3) 的更新量或误差。

class VertexSE3LieAlgebra : public g2o::BaseVertex<6, SE3d> { // 自定义 SE3 李代数顶点，估计值类型为 Sophus::SE3d。
public: // 以下成员函数对 g2o 框架可见。
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 保证含有 Eigen 固定大小对象的类按要求进行内存对齐。

    virtual bool read(istream &is) override { // 从 .g2o 文件流中读取一个顶点的估计值。
        double data[7]; // 保存 tx, ty, tz, qx, qy, qz, qw 这 7 个数。
        for (int i = 0; i < 7; i++) // 逐个读取位姿数据。
            is >> data[i]; // 将第 i 个数据读入数组。
        setEstimate(SE3d( // 将读入数据转换为 Sophus::SE3d 并设置为顶点估计。
            Quaterniond(data[6], data[3], data[4], data[5]), // Eigen 四元数构造顺序为 w, x, y, z。
            Vector3d(data[0], data[1], data[2]) // 平移部分为 tx, ty, tz。
        ));
        return true; // 返回 true 表示读取成功。
    }

    virtual bool write(ostream &os) const override { // 将顶点估计值写入输出流。
        os << id() << " "; // 先写出顶点 ID。
        Quaterniond q = _estimate.unit_quaternion(); // 从 SE3 估计中取出单位四元数。
        os << _estimate.translation().transpose() << " "; // 写出平移 tx, ty, tz。
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl; // 写出四元数 qx, qy, qz, qw。
        return true; // 返回 true 表示写出成功。
    }

    virtual void setToOriginImpl() override { // g2o 要求实现的重置函数。
        _estimate = SE3d(); // 将估计值重置为 SE3 单位变换。
    }

    // 左乘更新
    virtual void oplusImpl(const double *update) override { // g2o 在优化中调用该函数更新顶点估计。
        Vector6d upd; // 创建 6 维李代数更新量。
        upd << update[0], update[1], update[2], update[3], update[4], update[5]; // 将 g2o 传入的数组拷贝到 Eigen 向量。
        _estimate = SE3d::exp(upd) * _estimate; // 将李代数增量通过指数映射转成 SE3，并左乘到当前估计上。
    }
};

// 两个李代数节点之边
class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, SE3d, VertexSE3LieAlgebra, VertexSE3LieAlgebra> { // 自定义连接两个 SE3 李代数顶点的二元边。
public: // 以下成员函数对 g2o 框架可见。
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 保证 Eigen 固定大小对象按要求进行内存对齐。

    virtual bool read(istream &is) override { // 从 .g2o 文件流中读取一条边的测量和信息矩阵。
        double data[7]; // 保存相对位姿测量 tx, ty, tz, qx, qy, qz, qw。
        for (int i = 0; i < 7; i++) // 逐个读取边的相对位姿测量。
            is >> data[i]; // 将第 i 个数据读入数组。
        Quaterniond q(data[6], data[3], data[4], data[5]); // 按 w, x, y, z 的顺序构造四元数。
        q.normalize(); // 归一化四元数，避免输入数值误差破坏旋转合法性。
        setMeasurement(SE3d(q, Vector3d(data[0], data[1], data[2]))); // 将相对位姿测量设置为边的测量值。
        for (int i = 0; i < information().rows() && is.good(); i++) // 遍历信息矩阵上三角的行。
            for (int j = i; j < information().cols() && is.good(); j++) { // 从对角线开始读取上三角元素。
                is >> information()(i, j); // 读取信息矩阵的 (i, j) 元素。
                if (i != j) // 非对角元素需要同步到对称位置。
                    information()(j, i) = information()(i, j); // 补全信息矩阵的下三角元素。
            }
        return true; // 返回 true 表示读取成功。
    }

    virtual bool write(ostream &os) const override { // 将边的数据写入输出流。
        VertexSE3LieAlgebra *v1 = static_cast<VertexSE3LieAlgebra *> (_vertices[0]); // 取出边连接的第一个顶点。
        VertexSE3LieAlgebra *v2 = static_cast<VertexSE3LieAlgebra *> (_vertices[1]); // 取出边连接的第二个顶点。
        os << v1->id() << " " << v2->id() << " "; // 写出这条边连接的两个顶点 ID。
        SE3d m = _measurement; // 取出边的 SE3 相对位姿测量。
        Eigen::Quaterniond q = m.unit_quaternion(); // 将测量中的旋转部分转成单位四元数。
        os << m.translation().transpose() << " "; // 写出测量中的平移部分。
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " "; // 写出四元数 qx, qy, qz, qw。

        // information matrix 
        for (int i = 0; i < information().rows(); i++) // 遍历信息矩阵上三角的行。
            for (int j = i; j < information().cols(); j++) { // 从对角线开始遍历上三角元素。
                os << information()(i, j) << " "; // 只写出上三角元素，符合 .g2o 文件格式。
            }
        os << endl; // 当前边写完后换行。
        return true; // 返回 true 表示写出成功。
    }

    // 误差计算与书中推导一致
    virtual void computeError() override { // 计算当前边的 6 维误差。
        SE3d v1 = (static_cast<VertexSE3LieAlgebra *> (_vertices[0]))->estimate(); // 取出第一个顶点的当前位姿估计。
        SE3d v2 = (static_cast<VertexSE3LieAlgebra *> (_vertices[1]))->estimate(); // 取出第二个顶点的当前位姿估计。
        _error = (_measurement.inverse() * v1.inverse() * v2).log(); // 预测相对位姿与测量相对位姿作差，并映射到李代数。
    }

    // 雅可比计算
    virtual void linearizeOplus() override { // 计算误差对两个顶点扰动的雅可比矩阵。
        SE3d v1 = (static_cast<VertexSE3LieAlgebra *> (_vertices[0]))->estimate(); // 取出第一个顶点估计，保留书中推导形式。
        SE3d v2 = (static_cast<VertexSE3LieAlgebra *> (_vertices[1]))->estimate(); // 取出第二个顶点估计，用于计算伴随矩阵。
        Matrix6d J = JRInv(SE3d::exp(_error)); // 根据当前误差计算右雅可比逆的近似。
        // 尝试把J近似为I？
        _jacobianOplusXi = -J * v2.inverse().Adj(); // 误差对第一个顶点更新量的雅可比。
        _jacobianOplusXj = J * v2.inverse().Adj(); // 误差对第二个顶点更新量的雅可比。
    }
};

int main(int argc, char **argv) { // 程序入口，命令行参数中需要传入一个 .g2o 文件。
    if (argc != 2) { // 检查参数个数，程序名之外必须恰好有一个输入文件路径。
        cout << "Usage: pose_graph_g2o_SE3_lie sphere.g2o" << endl; // 参数错误时打印使用方式。
        return 1; // 返回非零值表示程序异常结束。
    }
    ifstream fin(argv[1]); // 打开命令行传入的 .g2o 文件。
    if (!fin) { // 判断文件是否成功打开。
        cout << "file " << argv[1] << " does not exist." << endl; // 打印文件不存在或无法打开的信息。
        return 1; // 文件读取失败，结束程序。
    }

    // 设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType; // 定义块求解器类型：位姿维度 6，路标维度也设为 6。
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType; // 定义线性求解器类型，求解块求解器产生的线性系统。
    auto solver = new g2o::OptimizationAlgorithmLevenberg( // 创建 Levenberg-Marquardt 非线性优化算法。
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>())); // 将 Eigen 线性求解器装入块求解器，再交给 LM 算法。
    g2o::SparseOptimizer optimizer;     // 创建稀疏优化器，它保存整个图模型。
    optimizer.setAlgorithm(solver);   // 给优化器设置前面创建的求解算法。
    optimizer.setVerbose(true);       // 打开调试输出，优化时会打印每次迭代信息。

    int vertexCnt = 0, edgeCnt = 0; // 记录读入的顶点和边的数量。

    vector<VertexSE3LieAlgebra *> vectices; // 保存所有顶点指针，后续手动写出结果时使用。
    vector<EdgeSE3LieAlgebra *> edges; // 保存所有边指针，后续手动写出结果时使用。
    while (!fin.eof()) { // 循环读取文件，直到到达文件末尾。
        string name; // 保存当前记录的类型标签，例如 VERTEX_SE3:QUAT 或 EDGE_SE3:QUAT。
        fin >> name; // 从文件中读取一条记录的标签。
        if (name == "VERTEX_SE3:QUAT") { // 如果当前记录是 SE3 位姿顶点。
            // 顶点
            VertexSE3LieAlgebra *v = new VertexSE3LieAlgebra(); // 创建自定义李代数 SE3 顶点。
            int index = 0; // 保存顶点编号。
            fin >> index; // 读取顶点编号。
            v->setId(index); // 设置顶点在优化图中的 ID。
            v->read(fin); // 调用自定义读函数，读取并设置 Sophus::SE3d 估计。
            optimizer.addVertex(v); // 将顶点加入优化器。
            vertexCnt++; // 顶点计数加一。
            vectices.push_back(v); // 保存顶点指针，便于最后按 .g2o 格式写出。
            if (index == 0) // 通常固定第一个位姿，消除位姿图的规范自由度。
                v->setFixed(true); // 将第 0 个顶点固定，不参与优化更新。
        } else if (name == "EDGE_SE3:QUAT") { // 如果当前记录是两个 SE3 顶点之间的约束边。
            // SE3-SE3 边
            EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra(); // 创建自定义李代数 SE3 二元边。
            int idx1, idx2;     // 关联的两个顶点。
            fin >> idx1 >> idx2; // 读取这条边连接的两个顶点 ID。
            e->setId(edgeCnt++); // 设置边的 ID，并在设置后让边计数加一。
            e->setVertex(0, optimizer.vertices()[idx1]); // 设置边的第一个端点为 idx1 对应的顶点。
            e->setVertex(1, optimizer.vertices()[idx2]); // 设置边的第二个端点为 idx2 对应的顶点。
            e->read(fin); // 调用自定义读函数，读取边的相对位姿测量和信息矩阵。
            optimizer.addEdge(e); // 将边加入优化器。
            edges.push_back(e); // 保存边指针，便于最后按 .g2o 格式写出。
        }
        if (!fin.good()) break; // 如果读取状态异常或到达文件末尾，跳出循环。
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl; // 打印图中顶点和边的数量。

    cout << "optimizing ..." << endl; // 提示即将开始优化。
    optimizer.initializeOptimization(); // 初始化优化器内部数据结构。
    optimizer.optimize(30); // 执行最多 30 次迭代的位姿图优化。

    cout << "saving optimization results ..." << endl; // 提示即将保存优化结果。

    // 因为用了自定义顶点且没有向g2o注册，这里保存自己来实现
    // 伪装成 SE3 顶点和边，让 g2o_viewer 可以认出
    ofstream fout("result_lie.g2o"); // 打开输出文件，保存李代数版本的优化结果。
    for (VertexSE3LieAlgebra *v:vectices) { // 遍历所有顶点。
        fout << "VERTEX_SE3:QUAT "; // 写出 g2o 标准 SE3 顶点标签，方便 g2o_viewer 读取。
        v->write(fout); // 调用自定义写函数，写出当前顶点的优化后位姿。
    }
    for (EdgeSE3LieAlgebra *e:edges) { // 遍历所有边。
        fout << "EDGE_SE3:QUAT "; // 写出 g2o 标准 SE3 边标签，方便 g2o_viewer 读取。
        e->write(fout); // 调用自定义写函数，写出边的约束和信息矩阵。
    }
    fout.close(); // 关闭输出文件。
    return 0; // 程序正常结束。
}
