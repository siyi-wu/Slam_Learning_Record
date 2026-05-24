#include <iostream> // 标准输入输出，用于打印运行信息。
#include <fstream>  // 文件流，用于读取 .g2o 文件。
#include <string>   // 字符串类型，用于读取每一行记录的标签。
#include <memory>   // 智能指针工具，供 make_unique 创建求解器组件。

#include <g2o/types/slam3d/types_slam3d.h>              // g2o 内置的 SE3 顶点和边类型。
#include <g2o/core/block_solver.h>                      // 块求解器，用于构造位姿图优化问题。
#include <g2o/core/optimization_algorithm_levenberg.h>   // Levenberg-Marquardt 优化算法。
#include <g2o/solvers/eigen/linear_solver_eigen.h>       // 基于 Eigen 的线性方程求解器。

using namespace std; // 直接使用 std 命名空间中的 cout、ifstream、string 等。

/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 这里使用g2o/types/slam3d/中的SE3表示位姿，它实质上是四元数而非李代数.
 * **********************************************/

int main(int argc, char **argv) { // 程序入口，命令行参数中需要传入一个 .g2o 文件。
    if (argc != 2) { // 检查参数个数，程序名之外必须恰好有一个输入文件路径。
        cout << "Usage: pose_graph_g2o_SE3 sphere.g2o" << endl; // 参数错误时打印使用方式。
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
    while (!fin.eof()) { // 循环读取文件，直到到达文件末尾。
        string name; // 保存当前记录的类型标签，例如 VERTEX_SE3:QUAT 或 EDGE_SE3:QUAT。
        fin >> name; // 从文件中读取一条记录的标签。
        if (name == "VERTEX_SE3:QUAT") { // 如果当前记录是 SE3 位姿顶点。
            // SE3 顶点
            g2o::VertexSE3 *v = new g2o::VertexSE3(); // 创建 g2o 内置 SE3 顶点。
            int index = 0; // 保存顶点编号。
            fin >> index; // 读取顶点编号。
            v->setId(index); // 设置顶点在优化图中的 ID。
            v->read(fin); // 调用 g2o 内置读函数，读取平移和四元数。
            optimizer.addVertex(v); // 将顶点加入优化器。
            vertexCnt++; // 顶点计数加一。
            if (index == 0) // 通常固定第一个位姿，消除位姿图的规范自由度。
                v->setFixed(true); // 将第 0 个顶点固定，不参与优化更新。
        } else if (name == "EDGE_SE3:QUAT") { // 如果当前记录是两个 SE3 顶点之间的约束边。
            // SE3-SE3 边
            g2o::EdgeSE3 *e = new g2o::EdgeSE3(); // 创建 g2o 内置 SE3 边。
            int idx1, idx2;     // 关联的两个顶点。
            fin >> idx1 >> idx2; // 读取这条边连接的两个顶点 ID。
            e->setId(edgeCnt++); // 设置边的 ID，并在设置后让边计数加一。
            e->setVertex(0, optimizer.vertices()[idx1]); // 设置边的第一个端点为 idx1 对应的顶点。
            e->setVertex(1, optimizer.vertices()[idx2]); // 设置边的第二个端点为 idx2 对应的顶点。
            e->read(fin); // 调用 g2o 内置读函数，读取边的测量值和信息矩阵。
            optimizer.addEdge(e); // 将边加入优化器。
        }
        if (!fin.good()) break; // 如果读取状态异常或到达文件末尾，跳出循环。
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl; // 打印图中顶点和边的数量。

    cout << "optimizing ..." << endl; // 提示即将开始优化。
    optimizer.initializeOptimization(); // 初始化优化器内部数据结构。
    optimizer.optimize(30); // 执行最多 30 次迭代的位姿图优化。

    cout << "saving optimization results ..." << endl; // 提示即将保存优化结果。
    optimizer.save("result.g2o"); // 使用 g2o 内置保存函数，把优化后的图保存到 result.g2o。

    return 0; // 程序正常结束。
}
