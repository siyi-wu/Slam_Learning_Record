#include <iostream>                         // 引入标准输入输出流，用于打印提示和优化信息
#include <ceres/ceres.h>                    // 引入 Ceres Solver 主头文件，用于构建和求解非线性最小二乘问题
#include "common.h"                         // 引入 BALProblem 等通用数据结构和工具函数
#include "SnavelyReprojectionError.h"       // 引入 Snavely 重投影误差模型，用于构造 BA 残差

using namespace std;                        // 使用 std 命名空间，简化 cout、endl 等标准库符号的书写

void SolveBA(BALProblem &bal_problem);      // 声明 BA 求解函数，输入为一个 BAL 数据问题对象

int main(int argc, char **argv) {           // 程序入口，argc 为命令行参数数量，argv 为参数字符串数组
    if (argc != 2) {                        // 检查是否只传入了一个 BAL 数据文件路径
        cout << "usage: bundle_adjustment_ceres bal_data.txt" << endl; // 参数数量错误时打印使用方法
        return 1;                           // 返回非零值，表示程序异常结束
    }

    BALProblem bal_problem(argv[1]);        // 从命令行指定的 BAL 数据文件中读取相机、路标点和观测数据
    bal_problem.Normalize();                // 对问题数据做归一化，改善数值尺度，利于优化稳定收敛
    bal_problem.Perturb(0.1, 0.5, 0.5);     // 给相机和路标点参数加入扰动，模拟带噪声的初始值
    bal_problem.WriteToPLYFile("initial.ply"); // 将优化前的相机和三维点写入 PLY 文件，便于可视化初始状态
    SolveBA(bal_problem);                   // 调用 Ceres 对当前 BAL 问题执行 Bundle Adjustment 优化
    bal_problem.WriteToPLYFile("final.ply"); // 将优化后的相机和三维点写入 PLY 文件，便于对比优化结果

    return 0;                               // 返回 0，表示程序正常结束
}

void SolveBA(BALProblem &bal_problem) {     // 定义 BA 求解函数，直接在 bal_problem 内部参数上进行优化
    const int point_block_size = bal_problem.point_block_size();   // 每个三维点参数块的大小，通常为 3 维坐标
    const int camera_block_size = bal_problem.camera_block_size(); // 每个相机参数块的大小，Snavely 模型中通常为 9 维
    double *points = bal_problem.mutable_points();                 // 获取可修改的三维点参数数组首地址
    double *cameras = bal_problem.mutable_cameras();               // 获取可修改的相机参数数组首地址

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x
    // and y position of the observation.
    const double *observations = bal_problem.observations();       // 获取观测数组，每个观测包含图像坐标 x、y 两个值
    ceres::Problem problem;                                        // 创建 Ceres 优化问题对象，用于存放残差块和参数块

    for (int i = 0; i < bal_problem.num_observations(); ++i) {     // 遍历所有二维观测，为每条观测构造一个重投影残差
        ceres::CostFunction *cost_function;                        // 声明代价函数指针，表示当前观测对应的误差项

        // Each Residual block takes a point and a camera as input
        // and outputs a 2 dimensional Residual
        cost_function = SnavelyReprojectionError::Create(          // 使用当前观测的像素坐标创建 Snavely 重投影误差代价函数
            observations[2 * i + 0],                               // 当前观测的 x 坐标，也就是第 i 个观测的第一个分量
            observations[2 * i + 1]);                              // 当前观测的 y 坐标，也就是第 i 个观测的第二个分量

        // If enabled use Huber's loss function.
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0); // 使用 Huber 鲁棒核函数，降低异常观测对优化的影响

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i]; // 根据观测对应的相机索引定位相机参数块
        double *point = points + point_block_size * bal_problem.point_index()[i];     // 根据观测对应的路标点索引定位三维点参数块

        problem.AddResidualBlock(cost_function, loss_function, camera, point); // 向 Ceres 问题中添加残差块，优化变量为该相机和该三维点
    }

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;        // 打印提示：BAL 问题文件已经加载完成
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and " // 打印相机数量
              << bal_problem.num_points() << " points. " << std::endl;              // 打印三维点数量
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl; // 打印观测数量

    std::cout << "Solving ceres BA ... " << endl;                  // 打印提示：开始使用 Ceres 求解 BA
    ceres::Solver::Options options;                                // 创建求解器配置对象
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR; // 使用稀疏 Schur 补线性求解器，适合 BA 的相机-点稀疏结构
    options.minimizer_progress_to_stdout = true;                   // 将每轮优化迭代信息输出到终端
    ceres::Solver::Summary summary;                                // 创建求解摘要对象，用于接收优化过程和结果报告
    ceres::Solve(options, &problem, &summary);                     // 调用 Ceres 求解器，根据 options 优化 problem，并把结果写入 summary
    std::cout << summary.FullReport() << "\n";                     // 打印完整优化报告，包括代价变化、迭代次数和收敛状态
}
