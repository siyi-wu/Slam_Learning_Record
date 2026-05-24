#ifndef SnavelyReprojection_H                          // 头文件保护宏：如果还没有定义该宏，则继续编译本文件内容
#define SnavelyReprojection_H                          // 定义头文件保护宏，防止该头文件被重复包含

#include <iostream>                                    // 引入标准输入输出流头文件，本文件中保留该依赖
#include "ceres/ceres.h"                               // 引入 Ceres Solver 头文件，用于 AutoDiffCostFunction 和 CostFunction
#include "rotation.h"                                  // 引入旋转相关函数，例如 AngleAxisRotatePoint

class SnavelyReprojectionError {                        // 定义 Snavely 相机模型下的重投影误差类
public:
    SnavelyReprojectionError(double observation_x, double observation_y) // 构造函数，接收当前观测点的二维图像坐标
        : observed_x(observation_x),                       // 初始化观测点的 x 坐标
          observed_y(observation_y) {}                      // 初始化观测点的 y 坐标

    template<typename T>                                  // 模板类型 T 由 Ceres 自动微分系统决定，可为 double 或 Jet 类型
    bool operator()(const T *const camera,                // 函数调用运算符：输入 9 维相机参数块
                    const T *const point,                 // 输入 3 维三维点参数块
                    T *residuals) const {                 // 输出 2 维重投影残差数组
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];                                // 存放由相机参数和三维点投影得到的二维预测坐标
        CamProjectionWithDistortion(camera, point, predictions); // 计算带径向畸变的相机投影结果
        residuals[0] = predictions[0] - T(observed_x);   // x 方向残差 = 预测 x 坐标 - 实际观测 x 坐标
        residuals[1] = predictions[1] - T(observed_y);   // y 方向残差 = 预测 y 坐标 - 实际观测 y 坐标

        return true;                                     // 返回 true 表示残差计算成功
    }

    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T>                                  // 模板函数，使该投影过程兼容 Ceres 自动微分
    static inline bool CamProjectionWithDistortion(const T *camera, // 输入 9 维相机参数数组
                                                   const T *point,  // 输入 3 维世界坐标点
                                                   T *predictions) { // 输出 2 维图像平面预测坐标
        // Rodrigues' formula
        T p[3];                                          // 存放三维点经过旋转后的相机坐标中间结果
        AngleAxisRotatePoint(camera, point, p);          // 使用角轴旋转向量 camera[0..2] 将 point 旋转到相机坐标系方向
        // camera[3,4,5] are the translation
        p[0] += camera[3];                               // 加上 x 方向平移，得到相机坐标系下的 x 坐标
        p[1] += camera[4];                               // 加上 y 方向平移，得到相机坐标系下的 y 坐标
        p[2] += camera[5];                               // 加上 z 方向平移，得到相机坐标系下的 z 坐标

        // Compute the center fo distortion
        T xp = -p[0] / p[2];                             // 归一化成像平面 x 坐标，负号来自 BAL/Snavely 相机坐标约定
        T yp = -p[1] / p[2];                             // 归一化成像平面 y 坐标，负号来自 BAL/Snavely 相机坐标约定

        // Apply second and fourth order radial distortion
        const T &l1 = camera[7];                         // 二阶径向畸变系数
        const T &l2 = camera[8];                         // 四阶径向畸变系数

        T r2 = xp * xp + yp * yp;                        // 计算归一化平面点到中心的半径平方
        T distortion = T(1.0) + r2 * (l1 + l2 * r2);     // 计算径向畸变因子：1 + l1*r^2 + l2*r^4

        const T &focal = camera[6];                      // 相机焦距参数
        predictions[0] = focal * distortion * xp;        // 计算最终预测图像 x 坐标：焦距 * 畸变因子 * 归一化 x
        predictions[1] = focal * distortion * yp;        // 计算最终预测图像 y 坐标：焦距 * 畸变因子 * 归一化 y

        return true;                                     // 返回 true 表示投影计算成功
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y) { // 工厂函数：根据观测坐标创建 Ceres 代价函数
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>( // 创建自动微分代价函数：2 维残差、9 维相机、3 维点
            new SnavelyReprojectionError(observed_x, observed_y))); // 创建误差对象，并将观测坐标保存到对象内部
    }

private:
    double observed_x;                                   // 当前观测点的 x 坐标
    double observed_y;                                   // 当前观测点的 y 坐标
};

#endif // SnavelyReprojection.h                          // 结束头文件保护宏范围
