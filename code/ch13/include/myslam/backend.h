//
// Created by gaoxiang on 19-5-2.
//

#ifndef MYSLAM_BACKEND_H  // 头文件保护：如果没有定义过该宏，才继续展开本头文件
#define MYSLAM_BACKEND_H  // 定义头文件保护宏，避免重复包含

#include "myslam/common_include.h"  // 项目通用头文件，包含常用类型、Eigen/Sophus/STL 等
#include "myslam/frame.h"           // Frame 定义，后端需要优化关键帧位姿
#include "myslam/map.h"             // Map 定义，后端从地图中取活跃关键帧和地图点

namespace myslam {  // 进入 myslam 命名空间，避免和其他库命名冲突
class Map;          // 前向声明 Map，告诉编译器后面会有这个类

/**
 * 后端
 * 有单独优化线程，在Map更新时启动优化
 * Map更新由前端触发
 */
class Backend {  // 后端类，负责在独立线程中做局部 BA 优化
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;      // 让含 Eigen 类型的对象动态分配时满足内存对齐要求
    typedef std::shared_ptr<Backend> Ptr; // Backend 智能指针别名，方便统一管理生命周期

    /// 构造函数中启动优化线程并挂起
    Backend();  // 创建后端对象时启动后台优化线程

    // 设置左右目的相机，用于获得内外参
    void SetCameras(Camera::Ptr left, Camera::Ptr right) {  // 保存双目相机模型
        cam_left_ = left;                                   // 左相机指针，用于内参和左相机外参
        cam_right_ = right;                                 // 右相机指针，用于右相机外参
    }

    /// 设置地图
    void SetMap(std::shared_ptr<Map> map) { map_ = map; }  // 保存地图指针，后端从这里取优化数据

    /// 触发地图更新，启动优化
    void UpdateMap();  // 前端插入关键帧/地图点后调用，用条件变量唤醒后端线程

    /// 关闭后端线程
    void Stop();  // 停止后台循环，并等待线程退出

   private:
    /// 后端线程
    void BackendLoop();  // 后端线程主循环，等待地图更新并执行优化

    /// 对给定关键帧和路标点进行优化
    void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);  // 对活跃关键帧和活跃地图点做局部 BA

    std::shared_ptr<Map> map_;  // 地图对象，保存关键帧和地图点
    std::thread backend_thread_; // 后端优化线程
    std::mutex data_mutex_;      // 保护后端线程等待/唤醒相关数据

    std::condition_variable map_update_;  // 条件变量，用于通知后端“地图更新了，可以优化”
    std::atomic<bool> backend_running_;   // 原子布尔量，控制后端线程是否继续运行

    Camera::Ptr cam_left_ = nullptr, cam_right_ = nullptr;  // 左右相机模型
};

}  // namespace myslam  // 结束 myslam 命名空间

#endif  // MYSLAM_BACKEND_H  // 结束头文件保护
