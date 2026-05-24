//
// Created by gaoxiang on 19-5-2.
//

#include "myslam/backend.h"   // Backend 类声明
#include "myslam/algorithm.h" // 通用算法函数，后端可能依赖项目算法工具
#include "myslam/feature.h"   // Feature 定义，后端需要访问特征观测
#include "myslam/g2o_types.h" // g2o 顶点和边类型定义
#include "myslam/map.h"       // Map 定义，后端从地图中取活跃数据
#include "myslam/mappoint.h"  // MapPoint 定义，后端优化地图点位置

namespace myslam {  // 进入 myslam 命名空间

Backend::Backend() {  // 后端构造函数
    backend_running_.store(true);  // 设置线程运行标志为 true
    backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));  // 启动后端线程，执行 BackendLoop
}

void Backend::UpdateMap() {  // 前端地图更新后调用此函数
    std::unique_lock<std::mutex> lock(data_mutex_);  // 加锁，配合条件变量使用
    map_update_.notify_one();                        // 唤醒一个正在等待的后端线程
}

void Backend::Stop() {  // 停止后端线程
    backend_running_.store(false);  // 将运行标志置为 false，让后端循环退出
    map_update_.notify_one();       // 唤醒线程，避免它一直阻塞在 wait 中
    backend_thread_.join();         // 等待后端线程结束，回收线程资源
}

void Backend::BackendLoop() {  // 后端线程主循环
    while (backend_running_.load()) {  // 只要运行标志为 true，就持续等待地图更新
        std::unique_lock<std::mutex> lock(data_mutex_);  // 加锁，供条件变量 wait 使用
        map_update_.wait(lock);                          // 等待前端通过 UpdateMap 发出通知

        /// 后端仅优化激活的Frames和Landmarks
        Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();       // 获取活跃关键帧的拷贝
        Map::LandmarksType active_landmarks = map_->GetActiveMapPoints(); // 获取活跃地图点的拷贝
        Optimize(active_kfs, active_landmarks);                           // 对局部地图执行 BA 优化
    }
}

void Backend::Optimize(Map::KeyframesType &keyframes,
                       Map::LandmarksType &landmarks) {  // 优化输入的关键帧位姿和地图点位置
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;  // 定义块求解器：位姿 6 维，路标点 3 维
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>
        LinearSolverType;  // 使用 CSparse 线性求解器，适合稀疏 BA 问题
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(
            std::make_unique<LinearSolverType>()));  // 创建 Levenberg-Marquardt 优化算法
    g2o::SparseOptimizer optimizer;                   // 创建 g2o 稀疏优化器
    optimizer.setAlgorithm(solver);                   // 给优化器设置求解算法

    // pose 顶点，使用Keyframe id
    std::map<unsigned long, VertexPose *> vertices;  // 保存 keyframe_id 到位姿顶点的映射
    unsigned long max_kf_id = 0;                     // 记录最大关键帧 id，用于给地图点顶点错开编号
    for (auto &keyframe : keyframes) {               // 遍历所有待优化关键帧
        auto kf = keyframe.second;                   // 取出关键帧指针
        VertexPose *vertex_pose = new VertexPose();  // 创建相机位姿顶点
        vertex_pose->setId(kf->keyframe_id_);        // 使用关键帧 id 作为 g2o 顶点 id
        vertex_pose->setEstimate(kf->Pose());        // 使用当前关键帧位姿作为初始估计
        optimizer.addVertex(vertex_pose);            // 将位姿顶点加入优化器
        if (kf->keyframe_id_ > max_kf_id) {          // 如果当前 id 更大
            max_kf_id = kf->keyframe_id_;            // 更新最大关键帧 id
        }

        vertices.insert({kf->keyframe_id_, vertex_pose});  // 记录该关键帧对应的位姿顶点
    }

    // 路标顶点，使用路标id索引
    std::map<unsigned long, VertexXYZ *> vertices_landmarks;  // 保存地图点 id 到三维点顶点的映射

    // K 和左右外参
    Mat33 K = cam_left_->K();          // 左相机内参矩阵，左右相机通常共用同一套内参
    SE3 left_ext = cam_left_->pose();  // 左相机相对于双目相机/机体坐标的外参
    SE3 right_ext = cam_right_->pose();// 右相机相对于双目相机/机体坐标的外参

    // edges
    int index = 1;                  // g2o 边 id，从 1 开始递增
    double chi2_th = 5.991;         // robust kernel 阈值，也是 2D 重投影误差的卡方阈值
    std::map<EdgeProjection *, Feature::Ptr> edges_and_features;  // 记录每条边对应哪个特征，便于后面剔除外点

    for (auto &landmark : landmarks) {                         // 遍历所有活跃地图点
        if (landmark.second->is_outlier_) continue;            // 如果地图点已被标记为外点，则跳过
        unsigned long landmark_id = landmark.second->id_;      // 取出地图点 id
        auto observations = landmark.second->GetObs();         // 获取所有观测到该地图点的 Feature
        for (auto &obs : observations) {                       // 遍历这些观测
            if (obs.lock() == nullptr) continue;               // weak_ptr 失效说明 Feature 已不存在，跳过
            auto feat = obs.lock();                            // 将 weak_ptr 转成 shared_ptr 使用
            if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;  // 特征是外点或所属帧失效则跳过

            auto frame = feat->frame_.lock();  // 取出观测该地图点的关键帧
            EdgeProjection *edge = nullptr;    // 准备创建一条重投影误差边
            if (feat->is_on_left_image_) {     // 如果该 Feature 来自左图
                edge = new EdgeProjection(K, left_ext);  // 使用左相机外参构造投影边
            } else {
                edge = new EdgeProjection(K, right_ext); // 使用右相机外参构造投影边
            }

            // 如果landmark还没有被加入优化，则新加一个顶点
            if (vertices_landmarks.find(landmark_id) ==
                vertices_landmarks.end()) {              // 如果该地图点还没有对应的 g2o 顶点
                VertexXYZ *v = new VertexXYZ;            // 创建三维路标点顶点
                v->setEstimate(landmark.second->Pos());  // 使用当前地图点位置作为初始估计
                v->setId(landmark_id + max_kf_id + 1);   // 设置顶点 id，避开关键帧顶点 id
                v->setMarginalized(true);                // BA 中路标点通常设为边缘化变量
                vertices_landmarks.insert({landmark_id, v});  // 记录地图点 id 与顶点的对应关系
                optimizer.addVertex(v);                  // 将地图点顶点加入优化器
            }


            if (vertices.find(frame->keyframe_id_) !=
                vertices.end() &&                         // 确认该观测所属关键帧在本次优化窗口内
                vertices_landmarks.find(landmark_id) !=
                vertices_landmarks.end()) {              // 确认该地图点顶点已经创建
                    edge->setId(index);                  // 设置边 id
                    edge->setVertex(0, vertices.at(frame->keyframe_id_));    // 连接第 0 个顶点：相机位姿
                    edge->setVertex(1, vertices_landmarks.at(landmark_id));  // 连接第 1 个顶点：地图点
                    edge->setMeasurement(toVec2(feat->position_.pt));        // 设置观测值：图像上的 2D 特征点坐标
                    edge->setInformation(Mat22::Identity());                 // 设置信息矩阵：默认 x/y 方向权重为 1
                    auto rk = new g2o::RobustKernelHuber();                  // 创建 Huber 鲁棒核，降低外点影响
                    rk->setDelta(chi2_th);                                   // 设置鲁棒核阈值
                    edge->setRobustKernel(rk);                               // 将鲁棒核绑定到边上
                    edges_and_features.insert({edge, feat});                 // 记录边和 Feature 的对应关系
                    optimizer.addEdge(edge);                                 // 将重投影边加入优化器
                    index++;                                                 // 边 id 递增
                }
            else delete edge;  // 如果关键帧或地图点顶点缺失，则删除刚创建的边，避免内存泄漏
	                
        }
    }

    // do optimization and eliminate the outliers
    optimizer.initializeOptimization();  // 初始化优化问题
    optimizer.optimize(10);              // 执行 10 次迭代优化

    int cnt_outlier = 0, cnt_inlier = 0;  // 统计外点和内点数量
    int iteration = 0;                    // 调整外点阈值的次数
    while (iteration < 5) {               // 最多调整 5 次阈值
        cnt_outlier = 0;                  // 每轮重新统计外点
        cnt_inlier = 0;                   // 每轮重新统计内点
        // determine if we want to adjust the outlier threshold
        for (auto &ef : edges_and_features) {  // 遍历所有投影边
            if (ef.first->chi2() > chi2_th) {  // 如果该边重投影误差超过阈值
                cnt_outlier++;                 // 统计为外点
            } else {
                cnt_inlier++;                  // 否则统计为内点
            }
        }
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);  // 计算内点比例
        if (inlier_ratio > 0.5) {  // 如果内点比例超过一半，认为阈值合理
            break;                 // 停止调整阈值
        } else {
            chi2_th *= 2;          // 如果外点太多，放宽阈值
            iteration++;           // 调整次数加一
        }
    }

    for (auto &ef : edges_and_features) {       // 再次遍历边和特征
        if (ef.first->chi2() > chi2_th) {       // 如果最终误差仍然过大
            ef.second->is_outlier_ = true;      // 将对应 Feature 标记为外点
            // remove the observation
            ef.second->map_point_.lock()->RemoveObservation(ef.second);  // 从地图点观测列表中移除该 Feature
        } else {
            ef.second->is_outlier_ = false;     // 误差正常，标记为内点
        }
    }

    LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
              << cnt_inlier;  // 打印外点/内点数量

    // Set pose and lanrmark position
    for (auto &v : vertices) {                         // 遍历所有优化后的位姿顶点
        keyframes.at(v.first)->SetPose(v.second->estimate());  // 将优化后的位姿写回关键帧
    }
    for (auto &v : vertices_landmarks) {               // 遍历所有优化后的地图点顶点
        landmarks.at(v.first)->SetPos(v.second->estimate());   // 将优化后的三维位置写回地图点
    }
}

}  // namespace myslam  // 结束 myslam 命名空间
