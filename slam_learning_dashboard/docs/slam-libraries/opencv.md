# OpenCV：视觉前端与图像处理

## 对齐级别
- ：你的本地 `code` 大量覆盖。

## 1. 核心数据结构

1. 图像与矩阵：`cv::Mat`
2. 特征相关：
- `cv::KeyPoint`
- `cv::DMatch`
- `cv::Ptr<T>`
3. 几何与估计：
- `cv::Point2f`, `cv::Point3f`
- `cv::Rodrigues`
- `cv::solvePnP`
- `cv::findEssentialMat`, `cv::recoverPose`

## 2. 适用范围

OpenCV 是视觉前端核心工具箱：
- 图像读写与预处理
- 特征提取与匹配（ORB/FAST/BRIEF）
- 极几何估计（E/F/H）
- PnP、三角化、光流

你的本地典型文件：
- `code/ch7/orb_cv.cpp`
- `code/ch7/pose_estimation_2d2d.cpp`
- `code/ch8/optical_flow.cpp`

## 3. 典型 API 范例

### 3.1 ORB 特征提取与匹配

```cpp
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::Mat img1 = cv::imread("1.png", cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread("2.png", cv::IMREAD_COLOR);

    // 创建 ORB 检测器和描述子
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create();

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    orb->detectAndCompute(img1, cv::Mat(), kp1, desc1);
    orb->detectAndCompute(img2, cv::Mat(), kp2, desc2);

    // Hamming 距离匹配
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);
    return 0;
}
```

### 3.2 PnP 初值估计

```cpp
#include <opencv2/opencv.hpp>

int main() {
    std::vector<cv::Point3f> pts3d;  // 世界坐标系 3D 点
    std::vector<cv::Point2f> pts2d;  // 图像像素坐标

    cv::Mat K = (cv::Mat_<double>(3,3) <<
        520.9, 0, 325.1,
        0, 521.0, 249.7,
        0, 0, 1);

    cv::Mat rvec, tvec;
    // 根据 3D-2D 对应关系估计位姿
    cv::solvePnP(pts3d, pts2d, K, cv::Mat(), rvec, tvec, false);

    // 旋转向量转旋转矩阵
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    return 0;
}
```

## 4. 常见坑与建议

1. 注意 `cv::Mat` 元素类型（`CV_8U`, `CV_32F`, `CV_64F`）与 `at<T>` 对应。
2. 深度图常是 `uint16`，需要按尺度因子还原米制深度。
3. OpenCV 与 Eigen 互转时统一坐标系与单位，避免“看起来能跑但结果偏移”。
