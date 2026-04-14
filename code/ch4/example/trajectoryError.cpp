#include <iostream>
#include <fstream>
#include <unistd.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>

using namespace Sophus;
using namespace std;

int main(){
    string groundtruth_file = "./groundtruth.txt";
    string estimated_file = "./estimated.txt";

    typedef vector<Sophus::SE3d,Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;

    return 0;
}