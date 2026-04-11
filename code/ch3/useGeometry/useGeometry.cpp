#include <iostream>
#include <cmath>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

int main(int argc,char **argv){

    //3D旋转矩阵
    Matrix3d rotation_matrix=Matrix3d::Identity();
    //旋转向量
    AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1).normalized());     //沿 Z 轴旋转 45 度，这个轴必须归一化

    cout.precision(3);
    cout << "rotation matrix =\n" << rotation_vector.matrix() << endl;   //用matrix()转换成矩阵

    //或者直接赋值
    rotation_matrix = rotation_vector.toRotationMatrix();

    Vector3d v(1, 0, 0);

    //用angleaxis进行坐标变换
    Vector3d v_rotated = rotation_vector * v;
    cout << "(1,0,0) after rotation (by angle axis) = " << v_rotated.transpose() << endl;

    // 或者用旋转矩阵
    v_rotated = rotation_matrix * v;
    cout << "(1,0,0) after rotation (by matrix) = " << v_rotated.transpose() << endl;

    // 欧拉角: 可以将旋转矩阵直接转换成欧拉角
    Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // ZYX顺序，即yaw-pitch-roll顺序
    cout << "yaw pitch roll = " << euler_angles.transpose() << endl;

    // 欧氏变换矩阵使用 Eigen::Isometry
    Isometry3d T = Isometry3d::Identity();                // 虽然称为3d，实质上是4＊4的矩阵
    T.rotate(rotation_vector);                                     // 按照rotation_vector进行旋转
    T.pretranslate(Vector3d(1, 3, 4));                     // 把平移向量设成(1,3,4)
    cout << "Transform matrix = \n" << T.matrix() << endl;

    // 用变换矩阵进行坐标变换
    Vector3d v_transformed = T * v;                              // 相当于R*v+t
    cout << "v tranformed = " << v_transformed.transpose() << endl;

    return 0;
}