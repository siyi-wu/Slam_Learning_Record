#include <iostream>
using namespace std;

#include <ctime>

#include <Eigen/Core>

#include <Eigen/Dense>
using namespace Eigen;

#define MATRIX_SIZE 50

int main(int argc,char **argv){
    //其实是Eigen::Matrix
    Matrix<float,2,3> matrix_23;
    
    Vector3d v_3d;
    //等同于：
    Matrix<float,3,1> vd_3d;

    //Matrix3d：Eigen::Matrix<double,3,3>
    Matrix3d matrix_33=Matrix3d::Zero();//初始化为0
    Matrix<double,Dynamic,Dynamic>
 matrix_dynamic;//可动态变化矩阵
    MatrixXd matrix_x;//可动态变化矩阵

    //输入数据
    matrix_23 << 1,2,3,4,5,6;

    //输出
    cout<<"matrix 2x3 from 1 to 6:\n"<<matrix_23<<endl;

    //访问矩阵中的元素
    cout<<"print matrix 2x3:"<<endl;
    for(int i=0;i<2;i++){
        for(int j=0;j<3;j++){
            cout<<matrix_23(i,j)<<"\t";
        }
        cout<<endl;
    }

    //矩阵与向量相乘/矩阵相乘
    v_3d<<3,2,1;//Vector3d
    vd_3d<<4,5,6;//Matrix<float,3,1>

    //不能混合不同类型矩阵
    //比如matrix_23和v_3d相乘
    //应该做一次转换
    Matrix<double,2,1> result=matrix_23.cast<double>()*v_3d;
    cout<<"[1,2,3;4,5,6]*[3,2,1]^T:"<<result.transpose()<<endl;

    Matrix<float,2,1> result2=matrix_23*vd_3d;
    cout<<"[1,2,3;4,5,6]*[4,5,6]^T:"<<result2.transpose()<<endl;

    matrix_33=Matrix3d::Random();
    cout<<"random matrix:\n"<<matrix_33<<endl;
    cout<<"transpose:\n"<<matrix_33.transpose()<<endl;//转置
    cout<<"sum: "<<matrix_33.sum()<<endl;//求和
    cout<<"trace: "<<matrix_33.trace()<<endl;//迹
    cout<<"inverse:\n"<<matrix_33.inverse()<<endl;//逆
    cout<<"det: "<<matrix_33.determinant()<<endl;//行列式

    //解方程
    //求解 matrix_NN * x= v_Nd 这个方程
    //直接求逆计算量大
    Matrix<double,MATRIX_SIZE,MATRIX_SIZE> matrix_NN=MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
    matrix_NN=matrix_NN*matrix_NN.transpose();//保证半正定
    Matrix<double,MATRIX_SIZE,1> v_Nd=MatrixXd::Random(MATRIX_SIZE,1);

    clock_t time_stt=clock();//计时
    //直接求逆
    Matrix<double,MATRIX_SIZE,1> x=matrix_NN.inverse()*v_Nd;
    cout<<"time of normal inverse is "
        <<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
    cout<<"x= "<<x.transpose()<<endl;

    //矩阵分解：正交矩阵乘上三角矩阵
    time_stt=clock();
    x=matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout<<"time of Qr decompositipon is "
        <<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
    cout<<"x= "<<x.transpose()<<endl;

    //正定矩阵还可以用cholesky分解来求解方程：cholesky将一个对称正定矩阵分解为一个狭三角矩阵及其转置的乘积；ldlt分解将其分解为ldl^T
    time_stt=clock();
    x=matrix_NN.ldlt().solve(v_Nd);
    cout<<"time of ldlt decomposition is "
        <<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
    cout<<"x= "<<x.transpose()<<endl;


    return 0;
}