// fft.h
#ifndef FFTLIBRARY_H
#define FFTLIBRARY_H

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <iostream>

namespace FFTLibrary {

// 函数声明：计算矩阵X的离散傅里叶变换
    Eigen::MatrixXcd fft_full(const Eigen::MatrixXd &X);

} // namespace FFTLibrary


namespace FFTLibrary_v2 {

    // 计算n点傅里叶变换，支持向量和矩阵
    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X, int n);  // 保持为 fft


}
namespace FFTLibrary_v3 {

    // 沿指定维度进行傅里叶变换
    Eigen::MatrixXcd fft_along_dim(const Eigen::MatrixXd& X, int n, int dim);
}

#endif // FFTLIBRARY_H

