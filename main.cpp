// main.cpp
#include <iostream>
#include "fft.h"

int main() {
    // 示例：输入一个简单的矩阵
    Eigen::MatrixXd X(4, 2);  // 4行2列的矩阵
    X << 1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0;

    int n = 8;

    std::cout << "Input Matrix X:\n" << X << "\n\n";

    // 计算X的FFT
    Eigen::MatrixXcd Y = FFTLibrary::fft_full(X);

    // 调用fft函数计算结果
    Eigen::MatrixXcd Y_2 = FFTLibrary_v2::fft(X, n);

    std::cout << "FFT Output Y:\n" << Y << std::endl;

    std::cout << "FFT Output Y_2:" << std::endl;
    std::cout << Y_2 << std::endl;

    // 计算X的FFT，沿着列（dim=1）进行傅里叶变换
    Eigen::MatrixXcd Y_3 = FFTLibrary_v3::fft_along_dim(X, n, 1);
    std::cout << "FFT along columns (dim=1):\n" << Y_3 << std::endl;

    // 计算X的FFT，沿着行（dim=2）进行傅里叶变换
    Eigen::MatrixXcd Y_4 = FFTLibrary_v3::fft_along_dim(X, n, 2);
    std::cout << "FFT along rows (dim=2):\n" << Y_4 << std::endl;

    return 0;
}
