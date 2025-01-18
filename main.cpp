#include <iostream>
#include <Eigen/Dense>
#include "fft.h"

int main() {
    // 创建一个简单的输入向量 X
    Eigen::VectorXd X(4);
    X << 1.0, 2.0, 3.0, 4.0;

    // 调用FFT计算离散傅里叶变换
    Eigen::VectorXd Y = FFTLibrary::fft(X);

    // 打印结果
    std::cout << "Input vector X:" << std::endl;
    std::cout << X << std::endl;

    std::cout << "FFT result Y:" << std::endl;
    std::cout << Y << std::endl;

    return 0;
}
