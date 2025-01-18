//
// Created by 86155 on 2025/1/18.
//

#ifndef FFT_H
#define FFT_H

#include <Eigen/Dense>

namespace FFTLibrary {
    // 函数声明：计算X的傅里叶变换
    Eigen::VectorXd fft(const Eigen::VectorXd& X);  // X 是输入，Y 是输出，大小相同
}

#endif // FFT_H
