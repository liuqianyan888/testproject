//
// Created by 86155 on 2025/1/18.
//

#include "fft.h"
#include <Eigen/Dense>
#include <complex>
#include <cmath>
#include <vector>

namespace FFTLibrary {

    // FFT 函数的实现
    Eigen::VectorXd fft(const Eigen::VectorXd& X) {
        int N = X.size();

        // 如果输入是空的，直接返回空的向量
        if (N == 0) return Eigen::VectorXd(N);

        // 定义复数类型向量来存储中间结果
        std::vector<std::complex<double>> X_complex(N);

        // 将输入向量X转化为复数形式
        for (int i = 0; i < N; ++i) {
            X_complex[i] = std::complex<double>(X(i), 0);
        }

        // 执行快速傅里叶变换（Cooley-Tukey算法）
        std::vector<std::complex<double>> Y_complex(N);

        // 计算FFT（递归实现或者迭代实现）
        for (int k = 0; k < N; ++k) {
            Y_complex[k] = 0;
            for (int n = 0; n < N; ++n) {
                double angle = 2.0 * M_PI * k * n / N;
                std::complex<double> w_n(cos(angle), -sin(angle));
                Y_complex[k] += X_complex[n] * w_n;
            }
        }

        // 将计算结果转回 Eigen::VectorXd
        Eigen::VectorXd Y(N);
        for (int i = 0; i < N; ++i) {
            Y(i) = Y_complex[i].real();  // 只取实部，假设返回的 FFT 是实数（没有虚部）
        }

        return Y;
    }

}
