// fft.cpp
#include "fft.h"

namespace FFTLibrary {

// 计算矩阵X的离散傅里叶变换
    Eigen::MatrixXcd fft_full(const Eigen::MatrixXd &X) {
        // 创建一个和输入矩阵相同大小的复数矩阵用于存储结果
        Eigen::MatrixXcd Y(X.rows(), X.cols());

        // 对矩阵每一列进行FFT变换
        for (int col = 0; col < X.cols(); ++col) {
            Eigen::VectorXd X_col = X.col(col); // 获取当前列
            Eigen::VectorXcd Y_col = Eigen::VectorXcd::Zero(X_col.size()); // 存储FFT结果
            Eigen::FFT<double> fft;  // 使用Eigen的FFT

            fft.fwd(Y_col, X_col); // 对列进行快速傅里叶变换
            Y.col(col) = Y_col;    // 将结果存储到输出矩阵
        }

        return Y;
    }

} // namespace FFTLibrary

namespace FFTLibrary_v2 {

    // 计算 n 点离散傅里叶变换（DFT）
    Eigen::MatrixXcd fft(const Eigen::MatrixXd& X, int n) {
        // 创建结果矩阵，确保结果的行数为 n
        Eigen::MatrixXcd result(n, X.cols());
        Eigen::FFT<double> fft;  // 使用 Eigen 库的 FFT

        // 如果 X 是矩阵，每列独立进行傅里叶变换
        if (X.rows() > 1) {
            for (int col = 0; col < X.cols(); ++col) {
                Eigen::VectorXd column = X.col(col);

                // 如果列的长度小于 n，补零；如果大于 n，截断
                if (column.size() < n) {
                    column.conservativeResize(n);
                    column.tail(n - column.size()).setZero();  // 填充零
                } else if (column.size() > n) {
                    column.conservativeResize(n);  // 截断
                }

                // 进行傅里叶变换
                Eigen::VectorXcd fft_result(n);  // 确保结果向量大小为 n
                fft.fwd(fft_result, column);
                result.col(col) = fft_result;  // 将傅里叶变换结果赋值给 result
            }
        }
            // 如果 X 是向量（单行或单列）
        else {
            Eigen::VectorXd row = X.row(0);

            // 如果行的长度小于 n，补零；如果大于 n，截断
            if (row.size() < n) {
                row.conservativeResize(n);
                row.tail(n - row.size()).setZero();  // 填充零
            } else if (row.size() > n) {
                row.conservativeResize(n);  // 截断
            }

            // 进行傅里叶变换
            Eigen::VectorXcd fft_result(n);  // 确保结果向量大小为 n
            fft.fwd(fft_result, row);
            result = fft_result.transpose();  // 转置结果以匹配矩阵格式
        }

        return result;
    }

}  // namespace FFTLibrary_v2

namespace FFTLibrary_v3 {

    // 新增的fft_along_dim(X, n, dim)实现
    Eigen::MatrixXcd fft_along_dim(const Eigen::MatrixXd& X, int n, int dim) {
        Eigen::MatrixXcd result;  // 最终的傅里叶变换结果
        Eigen::FFT<double> fft;   // 使用Eigen的FFT

        if (dim == 1) {  // 按列进行傅里叶变换
            result = Eigen::MatrixXcd(n, X.cols()); // 创建结果矩阵，确保行数为 n

            for (int col = 0; col < X.cols(); ++col) {
                Eigen::VectorXd column = X.col(col);

                // 处理每一列：根据要求补零或截断
                if (column.size() < n) {
                    column.conservativeResize(n);
                    column.tail(n - column.size()).setZero();  // 用零填充
                } else if (column.size() > n) {
                    column.conservativeResize(n);  // 截断
                }

                // 对列进行傅里叶变换
                Eigen::VectorXcd fft_result(n);  // 确保结果的大小为 n
                fft.fwd(fft_result, column);
                result.col(col) = fft_result;  // 存储傅里叶变换结果
            }

        } else if (dim == 2) {  // 按行进行傅里叶变换
            result = Eigen::MatrixXcd(X.rows(), n); // 创建结果矩阵，确保行数为 X.rows()，列数为 n

            for (int row = 0; row < X.rows(); ++row) {
                Eigen::VectorXd row_vector = X.row(row).transpose();  // 获取每一行并转置为列向量

                // 处理每一行：根据要求补零或截断
                if (row_vector.size() < n) {
                    row_vector.conservativeResize(n);
                    row_vector.tail(n - row_vector.size()).setZero();  // 用零填充
                } else if (row_vector.size() > n) {
                    row_vector.conservativeResize(n);  // 截断
                }

                // 对行进行傅里叶变换
                Eigen::VectorXcd fft_result(n);  // 确保结果的大小为 n
                fft.fwd(fft_result, row_vector);
                result.row(row) = fft_result.transpose();  // 存储傅里叶变换结果
            }
        } else {
            std::cerr << "Invalid dimension. Please specify either dim=1 (columns) or dim=2 (rows).\n";
        }

        return result;
    }

}  // namespace FFTLibrary_v3
