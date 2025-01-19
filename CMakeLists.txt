cmake_minimum_required(VERSION 3.10)
project(FFTExample)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)

add_executable(fft_example main.cpp fft.cpp)

target_link_libraries(fft_example PRIVATE Eigen3::Eigen)