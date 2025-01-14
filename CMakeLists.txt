cmake_minimum_required(VERSION 3.28)
project(main)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 CONFIG REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)