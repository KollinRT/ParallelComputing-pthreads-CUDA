cmake_minimum_required(VERSION 3.27)
project(Lab2 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(Lab2 main.cu)

set_target_properties(Lab2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
