cmake_minimum_required(VERSION 3.10)
project(time_evaluation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(time_evaluation src/time-evaluation.cpp)
target_link_libraries(time_evaluation ${OpenCV_LIBS})
target_include_directories(time_evaluation PRIVATE ${OpenCV_INCLUDE_DIRS})
