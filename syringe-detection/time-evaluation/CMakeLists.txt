cmake_minimum_required(VERSION 3.10)
project(syringe_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(syringe_detector src/time-evaluation.cpp)
target_link_libraries(syringe_detector ${OpenCV_LIBS})
target_include_directories(syringe_detector PRIVATE ${OpenCV_INCLUDE_DIRS})
