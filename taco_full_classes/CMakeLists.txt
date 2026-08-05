cmake_minimum_required(VERSION 3.10)
project(taco_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(taco_detector src/main.cpp)
target_link_libraries(taco_detector ${OpenCV_LIBS})
target_include_directories(taco_detector PRIVATE ${OpenCV_INCLUDE_DIRS})