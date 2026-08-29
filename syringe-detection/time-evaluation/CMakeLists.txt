cmake_minimum_required(VERSION 3.10)
project(syringe_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# 1. Eseguibile per il main principale
add_executable(syringe_detector src/main.cpp)
target_link_libraries(syringe_detector ${OpenCV_LIBS})
target_include_directories(syringe_detector PRIVATE ${OpenCV_INCLUDE_DIRS})

# 2. Eseguibile per la valutazione dei tempi
add_executable(time_evaluation src/time-evaluation.cpp)
target_link_libraries(time_evaluation ${OpenCV_LIBS})
target_include_directories(time_evaluation PRIVATE ${OpenCV_INCLUDE_DIRS})