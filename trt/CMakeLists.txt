cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(DaSiam)

find_package(Threads)
find_package(CUDA REQUIRED)
find_package(CUDNN REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)


set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER /usr/bin/g++)

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_86,code=sm_86)


SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -Wunused-function")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )




# TARGETS --------------------------------------------------------------------------------------------------------------

cuda_add_executable(${PROJECT_NAME} main.cpp dasiam.cpp utils.cpp d_utils.cu)
# cuda_add_executable(${PROJECT_NAME} utils.cu )
# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "86")
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}  ${CUDNN_LIBRARIES}  ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
# target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} )
message(${OpenCV_LIBS})