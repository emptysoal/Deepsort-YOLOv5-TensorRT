cmake_minimum_required(VERSION 2.6)

project(yolosort)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-std=c++0x")
find_package(OpenCV REQUIRED)
add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_53;code=sm_53)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories("/usr/include/eigen3")

include_directories(${PROJECT_SOURCE_DIR}/deepsort/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/deepsort/include
)
aux_source_directory(${PROJECT_SOURCE_DIR}/deepsort/src SRC_DIR)

# ===== deepsort =====
add_library(deepsort SHARED ${SRC_DIR})
target_link_libraries(deepsort 
    ${CUDA_LIBS} ${OpenCV_LIBS} 
    cudart nvinfer nvonnxparser
)

# ===== yolo =====
include_directories(${PROJECT_SOURCE_DIR}/yolo/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yolo_infer SHARED 
    ${PROJECT_SOURCE_DIR}/yolo/src/preprocess.cu 
    ${PROJECT_SOURCE_DIR}/yolo/src/yololayer.cu 
    ${PROJECT_SOURCE_DIR}/yolo/src/yolo_lib.cpp
)
target_link_libraries(yolo_infer nvinfer cudart ${OpenCV_LIBS}) 


# ===== main =====
aux_source_directory(${PROJECT_SOURCE_DIR}/src M_SRC_DIR)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(yolosort ${M_SRC_DIR})

target_link_libraries(yolosort nvinfer cudart yolo_infer deepsort)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)
set(CMAKE_CXX_FLAGS "-std=c++0x")
