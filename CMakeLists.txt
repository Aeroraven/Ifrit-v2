cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2)

set(IFRIT_ENABLE_CUDA True)
set(IFRIT_ENABLE_SIMD_AVX True)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(IFRIT_SHADER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/shader")
set(IFRIT_ASSET_PATH "${CMAKE_CURRENT_SOURCE_DIR}/asset")

set(IFRIT_REQ_GLFW3_BASE "C:/Program Files (x86)/GLFW/")
set(IFRIT_REQ_CUDA_INCLUDE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include")
set(IFRIT_REQ_CUDACRT_INCLUDE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include/crt")

# fsantize
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fsanitize=address")

add_definitions(-DIFRIT_SHADER_PATH="${IFRIT_SHADER_PATH}")
add_definitions(-DIFRIT_ASSET_PATH="${IFRIT_ASSET_PATH}")

if(${IFRIT_ENABLE_SIMD_AVX})
    add_definitions(-DIFRIT_FEATURE_SIMD)
    add_definitions(-DIFRIT_FEATURE_SIMD_AVX256)
endif()

if(IFRIT_ENABLE_CUDA)
    enable_language(CUDA)
    include_directories(${IFRIT_REQ_CUDA_INCLUDE})
    include_directories(${IFRIT_REQ_CUDACRT_INCLUDE})
endif()

set(glfw3_DIR ${IFRIT_REQ_GLFW3_BASE}/lib/cmake/glfw3/)
set(glfw3_LIB ${IFRIT_REQ_GLFW3_BASE}/lib/glfw3.lib)
find_package(glfw3 3.3 REQUIRED)

file(GLOB_RECURSE SOURCE_0 "include/*.h")
file(GLOB_RECURSE SOURCE_1 "src/*.cpp")
file(GLOB_RECURSE SOURCE_1B "src/*.c")
file(GLOB_RECURSE SOURCE_1C "src/*.cu")
file(GLOB_RECURSE SOURCE_2 "include/external_ref/*.h")

file(GLOB SOURCE_STAGES "demo/*.cpp")

include_directories(include)
include_directories(${IFRIT_REQ_GLFW3_BASE}/include)

foreach( SRC_FILE_NAME ${SOURCE_STAGES} )
    get_filename_component( SRC_FILE_NAME_SHORT ${SRC_FILE_NAME} NAME_WE )
    add_executable( ${SRC_FILE_NAME_SHORT} ${SRC_FILE_NAME} ${SOURCE_0} ${SOURCE_1} ${SOURCE_1B} ${SOURCE_1C})
    set_property(TARGET ${SRC_FILE_NAME_SHORT} PROPERTY CUDA_ARCHITECTURES  75)
    target_link_libraries( ${SRC_FILE_NAME_SHORT}  ${glfw3_LIB} )
endforeach( SRC_FILE_NAME ${SOURCE_STAGES} )
