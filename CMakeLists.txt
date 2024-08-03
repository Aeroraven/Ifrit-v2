cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2 LANGUAGES CXX C)

set(IFRIT_ENABLE_CUDA False)
set(IFRIT_ENABLE_SIMD_AVX True)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

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
    add_definitions(-DIFRIT_FEATURE_CUDA)
endif()


if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    
else()
    add_compile_options(-mfma)
    add_compile_options(-mavx2)
    add_compile_options(-lglfw3)
    add_compile_options(-lopengl32)
    add_compile_options(-lgdi32)
    message("Add compiler flag: -mfma -mavx2")
endif()

set(glfw3_DIR ${IFRIT_REQ_GLFW3_BASE}/lib/cmake/glfw3/)
set(glfw3_LIB ${IFRIT_REQ_GLFW3_BASE}/lib/glfw3.lib)
#find_package(glfw3 3.3 REQUIRED)

file(GLOB_RECURSE SOURCE_0 "include/*.h")
file(GLOB_RECURSE SOURCE_1 "src/*.cpp")
file(GLOB_RECURSE SOURCE_1B "src/*.c")
file(GLOB_RECURSE SOURCE_1C "src/*.cu")
file(GLOB_RECURSE SOURCE_2 "include/external_ref/*.h")

file(GLOB SOURCE_STAGES "./*.cpp")
file(GLOB SOURCE_STAGES_CUDA "./*.cu")

include_directories(include)
include_directories(${IFRIT_REQ_GLFW3_BASE}/include)

foreach( SRC_FILE_NAME ${SOURCE_STAGES} )
    get_filename_component( SRC_FILE_NAME_SHORT ${SRC_FILE_NAME} NAME_WE )
    add_executable( ${SRC_FILE_NAME_SHORT} ${SRC_FILE_NAME} ${SOURCE_0} ${SOURCE_1} ${SOURCE_1B} ${SOURCE_1C})
    set_property(TARGET ${SRC_FILE_NAME_SHORT} PROPERTY CUDA_ARCHITECTURES  75)
    target_link_libraries( ${SRC_FILE_NAME_SHORT}  ${glfw3_LIB} )
endforeach( SRC_FILE_NAME ${SOURCE_STAGES} )

foreach( SRC_FILE_NAME ${SOURCE_STAGES_CUDA} )
    get_filename_component( SRC_FILE_NAME_SHORT ${SRC_FILE_NAME} NAME_WE )
    add_executable( ${SRC_FILE_NAME_SHORT} ${SRC_FILE_NAME} ${SOURCE_0} ${SOURCE_1} ${SOURCE_1B} ${SOURCE_1C})
    set_property(TARGET ${SRC_FILE_NAME_SHORT} PROPERTY CUDA_ARCHITECTURES  75)
    set_property(TARGET ${SRC_FILE_NAME_SHORT} PROPERTY CXX_STANDARD 20)
    target_link_libraries( ${SRC_FILE_NAME_SHORT}  ${glfw3_LIB} )
endforeach( SRC_FILE_NAME ${SOURCE_STAGES} )
