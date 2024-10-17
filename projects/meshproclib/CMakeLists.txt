cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-MeshProcLib LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})
message(STATUS "[IFRIT/EnvCheck]: Working Directory ${IFRIT_PROJECT_DIR}")
set(IFRIT_PROJECT_SUBDIR "${IFRIT_PROJECT_DIR}/projects")
set(IFRIT_EXTERNAL_DEPENDENCY_PATH "${IFRIT_PROJECT_DIR}/dependencies")

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IFRIT_PROJECT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IFRIT_PROJECT_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IFRIT_PROJECT_DIR}/bin)

# set executable path
set(EXECUTABLE_OUTPUT_PATH ${IFRIT_PROJECT_DIR}/bin)

set(IFRIT_ENABLE_SIMD_AVX True)
set(IFRIT_ENABLE_OPTIMIZATION True)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Deps Check
include(CheckCXXCompilerFlag)
include(CheckCSourceCompiles)

include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/SIMDCheck.cmake)

# Versions for GCC
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
        message(FATAL_ERROR "GCC version must be at least 10.0!")
    endif()
endif()

if(${IFRIT_ENABLE_SIMD_SSE})
    add_definitions(-DIFRIT_FEATURE_SIMD)
    message(STATUS "[IFRIT/Feature]: Use Feature: SIMD/SSE")
endif()

if(${IFRIT_ENABLE_SIMD_AVX2})
    add_definitions(-DIFRIT_FEATURE_SIMD_AVX256)
    message(STATUS "[IFRIT/Feature]: Use Feature: SIMD/AVX2")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(${IFRIT_ENABLE_OPTIMIZATION})
        add_compile_options(/O2)
    endif()
else()
    add_compile_options(-g)
    if(${IFRIT_ENABLE_OPTIMIZATION})
        add_compile_options(-O3)
        add_compile_options(-fno-math-errno)
        add_compile_options(-funsafe-math-optimizations)
        add_compile_options(-fno-rounding-math)
        add_compile_options(-fno-signaling-nans)
        add_compile_options(-fexcess-precision=fast)
    endif()
    add_compile_options(-mfma)
    if(${IFRIT_ENABLE_SIMD_SSE})
        add_compile_options(-msse)
    endif()
    if(${IFRIT_ENABLE_SIMD_AVX2})
        add_compile_options(-mavx2)
    endif()
endif()

file(GLOB_RECURSE SOURCE_0 "include/*.h")
file(GLOB_RECURSE SOURCE_1 "src/*.cpp")
file(GLOB_RECURSE SOURCE_1B "src/*.c")

file(GLOB SOURCE_STAGES "./meshproclib.cpp")

include_directories(".")
include_directories(include)
include_directories(${IFRIT_EXTERNAL_DEPENDENCY_PATH})
include_directories(${IFRIT_PROJECT_SUBDIR})
include_directories(${IFRIT_PROJECT_DIR}/dependencies/metis/include)
include_directories(${IFRIT_PROJECT_DIR}/dependencies/meshoptimizer/include)

add_library(ifrit.meshproclib SHARED ${SOURCE_STAGES} ${SOURCE_0} ${SOURCE_1} ${SOURCE_1B})
target_link_libraries(ifrit.meshproclib PUBLIC metis meshoptimizer GKlib)     
