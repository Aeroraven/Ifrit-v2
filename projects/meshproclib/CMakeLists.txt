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

# Deps Check / AVX2
set(OLD_CMAKE_REQUIRED_FLAG ${CMAKE_REQUIRED_FLAGS})
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(AVX2_FLAGS "/arch:AVX2")
else()
    set(AVX2_FLAGS "-mavx2")
endif()
set(CMAKE_REQUIRED_FLAGS ${AVX2_FLAGS})
set(CHECK_AVX2_SOURCE "
#include <immintrin.h>
int main() {
    __m256i vec = _mm256_set1_epi32(1);  // AVX2 intrinsic
    return 0;
}
")
check_c_source_compiles("${CHECK_AVX2_SOURCE}" IFRIT_ENABLE_SIMD_AVX2)
set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAG})
message(STATUS "[IFRIT/EnvCheck]: AVX2 Support - ${IFRIT_ENABLE_SIMD_AVX2}")

# Deps Check / SSE
set(OLD_CMAKE_REQUIRED_FLAG ${CMAKE_REQUIRED_FLAGS})
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	set(SSE_FLAGS "/arch:SSE")
else()
	set(SSE_FLAGS "-msse")
endif()
set(CMAKE_REQUIRED_FLAGS ${SSE_FLAGS})
set(CHECK_SSE_SOURCE "
#include <xmmintrin.h>
int main() {
    __m128 vec = _mm_set1_ps(1.0f);  // SSE intrinsic
    return 0;
}
")
check_c_source_compiles("${CHECK_SSE_SOURCE}" IFRIT_ENABLE_SIMD_SSE)
set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAG})
message(STATUS "[IFRIT/EnvCheck]: SSE Support - ${IFRIT_ENABLE_SIMD_SSE}")

# Main

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
        #add_compile_options(-ffinite-math-only)
        add_compile_options(-fno-rounding-math)
        add_compile_options(-fno-signaling-nans)
        #add_compile_options(-fcx-limited-range)
        add_compile_options(-fexcess-precision=fast)
        #add_compile_options(-fallow-store-data-races)
        message(STATUS "[IFRIT/Flags]: Add compiler flag: -Ofast")
    endif()
    add_compile_options(-mfma)
    if(${IFRIT_ENABLE_SIMD_SSE})
        add_compile_options(-msse)
        message(STATUS "[IFRIT/Flags]: Add compiler flag: -msse")
    endif()
    if(${IFRIT_ENABLE_SIMD_AVX2})
        add_compile_options(-mavx2)
        message(STATUS "[IFRIT/Flags]: Add compiler flag: -mavx2")
    endif()
endif()

file(GLOB_RECURSE SOURCE_0 "include/*.h")
file(GLOB_RECURSE SOURCE_1 "src/*.cpp")
file(GLOB_RECURSE SOURCE_1B "src/*.c")
file(GLOB_RECURSE SOURCE_1C "src/*.cu")

file(GLOB SOURCE_STAGES "./meshproclib.cpp")

include_directories(".")
include_directories(include)
include_directories(${IFRIT_EXTERNAL_DEPENDENCY_PATH})
include_directories(${IFRIT_PROJECT_SUBDIR})

# Operation System
foreach( SRC_FILE_NAME ${SOURCE_STAGES} )
    get_filename_component( SRC_FILE_NAME_SHORT ${SRC_FILE_NAME} NAME_WE )
    add_executable( ${SRC_FILE_NAME_SHORT} ${SRC_FILE_NAME} ${SOURCE_0} ${SOURCE_1} ${SOURCE_1B} ${SOURCE_1C} ${SOURCE_3})   
endforeach( SRC_FILE_NAME ${SOURCE_STAGES} )
