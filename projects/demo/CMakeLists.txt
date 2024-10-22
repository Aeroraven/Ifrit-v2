cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-Demo LANGUAGES CXX)

# set output directory
set(IFRIT_DEMO_PROJECT_DIR "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IFRIT_DEMO_PROJECT_DIR}/../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IFRIT_DEMO_PROJECT_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IFRIT_DEMO_PROJECT_DIR}/../bin)

# set executable path
set(EXECUTABLE_OUTPUT_PATH ${IFRIT_DEMO_PROJECT_DIR}/../bin)
set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})
set(IFRIT_PROJECT_SUBDIR "${IFRIT_PROJECT_DIR}/projects")

# add shader path
add_definitions(-DIFRIT_DEMO_SHADER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/shader/")

# set compiler flags
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

if(${IFRIT_ENABLE_SIMD_SSE})
    add_definitions(-DIFRIT_FEATURE_SIMD)
    message(STATUS "[IFRIT/Feature]: Use Feature: SIMD/SSE")
endif()

if(${IFRIT_ENABLE_SIMD_AVX2})
    add_definitions(-DIFRIT_FEATURE_SIMD_AVX256)
    message(STATUS "[IFRIT/Feature]: Use Feature: SIMD/AVX2")
endif()

include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/GLFW3Resolve.cmake)
include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/VulkanResolve.cmake)
# add executable
add_executable(ifrit.demo ifrit.demo.cpp)


#if msvc
if(MSVC)
# add o2 optimization
target_compile_options(ifrit.demo PRIVATE -O2)
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
    add_compile_options(-lopengl32)
    message(STATUS "[IFRIT/Flags]: Add compiler flag: -lopengl32")
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        add_compile_options(-lgdi32)
        message(STATUS "[IFRIT/Flags]: Add compiler flag: -lgdi32")
    endif()
endif()

include_directories(${IFRIT_PROJECT_SUBDIR})
include_directories(${IFRIT_PROJECT_SUBDIR}/softrenderer/include)
include_directories(${IFRIT_PROJECT_SUBDIR}/display/include)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(MSVC)
        target_link_libraries(ifrit.demo ifrit.display ${glfw3_LIB} ifrit.vkrenderer)
    else()
        target_link_libraries(ifrit.demo ifrit.softrenderer ifrit.meshproclib ifrit.display ${glfw3_LIB} ifrit.vkrenderer)
    endif()
else()
    target_link_libraries(ifrit.demo  ifrit.softrenderer ifrit.meshproclib ifrit.display dl pthread glfw3 ifrit.vkrenderer)
endif()
