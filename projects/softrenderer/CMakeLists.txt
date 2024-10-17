cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-Core LANGUAGES CXX C)

set(IFRIT_REQ_GLFW3_BASE ${IFRIT_GLOBAL_GLFW_DIR_WINDOWS})
set(IFRIT_REQ_CUDA_INCLUDE ${IFRIT_GLOBAL_CUDA_DIR_WINDOWS}"/include")
set(IFRIT_REQ_CUDACRT_INCLUDE ${IFRIT_GLOBAL_CUDA_DIR_WINDOWS}"/include/crt")
set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})
message(STATUS "[IFRIT/EnvCheck]: Working Directory ${IFRIT_PROJECT_DIR}")

set(IFRIT_COMPONENT_LLVM "${IFRIT_PROJECT_DIR}/bin")
set(IFRIT_COMPONENT_LLVM_SOURCE "${IFRIT_PROJECT_DIR}/projects/ircompile")
set(IFRIT_PROJECT_SUBDIR "${IFRIT_PROJECT_DIR}/projects")
set(IFRIT_EXTERNAL_DEPENDENCY_PATH "${IFRIT_PROJECT_DIR}/dependencies")
link_directories(${IFRIT_COMPONENT_LLVM})

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IFRIT_PROJECT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IFRIT_PROJECT_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IFRIT_PROJECT_DIR}/bin)

# set executable path
set(EXECUTABLE_OUTPUT_PATH ${IFRIT_PROJECT_DIR}/bin)


set(IFRIT_ENABLE_CUDA False)
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


# Deps Check / glfw3
include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/GLFW3Resolve.cmake)

# Versions for GCC
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
        message(FATAL_ERROR "GCC version must be at least 10.0!")
    endif()
endif()

set(IFRIT_SHADER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/shader")
set(IFRIT_ASSET_PATH "${CMAKE_CURRENT_SOURCE_DIR}/asset")

# fsantize
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fsanitize=address")

add_definitions(-DIFRIT_SHADER_PATH="${IFRIT_SHADER_PATH}")
add_definitions(-DIFRIT_ASSET_PATH="${IFRIT_ASSET_PATH}")

if(${IFRIT_ENABLE_SIMD_SSE})
    add_definitions(-DIFRIT_FEATURE_SIMD)
    message(STATUS "[IFRIT/Feature]: Use Feature: SIMD/SSE")
endif()

if(${IFRIT_ENABLE_SIMD_AVX2})
    add_definitions(-DIFRIT_FEATURE_SIMD_AVX256)
    message(STATUS "[IFRIT/Feature]: Use Feature: SIMD/AVX2")
endif()


if(IFRIT_ENABLE_CUDA)
    enable_language(CUDA)
    include_directories(${IFRIT_REQ_CUDA_INCLUDE})
    include_directories(${IFRIT_REQ_CUDACRT_INCLUDE})
    add_definitions(-DIFRIT_FEATURE_CUDA)
    message(STATUS "[IFRIT/Feature]: Use Feature: CUDA")
endif()

if(${IFRIT_ENABLE_OPTIMIZATION})
    add_definitions(-DIFRIT_FEATURE_AGGRESSIVE_PERFORMANCE)
    message(STATUS "[IFRIT/Feature]: Use Feature: Aggessive Performance")
endif()

add_definitions(-DIFRIT_DLL)
add_definitions(-DIFRIT_API_EXPORT)

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
    add_compile_options(-lopengl32)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        add_compile_options(-lgdi32)
    endif()
    
endif()

file(GLOB_RECURSE SOURCE_0 "include/*.h")
file(GLOB_RECURSE SOURCE_1 "src/*.cpp")
file(GLOB_RECURSE SOURCE_1B "src/*.c")
file(GLOB_RECURSE SOURCE_1C "src/*.cu")
file(GLOB_RECURSE SOURCE_2 "include/external_ref/*.h")
# file(GLOB_RECURSE SOURCE_3 "demo/*")

file(GLOB SOURCE_STAGES "./ifrit.softrenderer.cpp")
file(GLOB SOURCE_STAGES_CUDA "./*.cu")

include_directories(".")
include_directories(include)
include_directories(${IFRIT_COMPONENT_LLVM_SOURCE}/include)
include_directories(${IFRIT_EXTERNAL_DEPENDENCY_PATH})
include_directories(${IFRIT_PROJECT_SUBDIR})

#print include
message(STATUS "[IFRIT/Include]: ${IFRIT_COMPONENT_LLVM_SOURCE}/include")

# Operation System
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(IFRIT_COMPONENT_LLVM_LIB "${IFRIT_COMPONENT_LLVM}/libifrit.ircompile.a")
else()
    set(IFRIT_COMPONENT_LLVM_LIB "${IFRIT_PROJECT_DIR}/bin/libifrit.ircompile.so")
    add_library(IFRIT_COMPONENT_LLVM_LIB_IMP SHARED IMPORTED)
    set_target_properties(IFRIT_COMPONENT_LLVM_LIB_IMP PROPERTIES IMPORTED_LOCATION ${IFRIT_COMPONENT_LLVM_LIB})
    # Linux pthread
    add_compile_options(-pthread)
    add_link_options(-pthread)
    # Linux dlsym
    add_compile_options(-ldl)
    add_link_options(-ldl)
    message(STATUS "[IFRIT/Flags]: Add compiler flag: -ldl -pthread")
endif()

add_library(ifrit.softrenderer SHARED ${SOURCE_0} ${SOURCE_1} ${SOURCE_1B} ${SOURCE_1C})
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(ifrit.softrenderer  ${glfw3_LIB} ifrit.ircompile )
else()
    message(STATUS "[IFRIT/Flags]: LLVM Component: ${IFRIT_COMPONENT_LLVM_LIB}")
    target_link_libraries(ifrit.softrenderer  glfw ifrit.ircompile dl pthread)
endif()

# If MinGW
include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/WindowsLibConv.cmake)
WindowsPrebuild(ifrit.softrenderer ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "libifrit.softrenderer")
WindowsPostbuild(ifrit.softrenderer "libifrit.softrenderer.def" "libifrit.softrenderer.dll" "libifrit.softrenderer.lib" ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "x64")
AddImpLib(ifrit.softrenderer ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "libifrit.softrenderer")