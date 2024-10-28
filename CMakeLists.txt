cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2x)

# Note following paths should be set to the correct paths on your system
# Fallback paths will be used if `find_package` fails to find the package 
set(IFRIT_GLOBAL_LLVM_DIR_LINUX "/usr/local/llvm-10/")
if(MSVC)
    set(IFRIT_GLOBAL_LLVM_DIR_WINDOWS "C:/llvm-project-msvc/")
else()
    set(IFRIT_GLOBAL_LLVM_DIR_WINDOWS "C:/llvm-project/")
endif()
set(IFRIT_GLOBAL_CUDA_DIR_WINDOWS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/")
set(IFRIT_GLOBAL_GLFW_DIR_WINDOWS "C:/Program Files (x86)/GLFW/")

set(IFRIT_GLOBAL_INTEGRATED_BUILD_FLAG 1)

# Find LLVM
find_package(LLVM 10...<12 CONFIG QUIET)
if(LLVM_FOUND)
    # update llvm directories
    set(IFRIT_GLOBAL_LLVM_DIR_WINDOWS "${LLVM_INCLUDE_DIRS}/../")
    set(IFRIT_GLOBAL_LLVM_DIR_LINUX "${LLVM_INCLUDE_DIRS}/../")
else()
    # warning message
    message(STATUS "[Ifrit/Deps] LLVM package not found. Using pre-defined paths.")
endif()

# Find GLFW
find_package(glfw3 3.3 CONFIG QUIET)
if(glfw3_FOUND)
    # Nothing todo
else()
    # warning message
    message(STATUS "[Ifrit/Deps] GLFW3 package not found. Using pre-defined paths.")
endif()

# set build output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(dependencies/glfw)
add_subdirectory(dependencies/meshoptimizer)
add_subdirectory(dependencies/gklib)
add_subdirectory(dependencies/metis)
add_subdirectory(projects/display)
add_subdirectory(projects/ircompile)
add_subdirectory(projects/softgraphics)
add_subdirectory(projects/meshproclib)
add_subdirectory(projects/vkgraphics)
add_subdirectory(projects/demo)

add_dependencies(ifrit.meshproclib meshoptimizer metis)
add_dependencies(ifrit.softrenderer ifrit.ircompile)
add_dependencies(ifrit.demo ifrit.vkrenderer ifrit.meshproclib ifrit.softrenderer ifrit.display)
