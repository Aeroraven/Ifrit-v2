cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2x)


# Note following paths should be set to the correct paths on your system
set(IFRIT_GLOBAL_LLVM_DIR_LINUX "/usr/local/llvm-10/")
set(IFRIT_GLOBAL_LLVM_DIR_WINDOWS "C:/llvm-project/")
set(IFRIT_GLOBAL_CUDA_DIR_WINDOWS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/")
set(IFRIT_GLOBAL_GLFW_DIR_WINDOWS "C:/Program Files (x86)/GLFW/")

set(IFRIT_GLOBAL_INTEGRATED_BUILD_FLAG 1)

# Find LLVM
find_package(LLVM 10...<12 CONFIG QUIET)
if(LLVM_FOUND)
    message(STATUS "[Ifrit/Deps] Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "[Ifrit/Deps] Using LLVMConfig.cmake in: ${LLVM_DIR}")
    # update llvm directories
    set(IFRIT_GLOBAL_LLVM_DIR_WINDOWS "${LLVM_INCLUDE_DIRS}/../")
    set(IFRIT_GLOBAL_LLVM_DIR_LINUX "${LLVM_INCLUDE_DIRS}/../")
    message(STATUS "[Ifrit/Deps] LLVM Global Path : ${LLVM_INCLUDE_DIRS}/../")
else()
    # warning message
    message(WARNING "[Ifrit/Deps] LLVM package not found. Using pre-defined paths.")
    message(STATUS "[Ifrit/Deps] LLVM Global Path (Windows): ${IFRIT_GLOBAL_LLVM_DIR_WINDOWS}")
    message(STATUS "[Ifrit/Deps] LLVM Global Path (Linux): ${IFRIT_GLOBAL_LLVM_DIR_LINUX}")
endif()

# Find GLFW
find_package(glfw3 3.3 CONFIG QUIET)
if(glfw3_FOUND)
    message(STATUS "[Ifrit/Deps] Found GLFW3 ${glfw3_VERSION}")
    message(STATUS "[Ifrit/Deps] Using GLFW3Config.cmake in: ${glfw3_DIR}")
else()
    # warning message
    message(WARNING "[Ifrit/Deps] GLFW3 package not found. Using pre-defined paths.")
    message(STATUS "[Ifrit/Deps] GLFW Global Path: ${IFRIT_GLOBAL_GLFW_DIR_WINDOWS}")
endif()

# set build output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
message (STATUS "[Ifrit/EnvCheck]: Output Directory ${CMAKE_BINARY_DIR}/bin")

add_subdirectory(dependencies/meshoptimizer)
add_subdirectory(dependencies/gklib)
add_subdirectory(dependencies/metis)
add_subdirectory(projects/ircompile)
add_subdirectory(projects/softrenderer)
add_subdirectory(projects/meshproclib)
add_subdirectory(projects/demo)

add_dependencies(ifrit.meshproclib meshoptimizer metis)
add_dependencies(ifrit.softrenderer ifrit.ircompile)
add_dependencies(ifrit.demo ifrit.meshproclib ifrit.softrenderer)