cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2x)

# Note following paths should be set to the correct paths on your system
# Fallback paths will be used if `find_package` fails to find the package 
set(IFRIT_GLOBAL_LLVM_DIR_LINUX "/usr/local/llvm-10/")
set(IFRIT_GLOBAL_LLVM_DIR_WINDOWS "C:/llvm-project/")
set(IFRIT_GLOBAL_CUDA_DIR_WINDOWS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/")

# Set global flags
set(IFRIT_GLOBAL_INTEGRATED_BUILD_FLAG 1)
set(IFRIT_GLOBAL_DISABLE_LLVM 1)
 
# llvm
if (IFRIT_GLOBAL_DISABLE_LLVM)
    message(STATUS "[Ifrit/Deps] LLVM disabled. Using pre-defined paths.")
else()
    find_package(LLVM 10...<12 CONFIG QUIET)
    if(LLVM_FOUND)
        # update llvm directories
        set(IFRIT_GLOBAL_LLVM_DIR_WINDOWS "${LLVM_INCLUDE_DIRS}/../")
        set(IFRIT_GLOBAL_LLVM_DIR_LINUX "${LLVM_INCLUDE_DIRS}/../")
    else()
        # warning message
        message(STATUS "[Ifrit/Deps] LLVM package not found. Using pre-defined paths.")
    endif()
endif()

# set build output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# fsr2
set(FSR2_API_LIBRARIES ffx_fsr2_api_vk_x64 ffx_fsr2_api_x64)

# ktx
set(KTX_FEATURE_TESTS OFF CACHE BOOL "Disable KTX feature tests" FORCE)
set(KTX_FEATURE_TOOLS_CTS OFF CACHE BOOL "Disable KTX tests" FORCE)
set(KTX_FEATURE_TOOLS OFF CACHE BOOL "Disable KTX tools" FORCE)

# start building
add_subdirectory(dependencies/glfw)
add_subdirectory(dependencies/meshoptimizer)
add_subdirectory(dependencies/gklib)
add_subdirectory(dependencies/metis)
add_subdirectory(dependencies/ktxsoftware)

add_subdirectory(projects/core)
add_subdirectory(projects/imaging)
add_subdirectory(projects/display)
add_subdirectory(projects/ircompile)
add_subdirectory(projects/softgraphics)
add_subdirectory(projects/meshproclib)
add_subdirectory(projects/vkgraphics)
add_subdirectory(projects/external/fsr2)
add_subdirectory(projects/runtime)
add_subdirectory(projects/rhi)
add_subdirectory(projects/demo)

add_dependencies(ifrit.meshproclib meshoptimizer metis)
add_dependencies(ifrit.softrenderer ifrit.ircompile)
add_dependencies(ifrit.imaging ktx)
add_dependencies(ifrit.runtime ifrit.rhi ifrit.display ifrit.meshproclib ifrit.core ${FSR2_API_LIBRARIES})

set(DEMO_DEPENDENCIES 
    ifrit.vkrenderer 
    ifrit.meshproclib 
    ifrit.softrenderer 
    ifrit.display 
    ifrit.runtime 
    ifrit.rhi
    ifrit.imaging
)
add_dependencies(ifrit.demo.syaro ${DEMO_DEPENDENCIES})
add_dependencies(ifrit.demo.ayanami ${DEMO_DEPENDENCIES})
add_dependencies(ifrit.demo.test ${DEMO_DEPENDENCIES})
add_dependencies(ifrit.demo.memdiag ${DEMO_DEPENDENCIES})
