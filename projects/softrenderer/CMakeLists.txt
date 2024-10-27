cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-Core LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/ProjectShared.cmake)
include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/SIMDCheck.cmake)
include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/GLFW3Resolve.cmake)

set(IFRIT_SHADER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/shader")
set(IFRIT_ASSET_PATH "${CMAKE_CURRENT_SOURCE_DIR}/asset")

add_definitions(-DIFRIT_SHADER_PATH="${IFRIT_SHADER_PATH}")
add_definitions(-DIFRIT_ASSET_PATH="${IFRIT_ASSET_PATH}")

add_definitions(-DIFRIT_DLL)
add_definitions(-DIFRIT_API_EXPORT)

#set c++20
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

file(GLOB_RECURSE SOURCE_0 "include/*.h")
file(GLOB_RECURSE SOURCE_0B "../common/include/*.h")
file(GLOB_RECURSE SOURCE_1 "src/*.cpp")
file(GLOB_RECURSE SOURCE_1B "src/*.c")
# file(GLOB_RECURSE SOURCE_3 "demo/*")

include_directories(".")
include_directories(include)
include_directories(${IFRIT_COMPONENT_LLVM_SOURCE}/include)
include_directories(${IFRIT_EXTERNAL_DEPENDENCY_PATH})

# Operation System
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_compile_options(-pthread)
endif()

add_library(ifrit.softrenderer SHARED ${SOURCE_0} ${SOURCE_1} ${SOURCE_1B} ${SOURCE_0B})
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(ifrit.softrenderer glfw ifrit.ircompile )
else()
    target_link_libraries(ifrit.softrenderer glfw ifrit.ircompile dl pthread)
endif()

# If MinGW
include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/WindowsLibConv.cmake)
WindowsPrebuild(ifrit.softrenderer ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "libifrit.softrenderer")
WindowsPostbuild(ifrit.softrenderer "libifrit.softrenderer.def" "libifrit.softrenderer.dll" "libifrit.softrenderer.lib" ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "x64")
AddImpLib(ifrit.softrenderer ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "libifrit.softrenderer")