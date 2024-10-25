cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-Demo LANGUAGES CXX)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/ProjectShared.cmake)
include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/SIMDCheck.cmake)
include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/GLFW3Resolve.cmake)
include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/VulkanResolve.cmake)

# add shader path
add_definitions(-DIFRIT_DEMO_SHADER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/shader/")
add_definitions(-DIFRIT_DEMO_ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/asset/")

# add executable
add_executable(ifrit.demo ifrit.demo.cpp)

include_directories(${IFRIT_PROJECT_DIR}/dependencies)
include_directories(${IFRIT_PROJECT_SUBDIR})
include_directories(${IFRIT_PROJECT_SUBDIR}/softrenderer/include)
include_directories(${IFRIT_PROJECT_SUBDIR}/display/include)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(ifrit.demo meshoptimizer ifrit.softrenderer ifrit.meshproclib ifrit.display glfw ifrit.vkrenderer)
else()
    target_link_libraries(ifrit.demo meshoptimizer ifrit.softrenderer ifrit.meshproclib ifrit.display glfw ifrit.vkrenderer dl pthread)
endif()
