cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-Rhi LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/ProjectShared.cmake)


file(GLOB_RECURSE IFRIT_RHI_SOURCE_FILES_CXX "src/*.cpp")
file(GLOB_RECURSE IFRIT_RHI_SOURCE_FILES_C "src/*.c")
file(GLOB_RECURSE IFRIT_RHI_HEADER_FILES "include/*.h")

# add api export
add_definitions(-DIFRIT_DLL)
add_definitions(-DIFRIT_API_EXPORT)

add_library(ifrit.rhi SHARED ${IFRIT_RHI_SOURCE_FILES_CXX} ${IFRIT_RHI_SOURCE_FILES_C} ${IFRIT_RHI_HEADER_FILES})
target_link_libraries(ifrit.rhi ifrit.vkrenderer)