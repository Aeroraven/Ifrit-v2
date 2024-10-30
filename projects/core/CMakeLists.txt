cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-CoreLib LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/ProjectShared.cmake)

file(GLOB_RECURSE IFRIT_CORELIB_SOURCE_FILES_CXX "src/*.cpp")
file(GLOB_RECURSE IFRIT_CORELIB_SOURCE_FILES_C "src/*.c")
file(GLOB_RECURSE IFRIT_CORELIB_HEADER_FILES ${IFRIT_PROJECT_DIR}"/include/ifrit/*.h")

# add api export
add_definitions(-DIFRIT_DLL)
add_definitions(-DIFRIT_API_EXPORT)

# set cxx20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(ifrit.core SHARED ${IFRIT_CORELIB_SOURCE_FILES_CXX} ${IFRIT_CORELIB_SOURCE_FILES_C} ${IFRIT_CORELIB_HEADER_FILES})
target_link_libraries(ifrit.core ifrit.rhi ifrit.display)