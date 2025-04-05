cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-CoreLib LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/projects/buildinfra/ProjectShared.cmake)
include(${IFRIT_PROJECT_DIR}/projects/buildinfra/SIMDCheck.cmake)

file(GLOB_RECURSE IFRIT_IMAGING_SOURCE_FILES_CXX "src/*.cpp")
file(GLOB_RECURSE IFRIT_IMAGING_SOURCE_FILES_C "src/*.c")
file(GLOB_RECURSE IFRIT_IMAGING_HEADER_FILES ${IFRIT_PROJECT_DIR}"/include/ifrit/*.h")

# add api export
add_definitions(-DIFRIT_DLL)
add_definitions(-DIFRIT_API_EXPORT)
add_definitions(-DIFRIT_LOG_MODULE_NAME="Ifrit.Imaging")
add_definitions(-DIFRIT_MODULE_IMAGING)

include_directories(${IFRIT_PROJECT_DIR}/dependencies/ktxsoftware/include)
include_directories(${IFRIT_PROJECT_DIR}/dependencies/stb)

# set cxx20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(ifrit.imaging SHARED ${IFRIT_IMAGING_SOURCE_FILES_CXX} ${IFRIT_IMAGING_SOURCE_FILES_C})
target_link_libraries(ifrit.imaging PUBLIC ktx)     

