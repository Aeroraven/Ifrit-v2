cmake_minimum_required(VERSION 3.15)
set(CMAKE_DEBUG_POSTFIX d)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})


set(BUILD_SHARED_LIBS ON)
option(BUILD_SHARED_LIBS "Build GLFW as shared library" ON)

add_subdirectory(${IFRIT_PROJECT_DIR}/dependencies/nfd ${CMAKE_BINARY_DIR}/nfd)
