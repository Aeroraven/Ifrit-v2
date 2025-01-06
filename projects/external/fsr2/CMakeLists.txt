cmake_minimum_required(VERSION 3.15)
set(CMAKE_DEBUG_POSTFIX d)
option (FFX_FSR2_API_DX12 "Build FSR 2.0 DX12 backend" OFF)
option (FFX_FSR2_API_VK "Build FSR 2.0 Vulkan backend" ON)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

add_definitions(-DFFX_FSR2_BUILD_AS_DLL)
add_definitions(-DFSR2_BUILD_AS_DLL)
set(FSR2_BUILD_AS_DLL ON)
message(STATUS "BIN_DIR: ${CMAKE_BINARY_DIR}")
add_subdirectory(${IFRIT_PROJECT_DIR}/dependencies/fsr2/src/ffx-fsr2-api ${CMAKE_BINARY_DIR}/ffx-fsr2-api)