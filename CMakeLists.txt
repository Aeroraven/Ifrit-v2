cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(IFRIT_SHADER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/shader")
set(IFRIT_ASSET_PATH "${CMAKE_CURRENT_SOURCE_DIR}/asset")

# fsantize
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fsanitize=address")

message(STATUS "IFRIT_SHADER_PATH: ${IFRIT_SHADER_PATH}")
add_definitions(-DIFRIT_SHADER_PATH="${IFRIT_SHADER_PATH}")
add_definitions(-DIFRIT_ASSET_PATH="${IFRIT_ASSET_PATH}")
add_definitions(-DIFRIT_FEATURE_SIMD)

set(AT_REQ_GLFW3_BASE "C:/Program Files (x86)/GLFW/")
set(glfw3_DIR ${AT_REQ_GLFW3_BASE}/lib/cmake/glfw3/)
set(glfw3_LIB ${AT_REQ_GLFW3_BASE}/lib/glfw3.lib)
find_package(glfw3 3.3 REQUIRED)

file(GLOB_RECURSE SOURCE_0 "include/*.h")
file(GLOB_RECURSE SOURCE_1 "src/*.cpp")
file(GLOB_RECURSE SOURCE_1B "src/*.c")
file(GLOB_RECURSE SOURCE_2 "include/external_ref/*.h")

file(GLOB SOURCE_STAGES "demo/*.cpp")

include_directories(include)
include_directories(${AT_REQ_GLFW3_BASE}/include)

foreach( SRC_FILE_NAME ${SOURCE_STAGES} )
    get_filename_component( SRC_FILE_NAME_SHORT ${SRC_FILE_NAME} NAME_WE )
    add_executable( ${SRC_FILE_NAME_SHORT} ${SRC_FILE_NAME} ${SOURCE_0} ${SOURCE_1} ${SOURCE_1B})
    target_link_libraries( ${SRC_FILE_NAME_SHORT}  ${glfw3_LIB} )
endforeach( SRC_FILE_NAME ${SOURCE_STAGES} )
