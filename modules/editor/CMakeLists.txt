cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-CoreLib LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/cmake/ProjectShared.cmake)
include(${IFRIT_PROJECT_DIR}/cmake/SIMDCheck.cmake)

file(GLOB_RECURSE IFRIT_CORELIB_SOURCE_FILES_CXX "src/*.cpp")
file(GLOB_RECURSE IFRIT_CORELIB_SOURCE_FILES_C "src/*.c")
file(GLOB_RECURSE IFRIT_CORELIB_HEADER_FILES ${IFRIT_PROJECT_DIR}"/include/ifrit/*.h")

# add api export
add_definitions(-DIFRIT_DLL)
add_definitions(-DIFRIT_API_EXPORT)
add_definitions(-DIFRIT_LOG_MODULE_NAME="Ifrit.Editor")
add_definitions(-DIFRIT_MODULE_EDITOR)

include_directories(${IFRIT_PROJECT_DIR}/dependencies/imgui)
include_directories(${IFRIT_PROJECT_DIR}/dependencies)

# set cxx20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(ifrit.editor SHARED ${IFRIT_CORELIB_SOURCE_FILES_CXX} ${IFRIT_CORELIB_SOURCE_FILES_C})

# rhi include
set(IFRIT_RUNTIME_INCLUDE_PATH ${IFRIT_PROJECT_DIR}/include/ifrit/editor)
set(IFRIT_RUNTIME_PCH_HEADER ${IFRIT_RUNTIME_INCLUDE_PATH}/common/Pch.h)

add_definitions(-DIFRIT_EDITOR_SHARED_ASSET_PATH="${IFRIT_PROJECT_DIR}/include/ifrit.asset")

target_precompile_headers(ifrit.editor
	PRIVATE 
	$<$<CXX_COMPILER_ID:MSVC>:"${IFRIT_RUNTIME_PCH_HEADER}"> 
	$<$<CXX_COMPILER_ID:GNU>:"${IFRIT_RUNTIME_PCH_HEADER}">)

target_link_libraries(ifrit.editor 
	nfd
	ifrit.rhi 
	ifrit.display 
	ifrit.vkrhi 
	ifrit.core 
	ifrit.imaging
	ifrit.external.imgui
	ifrit.runtime
	${FSR2_API_LIBRARIES}
	${IFRIT_COMMON_DEPENDENCIES_SHARED})

add_dependencies(ifrit.editor 
    nfd
    ifrit.external.imgui
    ifrit.vkrhi 
    ifrit.profiler
    ifrit.runtime)
