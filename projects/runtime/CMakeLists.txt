cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-CoreLib LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/projects/buildinfra/ProjectShared.cmake)
include(${IFRIT_PROJECT_DIR}/projects/buildinfra/SIMDCheck.cmake)

file(GLOB_RECURSE IFRIT_CORELIB_SOURCE_FILES_CXX "src/*.cpp")
file(GLOB_RECURSE IFRIT_CORELIB_SOURCE_FILES_C "src/*.c")
file(GLOB_RECURSE IFRIT_CORELIB_HEADER_FILES ${IFRIT_PROJECT_DIR}"/include/ifrit/*.h")

# add api export
add_definitions(-DIFRIT_DLL)
add_definitions(-DIFRIT_API_EXPORT)
add_definitions(-DIFRIT_RUNTIME_SHARED_SHADER_PATH="${IFRIT_PROJECT_DIR}/include/ifrit.shader")
add_definitions(-DIFRIT_RUNTIME_SHARED_ASSET_PATH="${IFRIT_PROJECT_DIR}/include/ifrit.asset")
add_definitions(-DIFRIT_LOG_MODULE_NAME="Ifrit.Runtime")
add_definitions(-DIFRIT_MODULE_RUNTIME)

include_directories(${IFRIT_PROJECT_DIR}/dependencies/stb)

# set cxx20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(ifrit.runtime SHARED ${IFRIT_CORELIB_SOURCE_FILES_CXX} ${IFRIT_CORELIB_SOURCE_FILES_C})
target_link_libraries(ifrit.runtime 
	ifrit.rhi 
	ifrit.display 
	ifrit.meshproclib 
	ifrit.core 
	ifrit.imaging
	${FSR2_API_LIBRARIES})
target_compile_options(ifrit.runtime
	PRIVATE 
	$<$<CXX_COMPILER_ID:MSVC>:/bigobj> 
	$<$<CXX_COMPILER_ID:GNU>:-Wa,-mbig-obj>)