cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-MeshProcLib LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/ProjectShared.cmake)

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IFRIT_PROJECT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IFRIT_PROJECT_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IFRIT_PROJECT_DIR}/bin)

# set executable path
set(EXECUTABLE_OUTPUT_PATH ${IFRIT_PROJECT_DIR}/bin)

set(IFRIT_ENABLE_SIMD_AVX True)
set(IFRIT_ENABLE_OPTIMIZATION True)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Deps Check
include(CheckCXXCompilerFlag)
include(CheckCSourceCompiles)

include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/SIMDCheck.cmake)

file(GLOB_RECURSE SOURCE_0 "include/*.h")
file(GLOB_RECURSE SOURCE_1 "src/*.cpp")
file(GLOB_RECURSE SOURCE_1B "src/*.c")

if(NOT MSVC)
    add_compile_options(-fPIC)
endif()

# metis bug fix
# add_library(metismod STATIC IMPORTED)
# set_target_properties(metismod PROPERTIES IMPORTED_LOCATION ${IFRIT_PROJECT_DIR}/dependencies/metismod/metis.lib)
# add_library(gklibmod STATIC IMPORTED)
# set_target_properties(gklibmod PROPERTIES IMPORTED_LOCATION ${IFRIT_PROJECT_DIR}/dependencies/metismod/GKlib/libgklib.a)

# end metis bug fix

# add address sanitizer for msvc
# if(MSVC)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fsanitize=address")
# endif()


include_directories(".")
include_directories(include)
include_directories(${IFRIT_EXTERNAL_DEPENDENCY_PATH})
include_directories(${IFRIT_PROJECT_SUBDIR})
include_directories(${IFRIT_PROJECT_DIR}/dependencies/metis/include)
include_directories(${IFRIT_PROJECT_DIR}/dependencies/meshoptimizer/include)

add_definitions(-DIFRIT_LOG_MODULE_NAME="Ifrit.MeshProcLib")

add_library(ifrit.meshproclib SHARED ${SOURCE_0} ${SOURCE_1} ${SOURCE_1B})
target_link_libraries(ifrit.meshproclib PUBLIC metis meshoptimizer GKlib)     
