# set output directory
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

set(IFRIT_DEMO_PROJECT_DIR "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IFRIT_DEMO_PROJECT_DIR}/../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IFRIT_DEMO_PROJECT_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IFRIT_DEMO_PROJECT_DIR}/../bin)

set(IFRIT_INCLUDE_DIR "${IFRIT_PROJECT_DIR}/include")
# set executable path
set(EXECUTABLE_OUTPUT_PATH ${IFRIT_DEMO_PROJECT_DIR}/../bin)
set(IFRIT_PROJECT_SUBDIR "${IFRIT_PROJECT_DIR}/projects")
set(IFRIT_EXTERNAL_DEPENDENCY_PATH "${IFRIT_PROJECT_DIR}/dependencies")
set(IFRIT_COMPONENT_LLVM_SOURCE "${IFRIT_PROJECT_DIR}/projects/ircompile")

# set c++20
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Versions for GCC
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
        message(FATAL_ERROR "GCC version must be at least 10.0!")
    endif()
endif()

if(MSVC)

else()
    add_compile_options(-g)
    if(${IFRIT_ENABLE_OPTIMIZATION})
        add_compile_options($<$<CONFIG:RELEASE>:-O3>)
        add_compile_options($<$<CONFIG:RELEASE>:-fno-math-errno>)
        add_compile_options($<$<CONFIG:RELEASE>:-funsafe-math-optimizations>)
        add_compile_options($<$<CONFIG:RELEASE>:-fno-rounding-math>)
        add_compile_options($<$<CONFIG:RELEASE>:-fno-signaling-nans>)
        add_compile_options($<$<CONFIG:RELEASE>:-fexcess-precision=fast>)
    endif()
endif()

if(${IFRIT_ENABLE_OPTIMIZATION})
    add_definitions(-DIFRIT_FEATURE_AGGRESSIVE_PERFORMANCE)
endif()

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(-lopengl32)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        add_compile_options(-lgdi32)
    endif()
endif()

include_directories(${IFRIT_PROJECT_SUBDIR})
include_directories(${IFRIT_PROJECT_DIR}/dependencies)
include_directories(${IFRIT_INCLUDE_DIR})
