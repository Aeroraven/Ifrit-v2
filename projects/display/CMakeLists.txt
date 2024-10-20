cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-Display LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})
message(STATUS "[IFRIT/EnvCheck]: Working Directory ${IFRIT_PROJECT_DIR}")

set(IFRIT_PROJECT_SUBDIR "${IFRIT_PROJECT_DIR}/projects")

# SET OUT  DIRECTORY
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IFRIT_PROJECT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IFRIT_PROJECT_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IFRIT_PROJECT_DIR}/bin)

# set executable path
set(EXECUTABLE_OUTPUT_PATH ${IFRIT_PROJECT_DIR}/bin)

set(CMAKE_CXX_STANDARD 20 REQUIRED)

add_definitions(-DIFRIT_DLL)
add_definitions(-DIFRIT_API_EXPORT)

include (${IFRIT_PROJECT_DIR}/projects/common/buildinfra/GLFW3Resolve.cmake)
include (${IFRIT_PROJECT_DIR}/projects/common/buildinfra/VulkanResolve.cmake)

# Glob required files
file(GLOB_RECURSE IFRIT_DISPLAY_SOURCE_FILES_CXX "src/*.cpp")
file(GLOB_RECURSE IFRIT_DISPLAY_SOURCE_FILES_C "src/*.c")
file(GLOB_RECURSE IFRIT_DISPLAY_HEADER_FILES "include/*.h")

# Set include directories
include_directories(".")
include_directories(include)
include_directories(${IFRIT_PROJECT_SUBDIR})

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(${IFRIT_ENABLE_OPTIMIZATION})
        add_compile_options(/O2)
    endif()
else()
    add_compile_options(-g)
    if(${IFRIT_ENABLE_OPTIMIZATION})
        add_compile_options(-O3)
        add_compile_options(-fno-math-errno)
        add_compile_options(-funsafe-math-optimizations)
        add_compile_options(-fno-rounding-math)
        add_compile_options(-fno-signaling-nans)
        add_compile_options(-fexcess-precision=fast)
    endif()
    add_compile_options(-mfma)
    if(${IFRIT_ENABLE_SIMD_SSE})
        add_compile_options(-msse)
    endif()
    if(${IFRIT_ENABLE_SIMD_AVX2})
        add_compile_options(-mavx2)
    endif()
    add_compile_options(-lopengl32)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        add_compile_options(-lgdi32)
    endif()
endif()

# if windows
if(WIN32)
    add_library(ifrit.display SHARED ${IFRIT_DISPLAY_SOURCE_FILES_CXX} ${IFRIT_DISPLAY_SOURCE_FILES_C} ${IFRIT_DISPLAY_HEADER_FILES})
    target_link_libraries(ifrit.display  ${glfw3_LIB} Vulkan::Vulkan)
else()
    add_library(ifrit.display SHARED ${IFRIT_DISPLAY_SOURCE_FILES_CXX} ${IFRIT_DISPLAY_SOURCE_FILES_C} ${IFRIT_DISPLAY_HEADER_FILES})
    target_link_libraries(ifrit.display  glfw3  Vulkan::Vulkan)
endif()

include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/WindowsLibConv.cmake)
WindowsPostbuild(ifrit.display "libifrit.display.def" "libifrit.display.dll" "libifrit.display.lib" ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "x64")
WindowsPrebuild(ifrit.display ${IFRIT_PROJECT_DIR}/bin x64)
AddImpLib(ifrit.display ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "libifrit.display")
