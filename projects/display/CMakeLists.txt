cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-Display LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/ProjectShared.cmake)
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

# if windows
if(WIN32)
    add_library(ifrit.display SHARED ${IFRIT_DISPLAY_SOURCE_FILES_CXX} ${IFRIT_DISPLAY_SOURCE_FILES_C} ${IFRIT_DISPLAY_HEADER_FILES})
    target_link_libraries(ifrit.display  glfw Vulkan::Vulkan)
else()
    add_library(ifrit.display SHARED ${IFRIT_DISPLAY_SOURCE_FILES_CXX} ${IFRIT_DISPLAY_SOURCE_FILES_C} ${IFRIT_DISPLAY_HEADER_FILES})
    target_link_libraries(ifrit.display  glfw  Vulkan::Vulkan)
endif()

add_definitions(-DIFRIT_LOG_MODULE_NAME="Ifrit.Display")

include(${IFRIT_PROJECT_DIR}/projects/common/buildinfra/WindowsLibConv.cmake)
WindowsPostbuild(ifrit.display "libifrit.display.def" "libifrit.display.dll" "libifrit.display.lib" ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "x64")
WindowsPrebuild(ifrit.display ${IFRIT_PROJECT_DIR}/bin x64)
AddImpLib(ifrit.display ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "libifrit.display")
