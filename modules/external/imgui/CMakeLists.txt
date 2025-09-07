cmake_minimum_required(VERSION 3.15)
set(CMAKE_DEBUG_POSTFIX d)

project(imgui LANGUAGES CXX C)
set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ImGui source files

# Create shared library
add_library(ifrit.external.imgui SHARED 
    "${IFRIT_PROJECT_DIR}/dependencies/imgui/imgui.h"
    "${IFRIT_PROJECT_DIR}/dependencies/imgui/imgui.cpp"
    "${IFRIT_PROJECT_DIR}/dependencies/imgui/imgui_demo.cpp"
    "${IFRIT_PROJECT_DIR}/dependencies/imgui/imgui_draw.cpp"
    "${IFRIT_PROJECT_DIR}/dependencies/imgui/imgui_widgets.cpp"
    "${IFRIT_PROJECT_DIR}/dependencies/imgui/imgui_tables.cpp"
    "${IFRIT_PROJECT_DIR}/dependencies/imgui/backends/imgui_impl_vulkan.cpp"
    "${IFRIT_PROJECT_DIR}/dependencies/imgui/backends/imgui_impl_glfw.cpp")


# Define export macro for DLL
target_compile_definitions(ifrit.external.imgui PRIVATE
    IMGUI_API=__declspec\(dllexport\)
)

target_compile_definitions(ifrit.external.imgui INTERFACE
    IMGUI_API=__declspec\(dllimport\)
)

# Include directories
target_include_directories(ifrit.external.imgui PUBLIC
    "${IFRIT_PROJECT_DIR}/dependencies/imgui"
    "${IFRIT_PROJECT_DIR}/dependencies/imgui/backends"
)

# Find required packages
find_package(Vulkan REQUIRED)

# Link libraries
target_link_libraries(ifrit.external.imgui PUBLIC
    Vulkan::Vulkan
    glfw
)

# Platform-specific settings
add_dependencies(ifrit.external.imgui glfw)