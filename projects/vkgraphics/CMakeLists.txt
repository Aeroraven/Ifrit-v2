cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-VkRenderer LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/projects/buildinfra/ProjectShared.cmake)

file(GLOB_RECURSE IFRIT_VKRENDERER_SOURCE_FILES_CXX "src/*.cpp")
file(GLOB_RECURSE IFRIT_VKRENDERER_SOURCE_FILES_C "src/*.c")
file(GLOB_RECURSE IFRIT_VKRENDERER_HEADER_FILES "include/*.h")

# add vulkan dependency
include(${IFRIT_PROJECT_DIR}/projects/buildinfra/VulkanResolve.cmake)

# add api export
add_definitions(-DIFRIT_DLL)
add_definitions(-DIFRIT_API_EXPORT)
add_definitions(-DIFRIT_VKGRAPHICS_SHARED_SHADER_PATH="${IFRIT_PROJECT_DIR}/include/ifrit.shader")
add_definitions(-DIFRIT_LOG_MODULE_NAME="Ifrit.VkGraphics")

add_library(ifrit.vkrenderer SHARED ${IFRIT_VKRENDERER_SOURCE_FILES_CXX} ${IFRIT_VKRENDERER_SOURCE_FILES_C} ${IFRIT_VKRENDERER_HEADER_FILES})
target_link_libraries(ifrit.vkrenderer Vulkan::Vulkan Vulkan::shaderc_combined ${FSR2_API_LIBRARIES})

# windows specific
include(${IFRIT_PROJECT_DIR}/projects/buildinfra/WindowsLibConv.cmake)
WindowsPrebuild(ifrit.vkrenderer ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "libifrit.vkrenderer")
WindowsPostbuild(ifrit.vkrenderer "libifrit.vkrenderer.def" "libifrit.vkrenderer.dll" "libifrit.vkrenderer.lib" ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "x64")
AddImpLib(ifrit.vkrenderer ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "libifrit.vkrenderer")