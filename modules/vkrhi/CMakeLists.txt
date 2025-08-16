cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-VkRHI LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/cmake/ProjectShared.cmake)

file(GLOB_RECURSE IFRIT_VKRHI_SOURCE_FILES_CXX "src/*.cpp")
file(GLOB_RECURSE IFRIT_VKRHI_SOURCE_FILES_C "src/*.c")
file(GLOB_RECURSE IFRIT_VKRHI_HEADER_FILES "include/*.h")

# add vulkan dependency
include(${IFRIT_PROJECT_DIR}/cmake/VulkanResolve.cmake)

# add api export
add_definitions(-DIFRIT_DLL)
add_definitions(-DIFRIT_API_EXPORT)
add_definitions(-DIFRIT_VKRHI_SHARED_SHADER_PATH="${IFRIT_PROJECT_DIR}/include/ifrit.shader")
add_definitions(-DIFRIT_VKRHI_SHARED_SHADER_NEXT_PATH="${IFRIT_PROJECT_DIR}/include/ifrit.shader.neo")
add_definitions(-DIFRIT_VKRHI_SHARED_SHADER_NEXT_INCLUDE_BASE="${IFRIT_PROJECT_DIR}/include/")
add_definitions(-DIFRIT_LOG_MODULE_NAME="Ifrit.VulkanRHI")

#pch
set(IFRIT_VKRHI_PCH_HEADER ${IFRIT_PROJECT_DIR}/include/ifrit/vkrhi/common/Pch.h)


add_library(ifrit.vkrhi SHARED ${IFRIT_VKRHI_SOURCE_FILES_CXX} ${IFRIT_VKRHI_SOURCE_FILES_C} ${IFRIT_VKRHI_HEADER_FILES})
target_precompile_headers(ifrit.vkrhi
    PRIVATE 
    $<$<CXX_COMPILER_ID:MSVC>:"${IFRIT_VKRHI_PCH_HEADER}"> 
    $<$<CXX_COMPILER_ID:GNU>:"${IFRIT_VKRHI_PCH_HEADER}">)

target_link_libraries(ifrit.vkrhi 
    Vulkan::Vulkan 
    ifrit.shadercompile
    ${FSR2_API_LIBRARIES} 
    ${IFRIT_COMMON_DEPENDENCIES_SHARED}
    ifrit.core)

# windows specific
include(${IFRIT_PROJECT_DIR}/cmake/WindowsLibConv.cmake)
WindowsPrebuild(ifrit.vkrhi ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "libifrit.vkrhi")
WindowsPostbuild(ifrit.vkrhi "libifrit.vkrhi.def" "libifrit.vkrhi.dll" "libifrit.vkrhi.lib" ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "x64")
AddImpLib(ifrit.vkrhi ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} "libifrit.vkrhi")

add_dependencies(ifrit.vkrhi 
    ifrit.core 
    ifrit.shadercompile
    ${FSR2_API_LIBRARIES})
