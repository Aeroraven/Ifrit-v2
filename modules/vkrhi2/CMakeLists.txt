cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-VkRHI LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/cmake/ProjectShared.cmake)

file(GLOB_RECURSE IFRIT_VKRHI2_SOURCE_FILES_CXX "src/*.cpp")
file(GLOB_RECURSE IFRIT_VKRHI2_SOURCE_FILES_C "src/*.c")
file(GLOB_RECURSE IFRIT_VKRHI2_HEADER_FILES "include/*.h")

# add vulkan dependency
include(${IFRIT_PROJECT_DIR}/cmake/VulkanResolve.cmake)

# add api export
add_definitions(-DIFRIT_DLL)
add_definitions(-DIFRIT_API_EXPORT)
add_definitions(-DIFRIT_VKRHI2_SHARED_SHADER_PATH="${IFRIT_PROJECT_DIR}/include/ifrit.shader")
add_definitions(-DIFRIT_VKRHI2_SHARED_SHADER_NEXT_PATH="${IFRIT_PROJECT_DIR}/include/ifrit.shader.neo")
add_definitions(-DIFRIT_VKRHI2_SHARED_SHADER_NEXT_INCLUDE_BASE="${IFRIT_PROJECT_DIR}/include/")
add_definitions(-DIFRIT_LOG_MODULE_NAME="Ifrit.VulkanRHI2")
add_definitions(-DIFRIT_MODULE_VKRHI2)
add_definitions(-DIFRIT_MODULE_RHI)
#pch
set(IFRIT_VKRHI2_PCH_HEADER ${IFRIT_PROJECT_DIR}/include/ifrit/vkrhi2/common/Pch.h)


add_library(ifrit.vkrhi2 SHARED ${IFRIT_VKRHI2_SOURCE_FILES_CXX} ${IFRIT_VKRHI2_SOURCE_FILES_C} ${IFRIT_VKRHI2_HEADER_FILES})
target_precompile_headers(ifrit.vkrhi2
    PRIVATE 
    $<$<CXX_COMPILER_ID:MSVC>:"${IFRIT_VKRHI2_PCH_HEADER}"> 
    $<$<CXX_COMPILER_ID:GNU>:"${IFRIT_VKRHI2_PCH_HEADER}">)

target_link_libraries(ifrit.vkrhi2 
    Vulkan::Vulkan 
    ifrit.rhi
    ifrit.shadercompile
    ${FSR2_API_LIBRARIES} 
    ${IFRIT_COMMON_DEPENDENCIES_SHARED}
    ifrit.core)

add_dependencies(ifrit.vkrhi2 
    ifrit.core 
    ifrit.rhi
    ifrit.shadercompile
    ${FSR2_API_LIBRARIES})
