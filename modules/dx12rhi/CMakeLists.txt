cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2-DX12RHI LANGUAGES CXX C)

set(IFRIT_PROJECT_DIR_CORE "${CMAKE_CURRENT_SOURCE_DIR}")
string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1" IFRIT_PROJECT_DIR  ${IFRIT_PROJECT_DIR_CORE})

include(${IFRIT_PROJECT_DIR}/cmake/ProjectShared.cmake)

file(GLOB_RECURSE IFRIT_DX12RHI_SOURCE_FILES_CXX "src/*.cpp")
file(GLOB_RECURSE IFRIT_DX12RHI_SOURCE_FILES_C "src/*.c")
file(GLOB_RECURSE IFRIT_DX12RHI_HEADER_FILES "include/*.h")


# add api export
add_definitions(-DIFRIT_DLL)
add_definitions(-DIFRIT_API_EXPORT)
add_definitions(-DIFRIT_DX12RHI_SHARED_SHADER_PATH="${IFRIT_PROJECT_DIR}/include/ifrit.shader")
add_definitions(-DIFRIT_DX12RHI_SHARED_SHADER_NEXT_PATH="${IFRIT_PROJECT_DIR}/include/ifrit.shader.neo")
add_definitions(-DIFRIT_DX12RHI_SHARED_SHADER_NEXT_INCLUDE_BASE="${IFRIT_PROJECT_DIR}/include/")
add_definitions(-DIFRIT_LOG_MODULE_NAME="Ifrit.DirectX12RHI")

#pch
set(IFRIT_DX12RHI_PCH_HEADER ${IFRIT_PROJECT_DIR}/include/ifrit/dx12rhi/common/Pch.h)


add_library(ifrit.dx12rhi SHARED ${IFRIT_DX12RHI_SOURCE_FILES_CXX} ${IFRIT_DX12RHI_SOURCE_FILES_C} ${IFRIT_DX12RHI_HEADER_FILES})
target_precompile_headers(ifrit.dx12rhi
    PRIVATE 
    $<$<CXX_COMPILER_ID:MSVC>:"${IFRIT_DX12RHI_PCH_HEADER}"> 
    $<$<CXX_COMPILER_ID:GNU>:"${IFRIT_DX12RHI_PCH_HEADER}">)

target_link_libraries(ifrit.dx12rhi 
    ifrit.shadercompile
    ${IFRIT_COMMON_DEPENDENCIES_SHARED}
    ifrit.core
    d3d12.lib
    dxgi.lib
    dxguid.lib
    dxcompiler.lib
)

add_dependencies(ifrit.dx12rhi 
    ifrit.core 
    ifrit.shadercompile
)
