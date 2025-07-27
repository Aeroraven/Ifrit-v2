cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2)

# set(ENV{http_proxy} "http://127.0.0.1:10809")
# set(ENV{https_proxy} "http://127.0.0.1:10809")

function(set_targets_folder folder_name group_name)
    message(STATUS "Setting targets folder to ${folder_name}")
    get_property(target_list DIRECTORY ${folder_name} PROPERTY BUILDSYSTEM_TARGETS)
    foreach(tgt IN LISTS target_list)
        # message
        message(STATUS "Setting target ${tgt} to folder ${group_name}")
        set_target_properties(${tgt} PROPERTIES FOLDER "${group_name}")
    endforeach()
endfunction()

function(ifrit_add_dependency dependency_name)
    if(MSVC)
        add_compile_options(/MP)
        add_compile_options(/Ob2)
        add_compile_options(/Ot)
        add_compile_options(/Oi)
    endif()
    message(STATUS "[Ifrit.Build.Dependencies] Adding dependency ${dependency_name}")
    add_subdirectory(dependencies/${dependency_name})
endfunction()

function(ifrit_add_core_component component_name)
    if(MSVC)
        add_compile_options(/MP)
        add_compile_options(/Ob2)
        add_compile_options(/Ot) 
        add_compile_options(/Oi)
    endif()
    message(STATUS "[Ifrit.Build.Core] Adding core component ${component_name}")
    add_subdirectory(modules/${component_name})
endfunction()
 
function(ifrit_add_demo_component component_name)
    message(STATUS "[Ifrit.Build.Core] Adding demo foldeer ${component_name}")
    add_subdirectory(${component_name})
endfunction()

# Note following paths should be set to the correct paths on your system
# Fallback paths will be used if `find_package` fails to find the package 
set(IFRIT_GLOBAL_LLVM_DIR_LINUX "/usr/local/llvm-10/")
set(IFRIT_GLOBAL_LLVM_DIR_WINDOWS "C:/llvm-project/")
set(IFRIT_GLOBAL_CUDA_DIR_WINDOWS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/")

set(IFRIT_BASE_PATH ${CMAKE_CURRENT_LIST_DIR})
message(STATUS "[Ifrit.Build.Dependencies] Using Ifrit base path: ${CMAKE_CURRENT_LIST_DIR}")
set(IFRIT_THIRDPARTH_INSTALL_DIR ${IFRIT_BASE_PATH}/build_thirdparty/)

# Set global flags
set(IFRIT_GLOBAL_INTEGRATED_BUILD_FLAG 1)
set(IFRIT_GLOBAL_DISABLE_LLVM 1)
set(CMAKE_WARN_DEPRECATED CACHE BOOL "Disable deprecated warnings" FORCE)


# some global checks
include(cmake/SIMDCheck.cmake)

# global compiler options
if(MSVC)
    add_compile_options(/MP)
endif()
 
# llvm
if (IFRIT_GLOBAL_DISABLE_LLVM)
    message(STATUS "[Ifrit.Build.Dependencies] LLVM disabled. Using pre-defined paths.")
else()
    find_package(LLVM 10...<12 CONFIG QUIET)
    if(LLVM_FOUND)
        # update llvm directories
        set(IFRIT_GLOBAL_LLVM_DIR_WINDOWS "${LLVM_INCLUDE_DIRS}/../")
        set(IFRIT_GLOBAL_LLVM_DIR_LINUX "${LLVM_INCLUDE_DIRS}/../")
    else()
        # warning message
        message(STATUS "[Ifrit.Build.Dependencies] LLVM package not found. Using pre-defined paths.")
    endif()
endif()

# set build output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

if(CMAKE_CONFIGURATION_TYPES)
    foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/../bin)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/../bin)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/../bin)
    endforeach()
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# fsr2
set(FSR2_API_LIBRARIES ffx_fsr2_api_vk_x64 ffx_fsr2_api_x64)

# ktx
set(KTX_FEATURE_TESTS OFF CACHE BOOL "Disable KTX feature tests" FORCE)
set(KTX_FEATURE_TOOLS_CTS OFF CACHE BOOL "Disable KTX tests" FORCE)
set(KTX_FEATURE_TOOLS OFF CACHE BOOL "Disable KTX tools" FORCE)
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1) # hack to prevent CTest added targets

# spdlog
set(SPDLOG_BUILD_SHARED ON CACHE BOOL "Disable KTX tools" FORCE)

# slang
set(SLANG_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Slang examples" FORCE)
set(SLANG_ENABLE_REPLAYER OFF CACHE BOOL "Disable Slang replayer" FORCE)
set(SLANG_ENABLE_TESTS OFF CACHE BOOL "Disable Slang replayer" FORCE)
set(SLANG_ENABLE_GFX OFF CACHE BOOL "Disable Slang GFX" FORCE)
set(SLANG_ENABLE_SLANG_RHI OFF CACHE BOOL "Disable Slang RHI" FORCE)
set(SLANG_LIB_TYPE STATIC CACHE STRING "Slang library type" FORCE)

# tetgen
set(TETGEN_SKIP_INSTALL ON CACHE BOOL "Disable Tetgen install" FORCE)

# openvdb
set(OPENVDB_BUILD_BINARIES  OFF CACHE BOOL "Disable OpenVDB tools" FORCE)
set(OPENVDB_USE_DELAYED_LOADING OFF CACHE BOOL "Disable OpenVDB tools" FORCE)
set(OPENVDB_BUILD_VDB_VIEW ON CACHE BOOL "Enable OpenVDB VDB viewer" FORCE)
set(USE_BLOSC ON CACHE BOOL "Disable OpenVDB Blosc support" FORCE)
set(TBB_ROOT "${IFRIT_THIRDPARTH_INSTALL_DIR}/tbb" CACHE PATH "TBB root directory" FORCE)
message(STATUS "[Ifrit.Build.Dependencies] Using TBB root directory: ${TBB_ROOT}")
set(BLOSC_ROOT "${IFRIT_THIRDPARTH_INSTALL_DIR}/blosc" CACHE PATH "Blosc root directory" FORCE)
set(ZLIB_ROOT "${IFRIT_THIRDPARTH_INSTALL_DIR}/zlib" CACHE PATH "Zlib root directory" FORCE)

# mimalloc
set(MI_BUILD_TESTS OFF CACHE BOOL "Disable mimalloc tests" FORCE)

# ifrit
set(IFRIT_DEMO_LIST CACHE INTERNAL "List of demo stages" FORCE)

# start building
ifrit_add_dependency(spdlog)
ifrit_add_dependency(meshoptimizer)
ifrit_add_dependency(gklib)
ifrit_add_dependency(metis)
ifrit_add_dependency(ktxsoftware)
ifrit_add_dependency(slang)
ifrit_add_dependency(tetgen)
ifrit_add_dependency(openvdb)
ifrit_add_dependency(mimalloc)

ifrit_add_core_component(external/fsr2)
ifrit_add_core_component(external/imgui)
ifrit_add_core_component(external/glfw)

ifrit_add_core_component(reflparser)
ifrit_add_core_component(core)
ifrit_add_core_component(shadercompile)
ifrit_add_core_component(imaging)
ifrit_add_core_component(display)
ifrit_add_core_component(ircompile)
ifrit_add_core_component(softgraphics)
ifrit_add_core_component(geomproc)
ifrit_add_core_component(vkgraphics)
ifrit_add_core_component(runtime)
ifrit_add_core_component(rhi)
ifrit_add_core_component(editor)

ifrit_add_demo_component(demo)

set(IFRIT_GROUP_ENGINE "Project Ifrit/Engine")
set(IFRIT_GROUP_DEPENDENCIES "Project Ifrit/ThirdParty")
set(IFRIT_GROUP_DEMO "Project Ifrit/Demo")
set(IFRIT_GROUP_TOOL "Project Ifrit/Tool")

# openvdb targets
include(cmake/OpenVDBBuildPostprocess.cmake)

# slang targets
include(cmake/SlangBuildPostprocess.cmake)

# mimalloc targets
include(cmake/MiMallocBuildPostprocess.cmake)

# spdlog targets
set_targets_folder(dependencies/spdlog ${IFRIT_GROUP_DEPENDENCIES}/spdlog)

# ktx targets
set_target_properties(ktx PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/ktx)
set_target_properties(ktx_read PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/ktx)
set_target_properties(ktx_version PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/ktx)
set_target_properties(objUtil PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/ktx)
set_target_properties(astcenc-avx2-static PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/ktx)
set_target_properties(obj_basisu_cbind PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/ktx)

# metis and gklib targets
set_target_properties(metis PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/metis)
set_target_properties(GKlib PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/metis)

# meshoptimizer targets
set_target_properties(meshoptimizer PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/meshoptimizer)

# fsr2 targets
set_target_properties(ffx_fsr2_api_x64 PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/fsr2)
set_target_properties(ffx_fsr2_api_vk_x64 PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/fsr2)
set_target_properties(shader_permutations_vk PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/fsr2)

# glfw3 targets
set_target_properties(glfw PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/glfw)
set_target_properties(update_mappings PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/glfw)

# tetgen targets
set_target_properties(tet PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/tetgen)
set_target_properties(tetgen PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/tetgen)

# imgui targets
set_target_properties(ifrit.external.imgui PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/imgui)

# ifrit targets
set_target_properties(ifrit.core PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.shadercompile PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.imaging PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.display PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.ircompile PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.softgraphics PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.geomproc PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.vkgraphics PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.runtime PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.rhi PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.editor PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})

set_target_properties(ifrit.reflparser PROPERTIES FOLDER ${IFRIT_GROUP_TOOL})

# print num demoss
foreach(DEMO_STAGE ${IFRIT_DEMO_LIST})
    set_target_properties(${DEMO_STAGE} PROPERTIES FOLDER ${IFRIT_GROUP_DEMO})
    message(STATUS "[Ifrit.Build.Demo] Added demo stage: ${DEMO_STAGE}")

endforeach()

add_dependencies(ifrit.external.imgui glfw)

add_dependencies(ifrit.core 
    ifrit.reflparser
    spdlog
    mimalloc)
add_dependencies(ifrit.geomproc 
    openvdb_shared
    meshoptimizer 
    metis
    ifrit.core)

add_dependencies(ifrit.softgraphics 
    ifrit.ircompile
    ifrit.core)

add_dependencies(ifrit.imaging 
    ktx
    ifrit.core)

add_dependencies(ifrit.runtime 
    ifrit.rhi 
    ifrit.display 
    ifrit.geomproc 
    ifrit.core 
    ${FSR2_API_LIBRARIES})

add_dependencies(ifrit.vkgraphics 
    ifrit.core 
    ifrit.shadercompile
    ${FSR2_API_LIBRARIES})

add_dependencies(ifrit.editor 
    ifrit.external.imgui
    ifrit.vkgraphics 
    ifrit.runtime)

set(DEMO_DEPENDENCIES 
    ifrit.vkgraphics 
    ifrit.geomproc 
    ifrit.softgraphics 
    ifrit.display 
    ifrit.runtime 
    ifrit.rhi
    ifrit.imaging
)

foreach(DEMO_STAGE ${IFRIT_DEMO_LIST})
    add_dependencies(${DEMO_STAGE} ${DEMO_DEPENDENCIES})
endforeach()
