cmake_minimum_required(VERSION 3.10)
project(Ifrit-v2x)

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
 
# Note following paths should be set to the correct paths on your system
# Fallback paths will be used if `find_package` fails to find the package 
set(IFRIT_GLOBAL_LLVM_DIR_LINUX "/usr/local/llvm-10/")
set(IFRIT_GLOBAL_LLVM_DIR_WINDOWS "C:/llvm-project/")
set(IFRIT_GLOBAL_CUDA_DIR_WINDOWS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/")
 
# Set global flags
set(IFRIT_GLOBAL_INTEGRATED_BUILD_FLAG 1)
set(IFRIT_GLOBAL_DISABLE_LLVM 1)
 
# llvm
if (IFRIT_GLOBAL_DISABLE_LLVM)
    message(STATUS "[Ifrit/Deps] LLVM disabled. Using pre-defined paths.")
else()
    find_package(LLVM 10...<12 CONFIG QUIET)
    if(LLVM_FOUND)
        # update llvm directories
        set(IFRIT_GLOBAL_LLVM_DIR_WINDOWS "${LLVM_INCLUDE_DIRS}/../")
        set(IFRIT_GLOBAL_LLVM_DIR_LINUX "${LLVM_INCLUDE_DIRS}/../")
    else()
        # warning message
        message(STATUS "[Ifrit/Deps] LLVM package not found. Using pre-defined paths.")
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

# spdlog
set(SPDLOG_BUILD_SHARED ON CACHE BOOL "Disable KTX tools" FORCE)

# slang
set(SLANG_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Slang examples" FORCE)
set(SLANG_ENABLE_REPLAYER OFF CACHE BOOL "Disable Slang replayer" FORCE)
set(SLANG_ENABLE_TESTS OFF CACHE BOOL "Disable Slang replayer" FORCE)
set(SLANG_ENABLE_GFX OFF CACHE BOOL "Disable Slang GFX" FORCE)
set(SLANG_ENABLE_SLANG_RHI OFF CACHE BOOL "Disable Slang RHI" FORCE)
set(SLANG_LIB_TYPE STATIC CACHE STRING "Slang library type" FORCE)

# glfw
set(GLFW_INSTALL OFF CACHE BOOL "Disable GLFW install" FORCE)

# start building
add_subdirectory(dependencies/spdlog)
add_subdirectory(dependencies/glfw)
add_subdirectory(dependencies/meshoptimizer)
add_subdirectory(dependencies/gklib)
add_subdirectory(dependencies/metis)
add_subdirectory(dependencies/ktxsoftware)
add_subdirectory(dependencies/slang)

add_subdirectory(projects/core)
add_subdirectory(projects/shadercompile)
add_subdirectory(projects/imaging)
add_subdirectory(projects/display)
add_subdirectory(projects/ircompile)
add_subdirectory(projects/softgraphics)
add_subdirectory(projects/meshproclib)
add_subdirectory(projects/vkgraphics)
add_subdirectory(projects/external/fsr2)
add_subdirectory(projects/runtime)
add_subdirectory(projects/rhi)
add_subdirectory(projects/demo)

set(IFRIT_GROUP_ENGINE "Project Ifrit/Engine")
set(IFRIT_GROUP_DEPENDENCIES "Project Ifrit/ThirdParty")
set(IFRIT_GROUP_DEMO "Project Ifrit/Demo")

# slang targets
# set_targets_folder(dependencies/slang ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(slang PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(slangc PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(slang-common-objects PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(slangd PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(slang-glslang PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(slangi PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(slang-rt PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(core PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)
set_target_properties(compiler-core PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang)

set(IFRIT_SLANG_GENERATED_TARGETS
    slang-capability-defs
    slang-capability-lookup
    slang-fiddle-output
    slang-lookup-tables
    slang-embedded-core-module
    slang-embedded-core-module-source
    slang-no-embedded-core-module
    slang-no-embedded-core-module-source
    slang-glsl-module
    prelude
    copy_slang_headers
    generate_core_module_headers
    copy-slang-llvm
    copy-slang-tint
    copy-webgpu_dawn
)
foreach(target IN LISTS IFRIT_SLANG_GENERATED_TARGETS)
    set_target_properties(${target} PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang/generated)
endforeach()

set(IFRIT_SLANG_GENERATOR_HEADERS
    slang-bootstrap
    slang-lookup-generator
    slang-cpp-parser
    slang-embed
    slang-generate
    slang-spirv-embed-generator
    slang-without-embedded-core-module
    slang-embed
    slang-fiddle
    slang-capability-generator
    all-generators
)
foreach(header IN LISTS IFRIT_SLANG_GENERATOR_HEADERS)
    set_target_properties(${header} PROPERTIES FOLDER ${IFRIT_GROUP_DEPENDENCIES}/slang/generator)
endforeach()

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

# ifrit targets
set_target_properties(ifrit.core PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.shadercompile PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.imaging PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.display PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.ircompile PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.softgraphics PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.meshproclib PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.vkgraphics PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.runtime PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})
set_target_properties(ifrit.rhi PROPERTIES FOLDER ${IFRIT_GROUP_ENGINE})

set_target_properties(ifrit.demo.syaro PROPERTIES FOLDER ${IFRIT_GROUP_DEMO})
set_target_properties(ifrit.demo.ayanami PROPERTIES FOLDER ${IFRIT_GROUP_DEMO})
set_target_properties(ifrit.demo.test PROPERTIES FOLDER ${IFRIT_GROUP_DEMO})
set_target_properties(ifrit.demo.memdiag PROPERTIES FOLDER ${IFRIT_GROUP_DEMO})
set_target_properties(ifrit.demo.cornell PROPERTIES FOLDER ${IFRIT_GROUP_DEMO})

add_dependencies(ifrit.core 
    spdlog)
add_dependencies(ifrit.meshproclib 
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
    ifrit.meshproclib 
    ifrit.core 
    ${FSR2_API_LIBRARIES})

add_dependencies(ifrit.vkgraphics 
    ifrit.core 
    ifrit.shadercompile
    ${FSR2_API_LIBRARIES})

set(DEMO_DEPENDENCIES 
    ifrit.vkgraphics 
    ifrit.meshproclib 
    ifrit.softgraphics 
    ifrit.display 
    ifrit.runtime 
    ifrit.rhi
    ifrit.imaging
)
add_dependencies(ifrit.demo.syaro ${DEMO_DEPENDENCIES})
add_dependencies(ifrit.demo.ayanami ${DEMO_DEPENDENCIES})
add_dependencies(ifrit.demo.test ${DEMO_DEPENDENCIES})
add_dependencies(ifrit.demo.memdiag ${DEMO_DEPENDENCIES})
add_dependencies(ifrit.demo.cornell ${DEMO_DEPENDENCIES})
    