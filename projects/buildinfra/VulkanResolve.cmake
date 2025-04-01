# if windows
if(WIN32)
    set(AT_REQ_VULKAN_BASE ${IFRIT_GLOBAL_VULKAN_DIR_WINDOWS})
else()
    set(AT_REQ_VULKAN_BASE ${IFRIT_GLOBAL_VULKAN_DIR_LINUX})
endif()
find_package(Vulkan REQUIRED COMPONENTS glslc shaderc_combined)

# check Vulkan_shaderc_combined_FOUND
if(NOT Vulkan_shaderc_combined_FOUND)
    message(FATAL_ERROR "Vulkan_shaderc_combined not found")
endif()

# check glslc
if(NOT Vulkan_glslc_FOUND)
    message(FATAL_ERROR "Vulkan_glslc not found")
endif()

# add include
include_directories(${Vulkan_INCLUDE_DIR})