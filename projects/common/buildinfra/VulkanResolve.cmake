# if windows
if(WIN32)
    set(AT_REQ_VULKAN_BASE ${IFRIT_GLOBAL_VULKAN_DIR_WINDOWS})
else()
    set(AT_REQ_VULKAN_BASE ${IFRIT_GLOBAL_VULKAN_DIR_LINUX})
endif()
set(Vulkan_LIBRARY ${AT_REQ_VULKAN_BASE}/Lib/vulkan-1.lib)
set(Vulkan_INCLUDE_DIR ${AT_REQ_VULKAN_BASE}/Include)
find_package(Vulkan REQUIRED)