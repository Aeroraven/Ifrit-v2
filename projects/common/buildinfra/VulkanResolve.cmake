# if windows
if(WIN32)
    set(AT_REQ_VULKAN_BASE ${IFRIT_GLOBAL_VULKAN_DIR_WINDOWS})
else()
    set(AT_REQ_VULKAN_BASE ${IFRIT_GLOBAL_VULKAN_DIR_LINUX})
endif()
find_package(Vulkan REQUIRED)

# add include
include_directories(${Vulkan_INCLUDE_DIR})
message(STATUS "[IFRIT/EnvCheck]: Vulkan Include Directory ${Vulkan_INCLUDE_DIR}")