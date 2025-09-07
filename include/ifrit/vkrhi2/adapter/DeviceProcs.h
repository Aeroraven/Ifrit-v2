#pragma once

#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include <vulkan/vulkan.h>

namespace Ifrit::RHI::VulkanRHI2
{
    struct IFRIT_APIDECL VA_DeviceProcs
    {
        PFN_vkCmdSetDepthTestEnable                    p_vkCmdSetDepthTestEnable       = nullptr;
        PFN_vkCmdSetDepthWriteEnable                   p_vkCmdSetDepthWriteEnable      = nullptr;
        PFN_vkCmdSetDepthCompareOp                     p_vkCmdSetDepthCompareOp        = nullptr;
        PFN_vkCmdSetDepthBoundsTestEnable              p_vkCmdSetDepthBoundsTestEnable = nullptr;
        PFN_vkCmdSetStencilTestEnable                  p_vkCmdSetStencilTestEnable     = nullptr;
        PFN_vkCmdSetStencilOp                          p_vkCmdSetStencilOp             = nullptr;

        PFN_vkCmdSetColorBlendEnableEXT                p_vkCmdSetColorBlendEnableEXT   = nullptr;
        PFN_vkCmdSetColorWriteEnableEXT                p_vkCmdSetColorWriteEnableEXT   = nullptr;
        PFN_vkCmdSetColorWriteMaskEXT                  p_vkCmdSetColorWriteMaskEXT     = nullptr;
        PFN_vkCmdSetColorBlendEquationEXT              p_vkCmdSetColorBlendEquationEXT = nullptr;
        PFN_vkCmdSetLogicOpEXT                         p_vkCmdSetLogicOpEXT            = nullptr;
        PFN_vkCmdSetLogicOpEnableEXT                   p_vkCmdSetLogicOpEnableEXT      = nullptr;
        PFN_vkCmdSetVertexInputEXT                     p_vkCmdSetVertexInputEXT        = nullptr;

        PFN_vkCmdDrawMeshTasksEXT                      p_vkCmdDrawMeshTasksEXT         = nullptr;
        PFN_vkCmdDrawMeshTasksIndirectEXT              p_vkCmdDrawMeshTasksIndirectEXT = nullptr;

        PFN_vkCmdBeginDebugUtilsLabelEXT               p_vkCmdBeginDebugUtilsLabelEXT = nullptr;
        PFN_vkCmdEndDebugUtilsLabelEXT                 p_vkCmdEndDebugUtilsLabelEXT   = nullptr;
        PFN_vkCmdSetCullModeEXT                        p_vkCmdSetCullModeEXT          = nullptr;

        PFN_vkGetRayTracingShaderGroupHandlesKHR       p_vkGetRayTracingShaderGroupHandlesKHR       = nullptr;
        PFN_vkCreateAccelerationStructureKHR           p_vkCreateAccelerationStructureKHR           = nullptr;
        PFN_vkCmdBuildAccelerationStructuresKHR        p_vkCmdBuildAccelerationStructuresKHR        = nullptr;
        PFN_vkGetAccelerationStructureDeviceAddressKHR p_vkGetAccelerationStructureDeviceAddressKHR = nullptr;
        PFN_vkGetAccelerationStructureBuildSizesKHR    p_vkGetAccelerationStructureBuildSizesKHR    = nullptr;
        PFN_vkCmdTraceRaysKHR                          p_vkCmdTraceRaysKHR                          = nullptr;
        PFN_vkCreateRayTracingPipelinesKHR             p_vkCreateRayTracingPipelinesKHR             = nullptr;

        PFN_vkSetDebugUtilsObjectNameEXT               p_vkSetDebugUtilsObjectNameEXT = nullptr;
        PFN_vkSetDebugUtilsObjectTagEXT                p_vkSetDebugUtilsObjectTagEXT  = nullptr;

        PFN_vkGetBufferDeviceAddress                   p_vkGetBufferDeviceAddress = nullptr;

        // Provided by VK_KHR_synchronization2
        PFN_vkCmdPipelineBarrier2                      p_vkCmdPipelineBarrier2 = nullptr;
        PFN_vkCmdSetEvent2                             p_vkCmdSetEvent2        = nullptr;
        PFN_vkCmdResetEvent2                           p_vkCmdResetEvent2      = nullptr;
        PFN_vkCmdWaitEvents2                           p_vkCmdWaitEvents2      = nullptr;
        PFN_vkCmdWriteTimestamp2                       p_vkCmdWriteTimestamp2  = nullptr;
    };
} // namespace Ifrit::RHI::VulkanRHI2
