
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <cstdint>
#include <vector>
#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#ifdef _WIN32
    #include <Windows.h>
#endif
#include <functional>

namespace Ifrit::Graphics::VulkanGraphics
{

    class IFRIT_APIDECL ResourceDeleteQueue : public Rhi::IRhiDeviceResourceDeleteQueue
    {
    private:
        std::queue<Rhi::RhiDeviceResource*> m_deleteQueue;

    public:
        virtual void AddResourceToDeleteQueue(Rhi::RhiDeviceResource* resource) { m_deleteQueue.push(resource); }

        virtual i32  ProcessDeleteQueue()
        {
            i32 count = 0;
            while (!m_deleteQueue.empty())
            {
                auto resource = m_deleteQueue.front();
                m_deleteQueue.pop();
                if (!resource->GetDebugName().empty())
                    iDebug("Deleting resource: {}", resource->GetDebugName());
                delete resource;
                count++;
            }
            return count;
        }

        virtual ~ResourceDeleteQueue() { ProcessDeleteQueue(); }
    };

    struct IFRIT_APIDECL DeviceQueueInfo
    {
        struct DeviceQueueFamily
        {
            u32     m_familyIndex;
            u32     m_queueCount;
            VkFlags m_capability;
        };
        struct DeviceQueue
        {
            VkQueue m_queue;
            u32     m_familyIndex;
            u32     m_queueIndex;
        };
        Vec<DeviceQueueFamily> m_queueFamilies;
        Vec<DeviceQueue>       m_graphicsQueues;
        Vec<DeviceQueue>       m_computeQueues;
        Vec<DeviceQueue>       m_transferQueues;
        Vec<DeviceQueue>       m_allQueues;
    };

    struct IFRIT_APIDECL ExtensionFunction
    {
        PFN_vkCmdSetDepthTestEnable                    p_vkCmdSetDepthTestEnable;
        PFN_vkCmdSetDepthWriteEnable                   p_vkCmdSetDepthWriteEnable;
        PFN_vkCmdSetDepthCompareOp                     p_vkCmdSetDepthCompareOp;
        PFN_vkCmdSetDepthBoundsTestEnable              p_vkCmdSetDepthBoundsTestEnable;
        PFN_vkCmdSetStencilTestEnable                  p_vkCmdSetStencilTestEnable;
        PFN_vkCmdSetStencilOp                          p_vkCmdSetStencilOp;

        PFN_vkCmdSetColorBlendEnableEXT                p_vkCmdSetColorBlendEnableEXT;
        PFN_vkCmdSetColorWriteEnableEXT                p_vkCmdSetColorWriteEnableEXT;
        PFN_vkCmdSetColorWriteMaskEXT                  p_vkCmdSetColorWriteMaskEXT;
        PFN_vkCmdSetColorBlendEquationEXT              p_vkCmdSetColorBlendEquationEXT;
        PFN_vkCmdSetLogicOpEXT                         p_vkCmdSetLogicOpEXT;
        PFN_vkCmdSetLogicOpEnableEXT                   p_vkCmdSetLogicOpEnableEXT;
        PFN_vkCmdSetVertexInputEXT                     p_vkCmdSetVertexInputEXT;

        PFN_vkCmdDrawMeshTasksEXT                      p_vkCmdDrawMeshTasksEXT;
        PFN_vkCmdDrawMeshTasksIndirectEXT              p_vkCmdDrawMeshTasksIndirectEXT;

        PFN_vkCmdBeginDebugUtilsLabelEXT               p_vkCmdBeginDebugUtilsLabelEXT;
        PFN_vkCmdEndDebugUtilsLabelEXT                 p_vkCmdEndDebugUtilsLabelEXT;
        PFN_vkCmdSetCullModeEXT                        p_vkCmdSetCullModeEXT;

        PFN_vkGetRayTracingShaderGroupHandlesKHR       p_vkGetRayTracingShaderGroupHandlesKHR;
        PFN_vkCreateAccelerationStructureKHR           p_vkCreateAccelerationStructureKHR;
        PFN_vkCmdBuildAccelerationStructuresKHR        p_vkCmdBuildAccelerationStructuresKHR;
        PFN_vkGetAccelerationStructureDeviceAddressKHR p_vkGetAccelerationStructureDeviceAddressKHR;
        PFN_vkGetAccelerationStructureBuildSizesKHR    p_vkGetAccelerationStructureBuildSizesKHR;
        PFN_vkCmdTraceRaysKHR                          p_vkCmdTraceRaysKHR;
        PFN_vkCreateRayTracingPipelinesKHR             p_vkCreateRayTracingPipelinesKHR;
    };

    class IFRIT_APIDECL EngineContext : public Rhi::RhiDevice
    {
    private:
        IF_CONSTEXPR static const char* s_validationLayerName = "VK_LAYER_KHRONOS_validation";
        Rhi::RhiInitializeArguments     m_args;
        VkInstance                      m_instance;
        VkDebugUtilsMessengerEXT        m_debugMessenger;
        VkPhysicalDevice                m_physicalDevice = VK_NULL_HANDLE;
        DeviceQueueInfo                 m_queueInfo;
        VkDevice                        m_device;

        VmaAllocator                    m_allocator;
        ExtensionFunction               m_extf;
        VkPhysicalDeviceProperties      m_phyDeviceProperties{};

        Uref<ResourceDeleteQueue>       m_deleteQueue;

        std::string                     cacheDirectory = "";

    private:
        void Init();
        void loadExtensionFunction();
        void Destructor();

    public:
        EngineContext(const Rhi::RhiInitializeArguments& args);
        ~EngineContext();

        EngineContext(const EngineContext&)            = delete;
        EngineContext& operator=(const EngineContext&) = delete;
        EngineContext(EngineContext&&)                 = delete;

        inline VkInstance                         GetInstance() const { return m_instance; }
        inline VkPhysicalDevice                   GetPhysicalDevice() const { return m_physicalDevice; }
        inline VkDevice                           GetDevice() const { return m_device; }
        inline const DeviceQueueInfo&             GetQueueInfo() const { return m_queueInfo; }
        inline const Rhi::RhiInitializeArguments& GetArgs() const { return m_args; }
        const Vec<const char*>                    GetDeviceExtensions() const;
        inline const VmaAllocator&                GetAllocator() const { return m_allocator; }
        void                                      WaitIdle();
        inline const ExtensionFunction            GetExtensionFunction() const { return m_extf; }
        inline const VkPhysicalDeviceProperties&  GetPhysicalDeviceProperties() const { return m_phyDeviceProperties; }
        inline const std::string&                 GetCacheDir() const { return cacheDirectory; }
        void                                      SetCacheDirectory(const std::string& dir) { cacheDirectory = dir; }
        inline bool                               IsDebugMode() { return m_args.m_enableValidationLayer; }

        inline ResourceDeleteQueue*               GetDeleteQueue() { return m_deleteQueue.get(); }
    };
} // namespace Ifrit::Graphics::VulkanGraphics