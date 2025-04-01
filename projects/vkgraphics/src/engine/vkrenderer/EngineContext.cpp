
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

#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/vkgraphics/utility/Logger.h"
#include <cstring>
#include <vector>
using namespace Ifrit;

#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>

namespace Ifrit::Graphics::VulkanGraphics
{

    Vec<const char*> m_instanceExtension = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };
    Vec<const char*> m_deviceExtensions  = { VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
         VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME,
         VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
         VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
         VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME, VK_KHR_SPIRV_1_4_EXTENSION_NAME, VK_EXT_MESH_SHADER_EXTENSION_NAME,
         VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME };

    Vec<const char*> m_deviceExtensionsExtended = { VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME };

    bool enableExtension(bool mandatory, const char* extension, const Vec<VkExtensionProperties>& availableExtensions,
        Vec<const char*>& targetExtension)
    {
        for (auto ext : availableExtensions)
        {
            if (strcmp(ext.extensionName, extension) == 0)
            {
                targetExtension.push_back(extension);
                return true;
            }
        }
        if (mandatory)
        {
            // print all available extensions

            iInfo("Available extensions:");
            for (auto ext : availableExtensions)
            {
                vkrLog(ext.extensionName);
            }
            iError("Extension not found: {}", extension);
        }
        return false;
    }
    bool enableLayer(bool mandatory, const char* layer, const Vec<VkLayerProperties>& availableLayers,
        Vec<const char*>& targetLayers)
    {
        for (auto lay : availableLayers)
        {
            if (strcmp(lay.layerName, layer) == 0)
            {
                targetLayers.push_back(layer);
                return true;
            }
        }
        if (mandatory)
        {
            iError("Layer not found: {}", layer);
        }
        return false;
    }
    int physicalDeviceRanking(const VkPhysicalDevice& device)
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        int score = 0;
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            score += 100000;
        }
        score += properties.limits.maxImageDimension2D;
        return score;
    }
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT                                                        messageType,

        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
    {
        if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        {
            iError("Validation layer called");
            iError("Error:{}", pCallbackData->pMessage);

            // This fixes Nsight bug for GPU Trace
            if (pCallbackData->messageIdNumber != 0x8ebf0028)
                std::abort();
        }
        else
        {
            iWarn("Validation layer called");
            iWarn(pCallbackData->pMessage);
        }

        // std::abort();
        return VK_FALSE;
    }
    // START CLASS DEFINITION
    template <typename T> void loadExtFunc(T& extf, const char* name, VkDevice device)
    {
        extf = (T)vkGetDeviceProcAddr(device, name);
        if (!extf)
        {
            iError("Failed to load extension function: {}", name);
        }
    }

    IFRIT_APIDECL const Vec<const char*> EngineContext::GetDeviceExtensions() const { return m_deviceExtensions; }

    IFRIT_APIDECL void                   EngineContext::loadExtensionFunction()
    {
        loadExtFunc(m_extf.p_vkCmdSetDepthTestEnable, "vkCmdSetDepthTestEnable", m_device);
        loadExtFunc(m_extf.p_vkCmdSetDepthWriteEnable, "vkCmdSetDepthWriteEnable", m_device);
        loadExtFunc(m_extf.p_vkCmdSetDepthCompareOp, "vkCmdSetDepthCompareOp", m_device);
        loadExtFunc(m_extf.p_vkCmdSetDepthBoundsTestEnable, "vkCmdSetDepthBoundsTestEnable", m_device);
        loadExtFunc(m_extf.p_vkCmdSetStencilTestEnable, "vkCmdSetStencilTestEnable", m_device);
        loadExtFunc(m_extf.p_vkCmdSetStencilOp, "vkCmdSetStencilOp", m_device);

        loadExtFunc(m_extf.p_vkCmdSetColorBlendEnableEXT, "vkCmdSetColorBlendEnableEXT", m_device);
        loadExtFunc(m_extf.p_vkCmdSetColorWriteEnableEXT, "vkCmdSetColorWriteEnableEXT", m_device);
        loadExtFunc(m_extf.p_vkCmdSetColorWriteMaskEXT, "vkCmdSetColorWriteMaskEXT", m_device);
        loadExtFunc(m_extf.p_vkCmdSetColorBlendEquationEXT, "vkCmdSetColorBlendEquationEXT", m_device);
        loadExtFunc(m_extf.p_vkCmdSetLogicOpEXT, "vkCmdSetLogicOpEXT", m_device);
        loadExtFunc(m_extf.p_vkCmdSetLogicOpEnableEXT, "vkCmdSetLogicOpEnableEXT", m_device);
        loadExtFunc(m_extf.p_vkCmdSetVertexInputEXT, "vkCmdSetVertexInputEXT", m_device);
        loadExtFunc(m_extf.p_vkCmdDrawMeshTasksEXT, "vkCmdDrawMeshTasksEXT", m_device);
        loadExtFunc(m_extf.p_vkCmdDrawMeshTasksIndirectEXT, "vkCmdDrawMeshTasksIndirectEXT", m_device);
        loadExtFunc(m_extf.p_vkCmdBeginDebugUtilsLabelEXT, "vkCmdBeginDebugUtilsLabelEXT", m_device);
        loadExtFunc(m_extf.p_vkCmdEndDebugUtilsLabelEXT, "vkCmdEndDebugUtilsLabelEXT", m_device);
        // loadExtFunc(m_extf.p_vkCmdSetCullModeEXT, "vkCmdSetCullModeEXT", m_device);

        loadExtFunc(m_extf.p_vkGetRayTracingShaderGroupHandlesKHR, "vkGetRayTracingShaderGroupHandlesKHR", m_device);
        loadExtFunc(m_extf.p_vkCreateAccelerationStructureKHR, "vkCreateAccelerationStructureKHR", m_device);
        loadExtFunc(m_extf.p_vkCmdBuildAccelerationStructuresKHR, "vkCmdBuildAccelerationStructuresKHR", m_device);
        loadExtFunc(m_extf.p_vkGetAccelerationStructureDeviceAddressKHR, "vkGetAccelerationStructureDeviceAddressKHR",
            m_device);
        loadExtFunc(
            m_extf.p_vkGetAccelerationStructureBuildSizesKHR, "vkGetAccelerationStructureBuildSizesKHR", m_device);
        loadExtFunc(m_extf.p_vkCmdTraceRaysKHR, "vkCmdTraceRaysKHR", m_device);
        loadExtFunc(m_extf.p_vkCreateRayTracingPipelinesKHR, "vkCreateRayTracingPipelinesKHR", m_device);

        vkrDebug("Extension functions loaded");
    }
    IFRIT_APIDECL
    EngineContext::EngineContext(const Rhi::RhiInitializeArguments& args) : m_args(args) { Init(); }
    IFRIT_APIDECL void EngineContext::WaitIdle() { vkDeviceWaitIdle(m_device); }
    IFRIT_APIDECL void EngineContext::Init()
    {
        VkApplicationInfo appInfo  = {};
        appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName   = "";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName        = "Ifrit-v2";
        appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion         = VK_API_VERSION_1_3;

        VkInstanceCreateInfo instanceCI    = {};
        instanceCI.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceCI.pApplicationInfo        = &appInfo;
        instanceCI.enabledExtensionCount   = 0;
        instanceCI.ppEnabledExtensionNames = nullptr;
        instanceCI.enabledLayerCount       = 0;
        instanceCI.ppEnabledLayerNames     = nullptr;

        // Create delete queue
        m_deleteQueue = std::make_unique<ResourceDeleteQueue>();

        // Instance : Extensions
        u32 extensionCount = 0;
        vkrVulkanAssert(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr),
            "Failed to enumerate instance extensions");
        Vec<VkExtensionProperties> availableExtensions(extensionCount);
        vkrVulkanAssert(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data()),
            "Failed to enumerate instance extensions");
        Vec<const char*> targetExtensions;

        if (m_args.m_extensionGetter)
        {
            u32          extensionCountExtra = 0;
            const char** extensionsExtra     = m_args.m_extensionGetter(&extensionCountExtra);
            for (u32 i = 0; i < extensionCountExtra; i++)
            {
                enableExtension(true, extensionsExtra[i], availableExtensions, targetExtensions);
            }
        }

        if (m_args.m_enableValidationLayer)
        {
            enableExtension(true, VK_EXT_DEBUG_UTILS_EXTENSION_NAME, availableExtensions, targetExtensions);
        }
        for (auto ext : m_instanceExtension)
        {
            enableExtension(true, ext, availableExtensions, targetExtensions);
        }

        instanceCI.enabledExtensionCount   = SizeCast<u32>(targetExtensions.size());
        instanceCI.ppEnabledExtensionNames = targetExtensions.data();

        // Instance : Layers
        u32 layerCount = 0;
        vkrVulkanAssert(
            vkEnumerateInstanceLayerProperties(&layerCount, nullptr), "Failed to enumerate instance layers");
        Vec<VkLayerProperties> availableLayers(layerCount);
        vkrVulkanAssert(vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()),
            "Failed to enumerate instance layers");
        Vec<const char*> targetLayers;

        if (m_args.m_enableValidationLayer)
        {
            enableLayer(true, s_validationLayerName, availableLayers, targetLayers);
        }
        instanceCI.enabledLayerCount   = SizeCast<u32>(targetLayers.size());
        instanceCI.ppEnabledLayerNames = targetLayers.data();

        vkrVulkanAssert(vkCreateInstance(&instanceCI, nullptr, &m_instance), "Failed to create Vulkan instance");
        vkrDebug("Instance created");

        // Debug Messenger
        if (m_args.m_enableValidationLayer)
        {
            VkDebugUtilsMessengerCreateInfoEXT debugCI = {};
            debugCI.sType                              = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            debugCI.messageSeverity                    = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            debugCI.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            debugCI.pfnUserCallback = debugCallback;
            debugCI.pUserData       = nullptr;

            auto func =
                (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
            if (func)
            {
                vkrVulkanAssert(
                    func(m_instance, &debugCI, nullptr, &m_debugMessenger), "Failed to create debug messenger");
            }
            else
            {
                vkrError("Failed to create debug messenger");
            }
        }

        // Physical Device
        u32 physicalDeviceCount = 0;
        vkrVulkanAssert(vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, nullptr),
            "Failed to enumerate physical devices");
        Vec<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
        vkrVulkanAssert(vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, physicalDevices.data()),
            "Failed to enumerate physical devices");

        VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
        int              bestScore  = 0;
        for (auto device : physicalDevices)
        {
            int score = physicalDeviceRanking(device);
            if (score > bestScore)
            {
                bestScore  = score;
                bestDevice = device;
            }
        }
        if (bestDevice == VK_NULL_HANDLE)
        {
            vkrError("No suitable physical device found");
        }
        m_physicalDevice = bestDevice;
        vkrDebug("Physical device selected");

        // Physical Device Propertie
        vkGetPhysicalDeviceProperties(m_physicalDevice, &m_phyDeviceProperties);

        // Queue Family
        u32 queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(bestDevice, &queueFamilyCount, nullptr);
        Vec<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(bestDevice, &queueFamilyCount, queueFamilies.data());
        for (u32 i = 0; i < queueFamilyCount; i++)
        {
            DeviceQueueInfo::DeviceQueueFamily family;
            family.m_familyIndex = i;
            family.m_queueCount  = queueFamilies[i].queueCount;
            family.m_capability  = queueFamilies[i].queueFlags;
            m_queueInfo.m_queueFamilies.push_back(family);
        }

        // Queue Creation
        Vec<VkDeviceQueueCreateInfo> queueCreateInfos;
        Vec<Vec<float>>              queuePriorities;
        for (auto family : m_queueInfo.m_queueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex        = family.m_familyIndex;
            queueCreateInfo.queueCount              = family.m_queueCount;
            queuePriorities.push_back(Vec<float>(family.m_queueCount, 1.0f));
            queueCreateInfo.pQueuePriorities = queuePriorities.back().data();
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // Device
        VkPhysicalDeviceFeatures                           deviceFeatures                      = {};
        VkPhysicalDeviceVulkan12Features                   deviceFeatures12                    = {};
        VkPhysicalDeviceDynamicRenderingFeaturesKHR        deviceFeaturesDynamic               = {};
        VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT deviceFeaturesDynamicVertexInput    = {};
        VkPhysicalDeviceExtendedDynamicState3FeaturesEXT   deviceFeaturesExtendedDynamicState3 = {};
        VkPhysicalDeviceExtendedDynamicState2FeaturesEXT   deviceFeaturesExtendedState2        = {};
        VkPhysicalDeviceExtendedDynamicStateFeaturesEXT    deviceFeaturesExtendedState         = {};
        VkPhysicalDeviceColorWriteEnableFeaturesEXT        deviceFeaturesColorWriteEnable      = {};
        VkPhysicalDeviceDescriptorIndexingFeatures         descriptorIndexingFeatures{};
        VkPhysicalDeviceMeshShaderFeaturesEXT              meshShaderFeatures{};
        VkPhysicalDeviceHostQueryResetFeaturesEXT          hostQueryResetFeatures{};

        deviceFeatures12.sType                           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        deviceFeatures12.timelineSemaphore               = VK_TRUE;
        deviceFeatures12.descriptorIndexing              = VK_TRUE;
        deviceFeatures12.descriptorBindingPartiallyBound = VK_TRUE;
        deviceFeatures12.descriptorBindingSampledImageUpdateAfterBind       = VK_TRUE;
        deviceFeatures12.descriptorBindingStorageBufferUpdateAfterBind      = VK_TRUE;
        deviceFeatures12.descriptorBindingStorageImageUpdateAfterBind       = VK_TRUE;
        deviceFeatures12.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;
        deviceFeatures12.descriptorBindingUniformBufferUpdateAfterBind      = VK_TRUE;
        deviceFeatures12.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
        deviceFeatures12.descriptorBindingUpdateUnusedWhilePending          = VK_TRUE;
        deviceFeatures12.descriptorBindingVariableDescriptorCount           = VK_TRUE;
        deviceFeatures12.runtimeDescriptorArray                             = VK_TRUE;
        deviceFeatures12.hostQueryReset                                     = VK_TRUE;
        deviceFeatures12.shaderSharedInt64Atomics                           = VK_TRUE;
        deviceFeatures12.shaderBufferInt64Atomics                           = VK_TRUE;
        deviceFeatures12.shaderFloat16                                      = VK_TRUE;
        deviceFeatures12.bufferDeviceAddress                                = VK_TRUE;
        deviceFeatures12.pNext                                              = &deviceFeaturesDynamic;

        deviceFeaturesDynamic.sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;
        deviceFeaturesDynamic.dynamicRendering = VK_TRUE;
        deviceFeaturesDynamic.pNext            = &deviceFeaturesDynamicVertexInput;

        deviceFeaturesDynamicVertexInput.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT;
        deviceFeaturesDynamicVertexInput.vertexInputDynamicState = VK_TRUE;
        deviceFeaturesDynamicVertexInput.pNext                   = &deviceFeaturesExtendedDynamicState3;

        deviceFeaturesExtendedDynamicState3.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
        deviceFeaturesExtendedDynamicState3.extendedDynamicState3ColorBlendEnable   = VK_TRUE;
        deviceFeaturesExtendedDynamicState3.extendedDynamicState3LogicOpEnable      = VK_TRUE;
        deviceFeaturesExtendedDynamicState3.extendedDynamicState3ColorBlendEquation = VK_TRUE;
        deviceFeaturesExtendedDynamicState3.extendedDynamicState3ColorWriteMask     = VK_TRUE;
        deviceFeaturesExtendedDynamicState3.pNext                                   = &deviceFeaturesExtendedState2;

        deviceFeaturesExtendedState2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT;
        deviceFeaturesExtendedState2.extendedDynamicState2        = VK_TRUE;
        deviceFeaturesExtendedState2.extendedDynamicState2LogicOp = VK_TRUE;
        deviceFeaturesExtendedState2.pNext                        = &deviceFeaturesExtendedState;

        deviceFeaturesExtendedState.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT;
        deviceFeaturesExtendedState.extendedDynamicState = VK_TRUE;
        deviceFeaturesExtendedState.pNext                = &deviceFeaturesColorWriteEnable;

        deviceFeaturesColorWriteEnable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT;
        deviceFeaturesColorWriteEnable.colorWriteEnable = VK_TRUE;
        deviceFeaturesColorWriteEnable.pNext            = &meshShaderFeatures;

        meshShaderFeatures.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
        meshShaderFeatures.taskShader = VK_TRUE;
        meshShaderFeatures.meshShader = VK_TRUE;

        deviceFeatures.samplerAnisotropy = VK_TRUE;
        deviceFeatures.geometryShader    = VK_TRUE;
        deviceFeatures.shaderFloat64     = VK_TRUE;
        deviceFeatures.shaderInt64       = VK_TRUE;
        deviceFeatures.shaderInt16       = VK_TRUE;

        VkDeviceCreateInfo deviceCI   = {};
        deviceCI.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCI.queueCreateInfoCount = SizeCast<u32>(queueCreateInfos.size());
        deviceCI.pQueueCreateInfos    = queueCreateInfos.data();
        deviceCI.pEnabledFeatures     = &deviceFeatures;
        deviceCI.pNext                = &deviceFeatures12;

        // Device : Extensions
        Vec<const char*> tarGetDeviceExtensions;
        u32              extensionCountDevice = 0;
        vkrVulkanAssert(vkEnumerateDeviceExtensionProperties(bestDevice, nullptr, &extensionCountDevice, nullptr),
            "Failed to enumerate device extensions");
        Vec<VkExtensionProperties> availableExtensionsDevice(extensionCountDevice);
        vkrVulkanAssert(vkEnumerateDeviceExtensionProperties(
                            bestDevice, nullptr, &extensionCountDevice, availableExtensionsDevice.data()),
            "Failed to enumerate device extensions");
        for (auto extension : m_deviceExtensions)
        {
            enableExtension(true, extension, availableExtensionsDevice, tarGetDeviceExtensions);
        }
        if (m_args.m_enableHardwareRayTracing)
        {
            iInfo("Hardware ray tracing enabled");
            for (auto extension : m_deviceExtensionsExtended)
            {
                enableExtension(true, extension, availableExtensionsDevice, tarGetDeviceExtensions);
            }
        }
        else
        {
            iInfo("Hardware ray tracing disabled");
        }

        deviceCI.enabledExtensionCount   = SizeCast<u32>(tarGetDeviceExtensions.size());
        deviceCI.ppEnabledExtensionNames = tarGetDeviceExtensions.data();

        // Device : Layers
        u32 layerCountDevice = 0;
        vkrVulkanAssert(vkEnumerateDeviceLayerProperties(bestDevice, &layerCountDevice, nullptr),
            "Failed to enumerate device layers");
        Vec<VkLayerProperties> availableLayersDevice(layerCountDevice);
        vkrVulkanAssert(vkEnumerateDeviceLayerProperties(bestDevice, &layerCountDevice, availableLayersDevice.data()),
            "Failed to enumerate device layers");
        Vec<const char*> targetLayersDevice;

        if (m_args.m_enableValidationLayer)
        {
            enableLayer(true, s_validationLayerName, availableLayersDevice, targetLayersDevice);
        }
        deviceCI.enabledLayerCount   = SizeCast<u32>(targetLayersDevice.size());
        deviceCI.ppEnabledLayerNames = targetLayersDevice.data();
        vkrVulkanAssert(vkCreateDevice(bestDevice, &deviceCI, nullptr, &m_device), "Failed to create logical device");
        vkrDebug("Logical device created");

        // Retrieve Queues
        for (u32 i = 0; i < m_queueInfo.m_queueFamilies.size(); i++)
        {
            for (u32 j = 0; j < m_queueInfo.m_queueFamilies[i].m_queueCount; j++)
            {
                DeviceQueueInfo::DeviceQueue queue;
                vkGetDeviceQueue(m_device, i, j, &queue.m_queue);
                queue.m_familyIndex = i;
                queue.m_queueIndex  = j;
                m_queueInfo.m_allQueues.push_back(queue);
                if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                {
                    m_queueInfo.m_graphicsQueues.push_back(queue);
                }
                if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
                {
                    m_queueInfo.m_computeQueues.push_back(queue);
                }
                if (queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT)
                {
                    m_queueInfo.m_transferQueues.push_back(queue);
                }
            }
        }

        // Allocator
        VmaAllocatorCreateInfo allocatorCI = {};
        allocatorCI.vulkanApiVersion       = VK_API_VERSION_1_3;
        allocatorCI.physicalDevice         = m_physicalDevice;
        allocatorCI.device                 = m_device;
        allocatorCI.instance               = m_instance;
        allocatorCI.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        vmaCreateAllocator(&allocatorCI, &m_allocator);

        loadExtensionFunction();
        vkrDebug("Allocator created");

        vkrLog("Engine context initialized");
    }

    void EngineContext::Destructor()
    {
        vmaDestroyAllocator(m_allocator);
        vkDestroyDevice(m_device, nullptr);
        if (m_args.m_enableValidationLayer)
        {
            auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
                m_instance, "vkDestroyDebugUtilsMessengerEXT");
            if (func)
            {
                func(m_instance, m_debugMessenger, nullptr);
            }
        }
        vkDestroyInstance(m_instance, nullptr);
    }

    EngineContext::~EngineContext() { Destructor(); }
} // namespace Ifrit::Graphics::VulkanGraphics
