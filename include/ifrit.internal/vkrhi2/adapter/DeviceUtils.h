#pragma once
#include <vulkan/vulkan.h>
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/core/typing/Traits.h"
#include "ifrit/core/typing/EnumReflection.h"

namespace Ifrit::RHI::VulkanRHI2
{
    IF_CONSTEXPR const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";
    IF_CONSTEXPR auto        kRequiredInstanceExtension =
        Array<const char*, 1>{ VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };

    using FpMessengerCallback = PFN_vkDebugUtilsMessengerCallbackEXT;

    struct VA_PhysicalDeviceCriteria
    {
        u32        mPreferredAdapterId = ~0u;
        ERhiVendor mPreferredVendor    = ERhiVendor::Any;
    };

    struct VA_PhysicalDeviceDesc
    {
        VkPhysicalDevice             mPhysicalDevice = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties2  mProperties{};
        VkPhysicalDeviceIDProperties mIdProperties{};
        u32                          mOriginalIndex = 0;

        Vec<VkExtensionProperties>   mAvailableExtensions;
        Vec<const char*>             mAvailableExtensionsNames;
    };

    struct VA_QueueFamilyDesc
    {
        u32  mFamilyIndex     = 0;
        u32  mQueueCount      = 0;
        bool mValid           = false;
        bool mSupportsPresent = false;
    };

    struct VA_ChosenQueueFamily
    {
        VA_QueueFamilyDesc mGraphics;
        VA_QueueFamilyDesc mCompute;
        VA_QueueFamilyDesc mTransfer;
    };

    // ===== Helper Functions =====
    template <typename T>
        requires IConceptIsFunctionPointer<T>
    T GetInstanceFunctionPtr(VkInstance instance, const char* functionName)
    {
        auto funcPtr = reinterpret_cast<T>(vkGetInstanceProcAddr(instance, functionName));
        if (!funcPtr)
        {
            IF_LOG_CRITICAL("VA_Device", "Failed to get instance function pointer: {}", functionName);
            std::abort();
        }
        return funcPtr;
    }
#define GET_INSTANCE_FUNC(instance, funcName) GetInstanceFunctionPtr<PFN_##funcName>(instance, #funcName)

    IF_NODISCARD Vec<VkExtensionProperties> EnumerateAvailableExtensions()
    {
        u32                        extensionCount = 0;
        Vec<VkExtensionProperties> exts;
        VA_AssertResult(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr),
            "Failed to enumerate instance extensions");
        exts.resize(extensionCount);
        VA_AssertResult(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, exts.data()),
            "Failed to enumerate instance extensions");
        return exts;
    }

    IF_NODISCARD Vec<VkLayerProperties> EnumerateAvailableLayers()
    {
        u32                    layerCount = 0;
        Vec<VkLayerProperties> layers;
        VA_AssertResult(
            vkEnumerateInstanceLayerProperties(&layerCount, nullptr), "Failed to enumerate instance layers");
        layers.resize(layerCount);
        VA_AssertResult(
            vkEnumerateInstanceLayerProperties(&layerCount, layers.data()), "Failed to enumerate instance layers");
        return layers;
    }

    bool EnableExtension(bool mandatory, const char* extension, const Vec<VkExtensionProperties>& availableExtensions,
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
            IF_LOG_CRITICAL("VA_Device", "Extension not found: {}", extension);
        else
            IF_LOG_WARNING("VA_Device", "Extension is not supported: {}", extension);
        return false;
    }

    bool EnableLayer(bool mandatory, const char* layer, const Vec<VkLayerProperties>& availableLayers,
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
            IF_LOG_ERROR("EngineContext", "Layer not found: {}", layer);
        else
            IF_LOG_WARNING("EngineContext", "Layer is not supported: {}", layer);
        return false;
    }

    void SetupValidationMessenger(FpMessengerCallback cb, VkInstance instance, VkDebugUtilsMessengerEXT& messenger)
    {
        VkDebugUtilsMessengerCreateInfoEXT debugCI = {};
        debugCI.sType                              = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCI.messageSeverity                    = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCI.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCI.pfnUserCallback = cb;
        debugCI.pUserData       = nullptr;

        auto func = GET_INSTANCE_FUNC(instance, vkCreateDebugUtilsMessengerEXT);
        IF_LOG_ASSERTION(
            "VA_Device", func != nullptr, "Failed to get function pointer: vkCreateDebugUtilsMessengerEXT");
        VA_AssertResult(func(instance, &debugCI, nullptr, &messenger), "Failed to create debug messenger");
    }

    void ShutdownValidationMessenger(VkInstance instance, VkDebugUtilsMessengerEXT& messenger)
    {
        if (messenger != VK_NULL_HANDLE)
        {
            auto func = GET_INSTANCE_FUNC(instance, vkDestroyDebugUtilsMessengerEXT);
            if (func)
            {
                func(instance, messenger, nullptr);
            }
            messenger = VK_NULL_HANDLE;
        }
    }

    IF_NODISCARD Vec<VkPhysicalDevice> EnumeratePhysicalDevices(VkInstance instance)
    {
        u32 deviceCount = 0;
        VA_AssertResult(
            vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr), "Failed to enumerate physical devices");
        if (deviceCount == 0)
        {
            IF_LOG_CRITICAL("VA_Device", "No Vulkan physical devices found");
            return {};
        }

        Vec<VkPhysicalDevice> devices(deviceCount);
        VA_AssertResult(
            vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()), "Failed to enumerate physical devices");
        return devices;
    }

    IF_NODISCARD VA_PhysicalDeviceDesc SelectPhysicalDevice(
        const Vec<VkPhysicalDevice>& devices, const VA_PhysicalDeviceCriteria& criteria)
    {
        // debug log all devices
        IF_LOG_DEBUG("VA_Device", "Available physical devices:");
        for (auto device : devices)
        {
            VkPhysicalDeviceProperties properties;
            vkGetPhysicalDeviceProperties(device, &properties);
            IF_LOG_DEBUG("VA_Device", "  - {} (type: {}, vendor: {}, id: {:#06x})", String(properties.deviceName),
                GetEnumName(properties.deviceType), properties.vendorID, properties.deviceID);
        }

        Vec<VA_PhysicalDeviceDesc> deviceWithDataList;
        for (auto device : devices)
        {
            VA_PhysicalDeviceDesc data;
            data.mPhysicalDevice = device;

            data.mProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            data.mProperties.pNext = &data.mIdProperties;

            data.mIdProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
            data.mIdProperties.pNext = nullptr;

            vkGetPhysicalDeviceProperties2(device, &data.mProperties);
            deviceWithDataList.push_back(data);
        }

        bool usePreferredId     = criteria.mPreferredAdapterId != ~0u;
        bool usePreferredVendor = criteria.mPreferredVendor != ERhiVendor::Any;
        if (usePreferredId)
        {
            IF_LOG_ASSERTION("VA_Device", criteria.mPreferredAdapterId < deviceWithDataList.size(),
                "Preferred adapter ID is out of range");
            auto& preferredDevice = deviceWithDataList[criteria.mPreferredAdapterId];
            return preferredDevice;
        }
        // sort, with discrete first, then integrated, then cpu. In each category, sort by original index
        std::sort(deviceWithDataList.begin(), deviceWithDataList.end(),
            [usePreferredVendor, criteria](const VA_PhysicalDeviceDesc& a, const VA_PhysicalDeviceDesc& b) {
                if (a.mProperties.properties.deviceType != b.mProperties.properties.deviceType)
                {
                    return (a.mProperties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                        || (b.mProperties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU);
                }
                return a.mOriginalIndex < b.mOriginalIndex;
            });
        for (const auto& deviceData : deviceWithDataList)
        {
            if (usePreferredVendor)
            {
                if (deviceData.mProperties.properties.vendorID == static_cast<u32>(criteria.mPreferredVendor))
                {
                    return deviceData;
                }
            }
            else
            {
                return deviceData;
            }
        }
        IF_LOG_CRITICAL("VA_Device", "No suitable physical device found");
        return {};
    }

    IF_NODISCARD VA_ChosenQueueFamily SelectQueueFamilies(VA_PhysicalDeviceDesc& device, bool enableAsyncCompute)
    {
        VA_ChosenQueueFamily chosen;
        u32                  queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device.mPhysicalDevice, &queueFamilyCount, nullptr);
        Vec<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device.mPhysicalDevice, &queueFamilyCount, queueFamilies.data());
        for (u32 i = 0; i < queueFamilyCount; i++)
        {
            const auto& family = queueFamilies[i];
            // Graphics
            if (!chosen.mGraphics.mValid && (family.queueFlags & VK_QUEUE_GRAPHICS_BIT))
            {
                chosen.mGraphics.mFamilyIndex = i;
                chosen.mGraphics.mQueueCount  = family.queueCount;
                chosen.mGraphics.mValid       = true;
                IF_LOG_DEBUG("VA_Device", "Selected graphics queue family: {}, count: {}", i, family.queueCount);
            }
            // Async Compute
            if (enableAsyncCompute)
            {
                if (!chosen.mCompute.mValid && (family.queueFlags & VK_QUEUE_COMPUTE_BIT)
                    && (chosen.mGraphics.mFamilyIndex != i))
                {
                    chosen.mCompute.mFamilyIndex = i;
                    chosen.mCompute.mQueueCount  = family.queueCount;
                    chosen.mCompute.mValid       = true;
                    IF_LOG_DEBUG(
                        "VA_Device", "Selected async compute queue family: {}, count: {}", i, family.queueCount);
                }
            }
            // Transfer
            if (!chosen.mTransfer.mValid && (family.queueFlags & VK_QUEUE_TRANSFER_BIT)
                && !(family.queueFlags & VK_QUEUE_GRAPHICS_BIT) && !(family.queueFlags & VK_QUEUE_COMPUTE_BIT))
            {
                chosen.mTransfer.mFamilyIndex = i;
                chosen.mTransfer.mQueueCount  = family.queueCount;
                chosen.mTransfer.mValid       = true;
                IF_LOG_DEBUG("VA_Device", "Selected transfer queue family: {}, count: {}", i, family.queueCount);
            }
        }
        return chosen;
    }

    IF_NODISCARD Vec<VkDeviceQueueCreateInfo> CreateQueueCreateInfos(const VA_ChosenQueueFamily& chosenFamilies)
    {
        Vec<VkDeviceQueueCreateInfo> queueCreateInfos;

        // give a static vector with all 1.0f priorities
        static float                 priorities[200];
        for (int i = 0; i < 200; ++i)
            priorities[i] = 1.0f;

        if (chosenFamilies.mGraphics.mValid)
        {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex        = chosenFamilies.mGraphics.mFamilyIndex;
            queueCreateInfo.queueCount              = chosenFamilies.mGraphics.mQueueCount;
            queueCreateInfo.pQueuePriorities        = priorities;
            queueCreateInfos.push_back(queueCreateInfo);
        }
        if (chosenFamilies.mCompute.mValid)
        {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex        = chosenFamilies.mCompute.mFamilyIndex;
            queueCreateInfo.queueCount              = chosenFamilies.mCompute.mQueueCount;
            queueCreateInfo.pQueuePriorities        = priorities;
            queueCreateInfos.push_back(queueCreateInfo);
        }
        if (chosenFamilies.mTransfer.mValid)
        {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex        = chosenFamilies.mTransfer.mFamilyIndex;
            queueCreateInfo.queueCount              = chosenFamilies.mTransfer.mQueueCount;
            queueCreateInfo.pQueuePriorities        = priorities;
            queueCreateInfos.push_back(queueCreateInfo);
        }
        return queueCreateInfos;
    }

    void GetAvailableDeviceExtensions(VA_PhysicalDeviceDesc& device)
    {
        u32 extensionCount = 0;
        VA_AssertResult(vkEnumerateDeviceExtensionProperties(device.mPhysicalDevice, nullptr, &extensionCount, nullptr),
            "Failed to enumerate device extensions");
        device.mAvailableExtensions.resize(extensionCount);
        VA_AssertResult(vkEnumerateDeviceExtensionProperties(
                            device.mPhysicalDevice, nullptr, &extensionCount, device.mAvailableExtensions.data()),
            "Failed to enumerate device extensions");
        for (const auto& ext : device.mAvailableExtensions)
        {
            device.mAvailableExtensionsNames.push_back(ext.extensionName);
        }
    }

    void FillingDeviceLimits(VkPhysicalDevice device, RhiPropertyList& outProps)
    {
        VkPhysicalDeviceProperties2  properties{};
        VkPhysicalDeviceIDProperties idProperties{};
        properties.sType   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext   = &idProperties;
        idProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        idProperties.pNext = nullptr;
        vkGetPhysicalDeviceProperties2(device, &properties);

        // max descriptors per set
        outProps.mMaxDescriptorsSetCBVBuffer = properties.properties.limits.maxPerStageDescriptorStorageBuffers;
        outProps.mMaxDescriptorsSetSRVBuffer = properties.properties.limits.maxPerStageDescriptorStorageBuffers;
        outProps.mMaxDescriptorsSetUAVBuffer = properties.properties.limits.maxPerStageDescriptorStorageBuffers;
        outProps.mMaxDescriptorsSetSRVImage  = properties.properties.limits.maxPerStageDescriptorSampledImages;
        outProps.mMaxDescriptorsSetUAVImage  = properties.properties.limits.maxPerStageDescriptorStorageImages;
        outProps.mMaxDescriptorsSetSampler   = properties.properties.limits.maxPerStageDescriptorSamplers;

        outProps.mRTColorSamplesSupported = properties.properties.limits.framebufferColorSampleCounts;
        outProps.mRTDepthSamplesSupported = properties.properties.limits.framebufferDepthSampleCounts;
        outProps.mRTSamplesSupported      = outProps.mRTColorSamplesSupported & outProps.mRTDepthSamplesSupported;
    }
} // namespace Ifrit::RHI::VulkanRHI2