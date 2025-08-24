#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/core/typing/Traits.h"
#include "ifrit.internal/vkrhi2/adapter/DeviceUtils.h"
#include "ifrit.internal/vkrhi2/adapter/DeviceExtensions.h"
#include "ifrit.internal/vkrhi2/adapter/AllocatorWrapper.h"
#include <vma/vk_mem_alloc.h>

namespace Ifrit::RHI::VulkanRHI2
{

    // ===== ResourceDeleteQueue Implementation =====
    IFRIT_APIDECL i32 ResourceDeleteQueue::ProcessDeleteQueue()
    {
        i32 count = 0;
        while (!mDeleteQueue.empty())
        {
            auto resource = mDeleteQueue.front();
            if (mFrameIdToDelete.front() > mCurrentFrameStep)
                break;
            mDeleteQueue.pop();
            mFrameIdToDelete.pop();
            if (!resource->GetDebugName().empty())
                IF_LOG_DEBUG("Device", "Deleting resource: {}", resource->GetDebugName());
            delete resource;
            count++;
        }
        mCurrentFrameStep++;
        return count;
    }

    IFRIT_APIDECL void ResourceDeleteQueue::AddResourceToDeleteQueue(RHI::RhiDeviceResource* resource)
    {
        ScopedLock lock(mLock);
        mDeleteQueue.push(resource);
        mFrameIdToDelete.push(mCurrentFrameStep + 2);
    }

    // ===== Validation Layer =====
    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
        if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        {
            // TODO: This is a temporarily patch for spirv-val problem for slang bindless
            if (pCallbackData->messageIdNumber == -1307510846 || pCallbackData->messageIdNumber == -1520283006)
            {
                return VK_FALSE;
            }
            if (pCallbackData->messageIdNumber != 0x8ebf0028)
            {
                IF_LOG_CRITICAL("VA_Device", "Aborting due to validation layer error: {}", pCallbackData->pMessage);
            }
        }
        else
        {
            IF_LOG_WARNING("VA_Device", "Validation layer called");
            IF_LOG_WARNING("VA_Device", "{}", pCallbackData->pMessage);
        }

        return VK_FALSE;
    }

    // ===== Device Implementation =====
    struct VA_DevicePrivate
    {
        RHI::RhiInitializeArguments           mArgs;
        ResourceDeleteQueue                   mDeleteQueue;

        VkInstance                            mInstance       = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT              mDebugMessenger = VK_NULL_HANDLE;
        VkDevice                              mDevice         = VK_NULL_HANDLE;

        VA_PhysicalDeviceDesc                 mPhysicalDevice = {};
        VA_ChosenQueueFamily                  mQueueInfo      = {};
        VA_DeviceProcs                        mProcs          = {};

        RhiCapabilityList                     mCapabilities = {};
        RhiPropertyList                       mProperties   = {};

        VmaAllocator                          mAllocator;
        VA_Allocator                          mAllocatorWrapper = {};

        HashMap<VkFormat, VkFormatProperties> mFormatPropertiesCache;

        u64                                   mFrameId = 0;
    };

    IFRIT_APIDECL VA_Device::VA_Device(const RHI::RhiInitializeArguments& args)
    {
        mData        = new VA_DevicePrivate();
        mData->mArgs = args;
    }

    IFRIT_APIDECL VA_Device::~VA_Device()
    {
        delete mData;
        mData = nullptr;
    }

    IFRIT_APIDECL void VA_Device::Init()
    {
        // Create Application

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

        // Instance extensions
        auto             availableInstanceExtensions = EnumerateAvailableExtensions();
        Vec<const char*> targetInstanceExtensions;
        if (mData->mArgs.mExtensionGetter)
        {
            u32          extensionCountExtra = 0;
            const char** extensionsExtra     = mData->mArgs.mExtensionGetter(&extensionCountExtra);
            for (u32 i = 0; i < extensionCountExtra; i++)
            {
                EnableExtension(true, extensionsExtra[i], availableInstanceExtensions, targetInstanceExtensions);
            }
        }
        if (mData->mArgs.mDesiredCapabilities.bValidationLayerEnabled)
        {
            EnableExtension(
                true, VK_EXT_DEBUG_UTILS_EXTENSION_NAME, availableInstanceExtensions, targetInstanceExtensions);
        }
        for (auto ext : kRequiredInstanceExtension)
        {
            EnableExtension(false, ext, availableInstanceExtensions, targetInstanceExtensions);
        }
        instanceCI.enabledExtensionCount   = SizeCast<u32>(targetInstanceExtensions.size());
        instanceCI.ppEnabledExtensionNames = targetInstanceExtensions.data();

        // Instance Layers
        auto             availableInstanceLayers = EnumerateAvailableLayers();
        Vec<const char*> targetInstanceLayers;
        if (mData->mArgs.mDesiredCapabilities.bValidationLayerEnabled)
        {
            EnableLayer(true, kValidationLayerName, availableInstanceLayers, targetInstanceLayers);
        }
        instanceCI.enabledLayerCount   = SizeCast<u32>(targetInstanceLayers.size());
        instanceCI.ppEnabledLayerNames = targetInstanceLayers.data();

        // Instance Create
        VA_AssertResult(vkCreateInstance(&instanceCI, nullptr, &mData->mInstance), "Failed to create Vulkan instance");

        // Setup Validation Messenger
        if (mData->mArgs.mDesiredCapabilities.bValidationLayerEnabled)
        {
            SetupValidationMessenger(DebugCallback, mData->mInstance, mData->mDebugMessenger);
        }

        // Physical Device
        {
            auto availablePhysicalDevices = EnumeratePhysicalDevices(mData->mInstance);
            auto physicalDeviceCriteria   = VA_PhysicalDeviceCriteria{
                  .mPreferredAdapterId = mData->mArgs.mPreferredGraphcisAdapterId,
                  .mPreferredVendor    = mData->mArgs.mPreferredVendor,
            };
            mData->mPhysicalDevice = SelectPhysicalDevice(availablePhysicalDevices, physicalDeviceCriteria);
        }

        // Queue Family
        Vec<VkDeviceQueueCreateInfo> queueCreateInfos;
        {
            mData->mQueueInfo =
                SelectQueueFamilies(mData->mPhysicalDevice, mData->mArgs.mDesiredCapabilities.bAsyncComputeEnable);
            queueCreateInfos = CreateQueueCreateInfos(mData->mQueueInfo);
        }

        // Prepare Device Extensions
        GetAvailableDeviceExtensions(mData->mPhysicalDevice);
        auto               extensionData = PrepareDeviceExtension(mData->mPhysicalDevice.mPhysicalDevice,
                          mData->mPhysicalDevice.mAvailableExtensionsNames, mData->mCapabilities, mData->mProperties);

        // Device Create
        VkDeviceCreateInfo deviceCI      = {};
        deviceCI.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCI.queueCreateInfoCount    = SizeCast<u32>(queueCreateInfos.size());
        deviceCI.pQueueCreateInfos       = queueCreateInfos.data();
        deviceCI.enabledExtensionCount   = extensionData.mEnabledExtensions.size();
        deviceCI.ppEnabledExtensionNames = extensionData.mEnabledExtensions.data();
        deviceCI.pEnabledFeatures        = extensionData.mBaseFeatures;
        deviceCI.pNext                   = extensionData.mExtensionChain;
        VA_AssertResult(vkCreateDevice(mData->mPhysicalDevice.mPhysicalDevice, &deviceCI, nullptr, &mData->mDevice),
            "Failed to create Vulkan device");

        // Device Procs
        LoadDeviceProcs(mData->mDevice);

        // Allocator
        VmaAllocatorCreateInfo allocatorCI = {};
        allocatorCI.physicalDevice         = mData->mPhysicalDevice.mPhysicalDevice;
        allocatorCI.device                 = mData->mDevice;
        allocatorCI.instance               = mData->mInstance;
        allocatorCI.vulkanApiVersion       = VK_API_VERSION_1_3;
        allocatorCI.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        VA_AssertResult(
            vmaCreateAllocator(&allocatorCI, &mData->mAllocator), "Failed to create Vulkan memory allocator");

        mData->mAllocatorWrapper.mAllocator = mData->mAllocator;
    }

    IFRIT_APIDECL IRhiDeviceResourceDeleteQueue* VA_Device::GetDeleteQueue() { return &mData->mDeleteQueue; }
    IFRIT_APIDECL VA_Allocator*                  VA_Device::GetAllocator() { return &mData->mAllocatorWrapper; }
    IFRIT_APIDECL VA_DeviceProcs&                VA_Device::GetDeviceProcs() const { return mData->mProcs; }
    IFRIT_APIDECL VkDevice                       VA_Device::GetVulkanDevice() const { return mData->mDevice; }
    IFRIT_APIDECL VkFormatProperties             VA_Device::GetFormatProperties(VkFormat format) const
    {
        auto it = mData->mFormatPropertiesCache.find(format);
        if (it != mData->mFormatPropertiesCache.end()) IF_LIKELY
        {
            return it->second;
        }
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(mData->mPhysicalDevice.mPhysicalDevice, format, &props);
        mData->mFormatPropertiesCache[format] = props;
        return props;
    }
    IFRIT_APIDECL VA_ActiveQueueFamilyInfo VA_Device::GetActiveQueueFamilies() const
    {
        VA_ActiveQueueFamilyInfo info;
        info.mGraphics     = mData->mQueueInfo.mGraphics.mFamilyIndex;
        info.mAsyncCompute = mData->mQueueInfo.mCompute.mFamilyIndex;
        info.mTransfer     = mData->mQueueInfo.mTransfer.mFamilyIndex;
        return info;
    }
    IFRIT_APIDECL u64 VA_Device::GetFrameId() const { return mData->mFrameId; }
} // namespace Ifrit::RHI::VulkanRHI2