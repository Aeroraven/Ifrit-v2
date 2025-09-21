#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/vkrhi2/adapter/Queue.h"
#include "ifrit/vkrhi2/adapter/CommandContext.h"
#include "ifrit/core/typing/Traits.h"
#include "ifrit.internal/vkrhi2/adapter/DeviceUtils.h"
#include "ifrit.internal/vkrhi2/adapter/DeviceExtensions.h"
#include "ifrit.internal/vkrhi2/adapter/AllocatorWrapper.h"
#include "ifrit/vkrhi2/adapter/MemoryResource.h"
#include "ifrit/core/tasks/TaskScheduler.h"
#include "ifrit/vkrhi2/adapter/DynamicUtils.h"
#include "ifrit/vkrhi2/adapter/DescriptorHeap.h"
#include "ifrit/vkrhi2/adapter/PipelineState.h"
#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>

namespace Ifrit::RHI::VulkanRHI2
{

    // ===== ResourceDeleteQueue Implementation =====
    IFRIT_APIDECL i32 ResourceDeleteQueue::ProcessDeleteQueue()
    {
        ScopedLock lock(mLock);
        i32        count = 0;
        while (!mDeleteQueue.empty())
        {
            auto resource = mDeleteQueue.front();
            if (mFrameIdToDelete.front() > mCurrentFrameStep)
                break;
            mDeleteQueue.pop();
            mFrameIdToDelete.pop();
            // if (!resource->GetDebugName().empty())
            //     IF_LOG_DEBUG("Device", "Deleting resource: {}", resource->GetDebugName());
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
    IFRIT_APIDECL void ResourceDeleteQueue::RemoveAllResources()
    {
        ScopedLock lock(mLock);
        while (!mDeleteQueue.empty())
        {
            auto resource = mDeleteQueue.front();
            mDeleteQueue.pop();
            mFrameIdToDelete.pop();
            delete resource;
        }
    }
    // ===== Validation Layer =====
    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallbackFunc(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
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

    struct VA_DeviceQueueInfo
    {
        Owner<VA_Queue> mGraphics;
        Owner<VA_Queue> mAsyncCompute;
        Owner<VA_Queue> mTransfer;
    };

    struct VA_DevicePrivate
    {
        RHI::RhiInitializeArguments            mArgs;
        ResourceDeleteQueue                    mDeleteQueue;

        VkInstance                             mInstance       = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT               mDebugMessenger = VK_NULL_HANDLE;
        VkDevice                               mDevice         = VK_NULL_HANDLE;

        VA_PhysicalDeviceDesc                  mPhysicalDevice = {};
        VA_ChosenQueueFamily                   mQueueInfo      = {};
        VA_DeviceProcs                         mProcs          = {};
        VA_DeviceQueueInfo                     mActiveQueues   = {};
        Owner<VA_DynamicUtils>                 mDynamicUtils   = nullptr;

        RhiCapabilityList                      mCapabilities = {};
        RhiPropertyList                        mProperties   = {};

        VmaAllocator                           mAllocator;
        VA_Allocator                           mAllocatorWrapper = {};

        THashMap<VkFormat, VkFormatProperties> mFormatPropertiesCache;

        u64                                    mFrameId = 0;

        // Pipeline State Cache
        Owner<VA_PipelineStateCacheRegistry>   mPipelineStateCache;

        // Descriptor Heap
        Owner<VA_BindlessDescriptorHeap>       mBindlessDescriptorHeap;

        // Staging Buffer
        Owner<VA_StagingBufferManager>         mStagingBufferManager;
        Owner<VA_SamplerRegistry>              mSamplerRegistry;

        // Commands
        Owner<VA_CommandListContext>           mImmediateContext;
    };

    IFRIT_APIDECL VA_Device::VA_Device(const RHI::RhiInitializeArguments& args)
    {
        mData        = new VA_DevicePrivate();
        mData->mArgs = args;
        Init();
    }

    IFRIT_APIDECL VA_Device::~VA_Device()
    {
        Shutdown();
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
            SetupValidationMessenger(DebugCallbackFunc, mData->mInstance, mData->mDebugMessenger);
        }

        // Physical Device
        {
            auto availablePhysicalDevices = EnumeratePhysicalDevices(mData->mInstance);
            auto physicalDeviceCriteria   = VA_PhysicalDeviceCriteria{
                  .mPreferredAdapterId = mData->mArgs.mPreferredGraphcisAdapterId,
                  .mPreferredVendor    = mData->mArgs.mPreferredVendor,
            };
            mData->mPhysicalDevice = SelectPhysicalDevice(availablePhysicalDevices, physicalDeviceCriteria);
            IF_LOG_INFO("VA_Device", "Selected physical device: {}",
                String(mData->mPhysicalDevice.mProperties.properties.deviceName));
            FillingDeviceLimits(mData->mPhysicalDevice.mPhysicalDevice, mData->mProperties);
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
        deviceCI.enabledExtensionCount   = SizeCast<u32>(extensionData.mEnabledExtensions.size());
        deviceCI.ppEnabledExtensionNames = extensionData.mEnabledExtensions.data();
        deviceCI.pEnabledFeatures        = extensionData.mBaseFeatures;
        deviceCI.pNext                   = extensionData.mExtensionChain;
        VA_AssertResult(vkCreateDevice(mData->mPhysicalDevice.mPhysicalDevice, &deviceCI, nullptr, &mData->mDevice),
            "Failed to create Vulkan device");
        IF_LOG_INFO("VA_Device", "Vulkan device created");

        // Dynamic Utils
        mData->mDynamicUtils = MakeOwner<VA_DynamicUtils>(this);

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

        // Create Queues
        mData->mActiveQueues.mGraphics =
            MakeOwner<VA_Queue>(this, ERhiCommandListPipelineType::Graphics, mData->mQueueInfo.mGraphics.mFamilyIndex);
        mData->mActiveQueues.mAsyncCompute =
            MakeOwner<VA_Queue>(this, ERhiCommandListPipelineType::Compute, mData->mQueueInfo.mCompute.mFamilyIndex);
        mData->mActiveQueues.mTransfer =
            MakeOwner<VA_Queue>(this, ERhiCommandListPipelineType::Transfer, mData->mQueueInfo.mTransfer.mFamilyIndex);

        // Create Immediate Context
        mData->mImmediateContext =
            MakeOwner<VA_CommandListContext>(this, mData->mActiveQueues.mGraphics.get(), nullptr);

        // Staging Buffer Manager
        mData->mStagingBufferManager = MakeOwner<VA_StagingBufferManager>(this);

        // Bindless Descriptor Heap
        mData->mBindlessDescriptorHeap = MakeOwner<VA_BindlessDescriptorHeap>(this);

        // Sampler Registry
        mData->mSamplerRegistry = MakeOwner<VA_SamplerRegistry>(this);

        // Pipeline State Cache
        mData->mPipelineStateCache = MakeOwner<VA_PipelineStateCacheRegistry>(this);

        // Command Submission Thread
        {
            auto scheduler        = Task::GetTaskScheduler();
            auto submissionThread = MakeOwner<VA_QueueSubmissionThread>(scheduler, this, 2333);
            scheduler->RegisterNamedWorker(std::move(submissionThread), Task::ENamedTaskThread::RHISubmissionThread);
        }

        IF_LOG_INFO("VA_Device", "Initialized VulkanRHI2 device");
    }

    IFRIT_APIDECL void VA_Device::Shutdown()
    {
        // Terminating
        auto taskScheduler = Task::GetTaskScheduler();
        taskScheduler->RequestTerminating(Task::ENamedTaskThread::RHISubmissionThread);
        taskScheduler->WaitForTerminating(Task::ENamedTaskThread::RHISubmissionThread);

        // Destroy Immediate Context
        IF_LOG_INFO("VA_Device", "Removing VulkanRHI2 device");
        mData->mImmediateContext = nullptr;
        mData->mActiveQueues     = {};

        // Destroy Resources
        vkDeviceWaitIdle(mData->mDevice);
        mData->mDeleteQueue.RemoveAllResources();

        mData->mPipelineStateCache     = nullptr;
        mData->mStagingBufferManager   = nullptr;
        mData->mSamplerRegistry        = nullptr;
        mData->mBindlessDescriptorHeap = nullptr;

        // Then wait for shader unload
        vkDeviceWaitIdle(mData->mDevice);
        mData->mDeleteQueue.RemoveAllResources();

        vmaDestroyAllocator(mData->mAllocator);
        vkDestroyDevice(mData->mDevice, nullptr);
        if (mData->mArgs.mDesiredCapabilities.bValidationLayerEnabled)
        {
            ShutdownValidationMessenger(mData->mInstance, mData->mDebugMessenger);
        }
        vkDestroyInstance(mData->mInstance, nullptr);
        IF_LOG_INFO("VA_Device", "Shutdown VulkanRHI2 device");
    }

    IFRIT_APIDECL IRhiDeviceResourceDeleteQueue* VA_Device::GetResourceDeleteQueue() { return &mData->mDeleteQueue; }
    IFRIT_APIDECL VA_Allocator*                  VA_Device::GetAllocator() { return &mData->mAllocatorWrapper; }
    IFRIT_APIDECL VA_DeviceProcs&                VA_Device::GetDeviceProcs() const { return mData->mProcs; }
    IFRIT_APIDECL VkDevice                       VA_Device::GetVulkanDevice() const { return mData->mDevice; }
    IFRIT_APIDECL RhiInitializeArguments         VA_Device::GetInitializationArgs() const { return mData->mArgs; }
    IFRIT_APIDECL VkInstance                     VA_Device::GetVulkanInstance() const { return mData->mInstance; }
    IFRIT_APIDECL VkPhysicalDevice               VA_Device::GetVulkanPhysicalDevice() const
    {
        return mData->mPhysicalDevice.mPhysicalDevice;
    }
    IFRIT_APIDECL void* VA_Device::GetVmaAllocator() const { return reinterpret_cast<void*>(mData->mAllocator); }
    IFRIT_APIDECL VA_CommandListContext* VA_Device::GetImmediateContext() const
    {
        return mData->mImmediateContext.get();
    }
    IFRIT_APIDECL Owner<VA_CommandListContext> VA_Device::GetUploadContext()
    {
        return MakeOwner<VA_CommandListContext>(this, mData->mActiveQueues.mGraphics.get(), nullptr);
    }
    IFRIT_APIDECL Owner<VA_CommandListContext> VA_Device::GetCommandContext(ERhiCommandListPipelineType type)
    {
        switch (type)
        {
            case ERhiCommandListPipelineType::Graphics:
                return MakeOwner<VA_CommandListContext>(this, mData->mActiveQueues.mGraphics.get(), nullptr);
            case ERhiCommandListPipelineType::Compute:
                return MakeOwner<VA_CommandListContext>(this, mData->mActiveQueues.mAsyncCompute.get(), nullptr);
            case ERhiCommandListPipelineType::Transfer:
                return MakeOwner<VA_CommandListContext>(this, mData->mActiveQueues.mTransfer.get(), nullptr);
            default:
                IF_LOG_ASSERTION("VA_Device", false, "Unsupported command list type");
                return nullptr;
        }
    }
    IFRIT_APIDECL VA_StagingBufferManager* VA_Device::GetStagingBufferManager()
    {
        return mData->mStagingBufferManager.get();
    }
    IFRIT_APIDECL VkFormatProperties VA_Device::GetFormatProperties(VkFormat format) const
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

    IFRIT_APIDECL VA_ActiveQueueInfo VA_Device::GetActiveQueues() const
    {
        VA_ActiveQueueInfo info;
        info.mGraphics     = mData->mActiveQueues.mGraphics.get();
        info.mAsyncCompute = mData->mActiveQueues.mAsyncCompute.get();
        info.mTransfer     = mData->mActiveQueues.mTransfer.get();
        return info;
    }

    IFRIT_APIDECL u64                     VA_Device::GetFrameId() const { return mData->mFrameId; }

    IFRIT_APIDECL RhiCapabilityList       VA_Device::GetCapabilities() const { return mData->mCapabilities; }
    IFRIT_APIDECL RhiPropertyList         VA_Device::GetProperties() const { return mData->mProperties; }

    IFRIT_APIDECL RhiCommandListExecutor* VA_Device::GetCommandListExecutor() const { return nullptr; }
    IFRIT_APIDECL RhiDynamicUtils* VA_Device::GetDeviceRHIFunctions() const { return mData->mDynamicUtils.get(); }

    IFRIT_APIDECL void             VA_Device::SetupPresentQueue(VkSurfaceKHR surface)
    {
        if (mData->mQueueInfo.mGraphics.mSupportsPresent)
        {
            return;
        }
        u32 queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(mData->mPhysicalDevice.mPhysicalDevice, &queueFamilyCount, nullptr);
        Vec<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(
            mData->mPhysicalDevice.mPhysicalDevice, &queueFamilyCount, queueFamilies.data());
        for (u32 i = 0; i < queueFamilyCount; i++)
        {
            VkBool32 supportsPresent = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(mData->mPhysicalDevice.mPhysicalDevice, i, surface, &supportsPresent);
            if (supportsPresent)
            {
                IF_LOG_INFO("VA_Device", "Selected present queue family: {}", i);
                mData->mQueueInfo.mGraphics.mFamilyIndex     = i;
                mData->mQueueInfo.mGraphics.mSupportsPresent = true;
                return;
            }
        }
        IF_LOG_CRITICAL("VA_Device", "No suitable present queue found");
    }

    IFRIT_APIDECL VA_Queue* VA_Device::GetPresentQueue()
    {
        if (!mData->mQueueInfo.mGraphics.mSupportsPresent)
        {
            IF_LOG_CRITICAL("VA_Device", "Present queue not set up. Call SetupPresentQueue first.");
            return nullptr;
        }
        return mData->mActiveQueues.mGraphics.get();
    }
    IFRIT_APIDECL void VA_Device::WaitIdle() { vkDeviceWaitIdle(mData->mDevice); }

    IFRIT_APIDECL void VA_Device::CpuWaitForAllQueuedTaskSubmission()
    {
        auto graphicsWaitTask     = VA_CommandTask::CreateCpuWaitTask();
        auto asyncComputeWaitTask = VA_CommandTask::CreateCpuWaitTask();
        auto transferWaitTask     = VA_CommandTask::CreateCpuWaitTask();
        mData->mActiveQueues.mGraphics->EnqueueCommandTask(graphicsWaitTask);
        mData->mActiveQueues.mAsyncCompute->EnqueueCommandTask(asyncComputeWaitTask);
        mData->mActiveQueues.mTransfer->EnqueueCommandTask(transferWaitTask);
        graphicsWaitTask->Wait();
        asyncComputeWaitTask->Wait();
        transferWaitTask->Wait();
    }

    IFRIT_APIDECL void VA_Device::FrameAdvance()
    {
        mData->mFrameId++;
        mData->mDeleteQueue.ProcessDeleteQueue();
    }

    IFRIT_APIDECL bool VA_Device::IsDebugMode() const
    {
        return mData->mArgs.mDesiredCapabilities.bValidationLayerEnabled;
    }

    IFRIT_APIDECL String                     VA_Device::GetCacheDir() const { return mData->mArgs.mCachePath; }
    IFRIT_APIDECL VA_BindlessDescriptorHeap* VA_Device::GetBindlessDescriptorHeap()
    {
        return mData->mBindlessDescriptorHeap.get();
    }

    IFRIT_APIDECL VA_SamplerRegistry* VA_Device::GetSamplerRegistry() { return mData->mSamplerRegistry.get(); }

    IFRIT_APIDECL VA_PipelineStateCacheRegistry* VA_Device::GetPipelineStateCache()
    {
        return mData->mPipelineStateCache.get();
    }

} // namespace Ifrit::RHI::VulkanRHI2