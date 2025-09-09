
#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/vkrhi2/adapter/DeviceProcs.h"
#include "ifrit/core/algo/Parallel.h"
#include "ifrit/vkrhi2/adapter/Queue.h"

#include <vulkan/vulkan.h>
#ifdef _WIN32
    #include <Windows.h>
#endif

namespace Ifrit::RHI::VulkanRHI2
{
    struct VA_Allocator;
    class VA_Queue;
    class VA_CommandListContext;
    class VA_StagingBufferManager;
    class VA_SamplerRegistry;
    class VA_BindlessDescriptorHeap;
    class VA_PipelineStateCacheRegistry;

    enum class EVA_QueueType : u32
    {
        Graphics,
        AsyncCompute,
        Transfer,
    };

    class IFRIT_VKRHI2_API ResourceDeleteQueue : public RHI::IRhiDeviceResourceDeleteQueue, public NonCopyable
    {
    public:
        virtual void AddResourceToDeleteQueue(RHI::RhiDeviceResource* resource);
        virtual i32  ProcessDeleteQueue();
        virtual ~ResourceDeleteQueue() { ProcessDeleteQueue(); }

        void RemoveAllResources();

    private:
        u64                            mCurrentFrameStep = 0;
        Queue<RHI::RhiDeviceResource*> mDeleteQueue;
        Queue<u64>                     mFrameIdToDelete;
        Mutex                          mLock;
    };

    struct VA_ActiveQueueFamilyInfo
    {
        u32 mGraphics     = ~0u;
        u32 mAsyncCompute = ~0u;
        u32 mTransfer     = ~0u;
    };

    struct VA_ActiveQueueInfo
    {
        VA_Queue* mGraphics     = nullptr;
        VA_Queue* mAsyncCompute = nullptr;
        VA_Queue* mTransfer     = nullptr;
    };

    struct VA_DevicePrivate;
    class IFRIT_VKRHI2_API VA_Device : public RHI::RhiDevice, public NonCopyable
    {
    public:
        VA_Device(const RHI::RhiInitializeArguments& args);
        ~VA_Device();

    public: // RHI Overrides
        virtual RhiCapabilityList              GetCapabilities() const override;
        virtual RhiPropertyList                GetProperties() const override;
        virtual IRhiDeviceResourceDeleteQueue* GetResourceDeleteQueue() override;
        virtual RhiDynamicUtils*               GetDeviceRHIFunctions() const override;
        virtual RhiCommandListExecutor*        GetCommandListExecutor() const override;
        virtual String                         GetCacheDir() const override;

    public:
        VA_Allocator*                  GetAllocator();
        VA_DeviceProcs&                GetDeviceProcs() const;
        VA_ActiveQueueFamilyInfo       GetActiveQueueFamilies() const;
        VA_ActiveQueueInfo             GetActiveQueues() const;
        VA_SamplerRegistry*            GetSamplerRegistry();

        VA_CommandListContext*         GetImmediateContext() const;
        Owner<VA_CommandListContext>   GetUploadContext();
        Owner<VA_CommandListContext>   GetCommandContext(ERhiCommandListPipelineType type);
        VA_StagingBufferManager*       GetStagingBufferManager();
        RhiInitializeArguments         GetInitializationArgs() const;
        VA_BindlessDescriptorHeap*     GetBindlessDescriptorHeap();
        VA_PipelineStateCacheRegistry* GetPipelineStateCache();

        VA_Queue*                      GetPresentQueue();

        void                           CpuWaitForAllQueuedTaskSubmission();

        void                           FrameAdvance();

        // Vulkan specific
        VkDevice                       GetVulkanDevice() const;
        VkFormatProperties             GetFormatProperties(VkFormat format) const;
        u64                            GetFrameId() const;
        VkInstance                     GetVulkanInstance() const;
        VkPhysicalDevice               GetVulkanPhysicalDevice() const;
        void                           SetupPresentQueue(VkSurfaceKHR surface);
        void                           WaitIdle();

        // Utility
        bool                           IsDebugMode() const;

    private:
        void Init();
        void Shutdown();

    private:
        VA_DevicePrivate* mData;
    };

} // namespace Ifrit::RHI::VulkanRHI2