
#pragma once
#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
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

    struct VA_DevicePrivate;
    class IFRIT_VKRHI2_API VA_Device final : public RHI::RhiDevice, public NonCopyable
    {
    public:
        VA_Device(const RHI::RhiInitializeArguments& args);
        ~VA_Device();

        IRhiDeviceResourceDeleteQueue* GetDeleteQueue();

        VA_Allocator*                  GetAllocator();
        VA_DeviceProcs&                GetDeviceProcs() const;
        VA_ActiveQueueFamilyInfo       GetActiveQueueFamilies() const;

        // Vulkan specific
        VkDevice                       GetVulkanDevice() const;
        VkFormatProperties             GetFormatProperties(VkFormat format) const;
        u64                            GetFrameId() const;

    private:
        void Init();

    private:
        VA_DevicePrivate* mData;
    };

} // namespace Ifrit::RHI::VulkanRHI2