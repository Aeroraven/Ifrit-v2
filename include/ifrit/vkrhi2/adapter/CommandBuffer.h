#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/vkrhi2/adapter/Device.h"

namespace Ifrit::RHI::VulkanRHI2
{

    enum class EVA_CommandListNativeState
    {
        Undefined,
        ReadyToBegin,
        Recording,
        ReadyToSubmit,
        Submitted
    };

    struct VA_CommandListNativeInternal;
    class IFRIT_VKRHI2_API VA_CommandListNative
    {
    public:
        VA_CommandListNative(VkCommandBuffer cmd, VA_Device* device);
        ~VA_CommandListNative();

        VkCommandBuffer            GetCmd();
        EVA_CommandListNativeState GetState();

        void                       Begin();
        void                       End();

        u64                        GetSubmitTimestamp() const;
        void                       SetSubmitState();
        void                       ForceSetToReadyState();

    private:
        VA_CommandListNativeInternal* mInternal;
    };

    struct VA_CommandListPoolInternal;
    class IFRIT_VKRHI2_API VA_CommandListPool
    {
    public:
        VA_CommandListPool(VA_Device* device, ERhiCommandListPipelineType type, VkQueue queue, u32 familyIndex);
        ~VA_CommandListPool();

        void                  RecycleCommandBuffers();
        VA_CommandListNative* AllocateCommandBuffer();

    private:
        VA_CommandListPoolInternal* mInternal;
    };

} // namespace Ifrit::RHI::VulkanRHI2