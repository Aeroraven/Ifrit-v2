#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"

namespace Ifrit::RHI::VulkanRHI2
{

    class VA_Device;

    struct VA_BackendInternal;
    class IFRIT_VKRHI2_API VA_Backend : public RhiBackend
    {
    public:
        VA_Backend();
        ~VA_Backend();

        // RhiBackend Overrides
        void                Init(const RhiInitializeArguments& args) override final;
        void                Finalize() override final;

        IRhiCommandContext* GetImmediateContext() override final;

        // VA_Backend specific
        VA_Device*          GetDevice() const;

    private:
        VA_BackendInternal* mInternal;
    };

} // namespace Ifrit::RHI::VulkanRHI2