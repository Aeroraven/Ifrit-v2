#include "ifrit/vkrhi2/adapter/Backend.h"
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/adapter/CommandContext.h"

namespace Ifrit::RHI::VulkanRHI2
{
    struct VA_BackendInternal
    {
        bool             mSetup = false;
        Owner<VA_Device> mDevice;
    };

    IFRIT_VKRHI2_API VA_Backend::VA_Backend() { mInternal = new VA_BackendInternal(); }

    IFRIT_VKRHI2_API VA_Backend::~VA_Backend()
    {
        if (mInternal->mSetup)
            Finalize();
        delete mInternal;
        mInternal = nullptr;
    }

    IFRIT_VKRHI2_API void VA_Backend::Init(const RhiInitializeArguments& args)
    {
        mInternal->mDevice = MakeOwner<VA_Device>(args);

        mInternal->mSetup = true;
    }

    IFRIT_VKRHI2_API void VA_Backend::Finalize()
    {
        mInternal->mDevice = nullptr;
        mInternal->mSetup  = false;
    }

    IFRIT_VKRHI2_API VA_Device*          VA_Backend::GetDevice() const { return mInternal->mDevice.get(); }
    IFRIT_VKRHI2_API IRhiCommandContext* VA_Backend::GetImmediateContext()
    {
        auto ret = mInternal->mDevice->GetImmediateContext();
        return ret;
    }

} // namespace Ifrit::RHI::VulkanRHI2
