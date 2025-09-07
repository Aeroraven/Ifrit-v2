#include "ifrit/vkrhi2/adapter/MemoryResource.h"
#include "ifrit/vkrhi2/adapter/DescriptorHeap.h"
#include "ifrit/vkrhi2/adapter/Device.h"

namespace Ifrit::RHI::VulkanRHI2
{
    void VA_Sampler::RegisterSamplerDescriptor()
    {
        auto device    = CheckedCast<VA_Device>(mContext);
        auto heap      = device->GetBindlessDescriptorHeap();
        u32  idx       = heap->RegisterSampler(*this);
        mHandle.mType  = ERhiDescriptorHeapType::Sampler;
        mHandle.mIndex = idx;
    }
    void VA_Sampler::UnregisterSamplerDescriptor()
    {
        auto device = CheckedCast<VA_Device>(mContext);
        auto heap   = device->GetBindlessDescriptorHeap();
        if (mHandle.mIndex != ~0u)
        {
            heap->FreeSampler(mHandle.mIndex);
            mHandle.mIndex = ~0u;
        }
    }
} // namespace Ifrit::RHI::VulkanRHI2