#include "ifrit/vkrhi2/adapter/MemoryResource.h"
#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/util/Helpers.h"
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit.internal/vkrhi2/adapter/AllocatorWrapper.h"
#include <vma/vk_mem_alloc.h>

namespace Ifrit::RHI::VulkanRHI2
{
    struct VA_BufferInternal
    {
        VkBuffer          mBuffer;
        VmaAllocation     mAllocation;
        VmaAllocationInfo mAllocInfo;
        u64               mDeviceAddress = 0;
        char*             mMappedMemory  = nullptr;
        bool              mCreated       = false;
    };

    VA_Buffer::VA_Buffer(RhiDevice* device, const RhiBufferDesc& desc) : RhiBuffer(desc)
    {
        auto castedCtx = CheckedCast<VA_Device>(device);

        IF_LOG_ASSERTION("VulkanRHI2", desc.mSize > 0, "Buffer size must be greater than 0");

        mContext        = device;
        mData           = new VA_BufferInternal();
        mData->mCreated = false;

        VkBufferUsageFlags desiredUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        bool               isTransferDst      = false;
        bool               isBufferDeviceAddr = false;
        bool               isHostVisible      = false;

        if (HasFlagBit(desc.mFlags, ERhiBufferUsageFlag::VertexBuffer))
        {
            desiredUsage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        }
        if (HasFlagBit(desc.mFlags, ERhiBufferUsageFlag::IndexBuffer))
        {
            desiredUsage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        }
        if (HasFlagBit(desc.mFlags, ERhiBufferUsageFlag::IndirectBuffer))
        {
            desiredUsage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        }
        if (HasFlagBit(desc.mFlags, ERhiBufferUsageFlag::UniformBuffer))
        {
            isTransferDst = true;
            if (mContext->GetCapabilities().bTreatConstantBufferAsStorageBuffer)
                desiredUsage |= (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            else
                desiredUsage |= (VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        }
        if (HasFlagBit(desc.mFlags, ERhiBufferUsageFlag::UnorderedAccess))
        {
            // we follows unreal engine's convention. unordered access only means RW qualifier
            desiredUsage |= VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
        }
        if (HasFlagBit(desc.mFlags, ERhiBufferUsageFlag::DeviceAddr))
        {
            desiredUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            isBufferDeviceAddr = true;
        }
        if (HasFlagBit(desc.mFlags, ERhiBufferUsageFlag::CopyDst))
        {
            IF_LOG_WARNING("VulkanRHI2", "Buffer CopyDst flag is deprecated");
            desiredUsage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        }
        if (HasFlagBit(desc.mFlags, ERhiBufferUsageFlag::StructuredBuffer))
        {
            desiredUsage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        }
        if (HasFlagBit(desc.mFlags, ERhiBufferUsageFlag::CPUAccess))
        {
            // we follows unreal engine's convention. CPUAccess means staging buffer
            desiredUsage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            isTransferDst = true;
            isHostVisible = true;
        }
        if (HasFlagBit(desc.mFlags, ERhiBufferUsageFlag::Dynamic))
        {
            // we follows unreal engine's convention. Dynamic means frequently updated from CPU
            desiredUsage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            isTransferDst = true;
        }

        VkBufferCreateInfo bufferCI{};
        bufferCI.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCI.size        = desc.mSize;
        bufferCI.usage       = desiredUsage;
        bufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_AUTO;
        if (isHostVisible)
        {
            allocCI.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }
        VA_AssertResult(vmaCreateBuffer(castedCtx->GetAllocator()->mAllocator, &bufferCI, &allocCI, &mData->mBuffer,
                            &mData->mAllocation, &mData->mAllocInfo),
            "Failed to create buffer");

        // get device address
        if (isBufferDeviceAddr)
        {
            VkBufferDeviceAddressInfo bufferDeviceAddrInfo{};
            bufferDeviceAddrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            bufferDeviceAddrInfo.buffer = mData->mBuffer;
            mData->mDeviceAddress       = castedCtx->GetDeviceProcs().p_vkGetBufferDeviceAddress(
                castedCtx->GetVulkanDevice(), &bufferDeviceAddrInfo);
        }

        mData->mCreated = true;
    }

    VA_Buffer::~VA_Buffer()
    {
        if (mData->mCreated)
        {
            auto castedCtx = CheckedCast<VA_Device>(mContext);
            vmaDestroyBuffer(castedCtx->GetAllocator()->mAllocator, mData->mBuffer, mData->mAllocation);
        }
        delete mData;
        mData = nullptr;
    }

} // namespace Ifrit::RHI::VulkanRHI2