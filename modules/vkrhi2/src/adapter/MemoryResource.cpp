#include "ifrit/vkrhi2/adapter/MemoryResource.h"
#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/util/Helpers.h"
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit.internal/vkrhi2/adapter/AllocatorWrapper.h"
#include "ifrit/core/typing/EnumReflection.h"
#include "ifrit.internal/vkrhi2/adapter/TextureUtils.h"
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

    struct VA_TextureInternal
    {
        VkImage           mImage;
        VmaAllocation     mAllocation;
        VmaAllocationInfo mAllocInfo;
        bool              mCreated = false;
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

    VA_Texture::VA_Texture(RhiDevice* device, const RhiTextureDesc& desc) : RhiTexture(desc)
    {
        auto              castedCtx = CheckedCast<VA_Device>(device);

        VkImageCreateInfo imageCI{};
        imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;

        // select image type
        switch (desc.mDimension)
        {
            case ERhiImageDimension::Texture2D:
                imageCI.imageType = VK_IMAGE_TYPE_2D;
                break;
            case ERhiImageDimension::Texture2DArray:
                imageCI.imageType = VK_IMAGE_TYPE_2D;
                break;
            case ERhiImageDimension::Texture3D:
                imageCI.imageType = VK_IMAGE_TYPE_3D;
                break;
            case ERhiImageDimension::TextureCube:
                imageCI.imageType = VK_IMAGE_TYPE_2D;
                imageCI.flags     = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
                break;
            case ERhiImageDimension::TextureCubeArray:
                imageCI.imageType = VK_IMAGE_TYPE_2D;
                imageCI.flags     = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
                break;
            default:
                IF_LOG_ASSERTION("VulkanRHI2", false, "Unsupported texture dimension");
        }

        // translate format
        auto fnFmtTranslator = [](ERhiImageFormat fmt) -> VkFormat {
            IF_LOG_ASSERTION("VulkanRHI2", fmt != ERhiImageFormat::Undefined, "Invalid image format");
            return static_cast<VkFormat>(fmt);
        };
        imageCI.format = fnFmtTranslator(desc.mFormat);

        // extent
        imageCI.extent.width  = desc.mWidth;
        imageCI.extent.height = desc.mHeight;
        imageCI.extent.depth  = desc.mDepth;

        // samples
        switch (desc.mSamples)
        {
            case 1:
                imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
                break;
            case 2:
                imageCI.samples = VK_SAMPLE_COUNT_2_BIT;
                break;
            case 4:
                imageCI.samples = VK_SAMPLE_COUNT_4_BIT;
                break;
            case 8:
                imageCI.samples = VK_SAMPLE_COUNT_8_BIT;
                break;
            case 16:
                imageCI.samples = VK_SAMPLE_COUNT_16_BIT;
                break;
            case 32:
                imageCI.samples = VK_SAMPLE_COUNT_32_BIT;
                break;
            case 64:
                imageCI.samples = VK_SAMPLE_COUNT_64_BIT;
                break;
            default:
                IF_LOG_ASSERTION("VulkanRHI2", false, "Unsupported sample count");
        }

        // mip levels
        imageCI.mipLevels   = desc.mMips;
        imageCI.arrayLayers = desc.mArraySize;
        imageCI.tiling      = VK_IMAGE_TILING_OPTIMAL;
        imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        // usage
        auto desiredUsage  = TranslateCreateFlags(castedCtx, desc.mUsage);
        auto fmtProperties = castedCtx->GetFormatProperties(imageCI.format);
        auto tilingProps   = fmtProperties.optimalTilingFeatures;

        if (!HasFlagBit(tilingProps, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
        {
            IF_LOG_ASSERTION("VulkanRHI2", HasFlagBit(desc.mUsage, ERhiImageUsageFlag::UnorderedAccess),
                "The format does not support sampled image usage");
            imageCI.usage &= ~VK_IMAGE_USAGE_SAMPLED_BIT;
        }
        if (!HasFlagBit(tilingProps, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
        {
            IF_LOG_ASSERTION("VulkanRHI2", !HasFlagBit(imageCI.usage, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT),
                "Image cannot be both color and depth attachment");
            imageCI.usage &= ~VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        }
        if (!HasFlagBit(tilingProps, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
        {
            IF_LOG_ASSERTION("VulkanRHI2", !HasFlagBit(imageCI.usage, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT),
                "Image cannot be both color and depth attachment");
            imageCI.usage &= ~VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        }
        if (!HasFlagBit(tilingProps, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
        {
            IF_LOG_ASSERTION("VulkanRHI2", !HasFlagBit(imageCI.usage, VK_IMAGE_USAGE_STORAGE_BIT),
                "Image cannot be both storage and depth/color attachment");
            imageCI.usage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
        }
        if (!HasFlagBit(tilingProps, VK_FORMAT_FEATURE_BLIT_SRC_BIT))
        {
            imageCI.usage &= ~VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        }
        if (!HasFlagBit(tilingProps, VK_FORMAT_FEATURE_BLIT_DST_BIT))
        {
            imageCI.usage &= ~VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        }

        // set initial layout request
        imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        // todo:

        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_AUTO;
        if (HasFlagBit(desc.mUsage, ERhiImageUsageFlag::CPUWritable))
        {
            allocCI.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }

        // create image
        VA_AssertResult(vmaCreateImage(castedCtx->GetAllocator()->mAllocator, &imageCI, &allocCI, &mData->mImage,
                            &mData->mAllocation, &mData->mAllocInfo),
            "Failed to create image");
        mData->mCreated = true;
    }

    VA_Texture::~VA_Texture()
    {
        if (mData->mCreated)
        {
            auto castedCtx = CheckedCast<VA_Device>(mContext);
            vmaDestroyImage(castedCtx->GetAllocator()->mAllocator, mData->mImage, mData->mAllocation);
        }
        delete mData;
        mData = nullptr;
    }

} // namespace Ifrit::RHI::VulkanRHI2