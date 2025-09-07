#include "ifrit/vkrhi2/adapter/MemoryResource.h"
#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/util/Helpers.h"
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit.internal/vkrhi2/adapter/AllocatorWrapper.h"
#include "ifrit/core/typing/EnumReflection.h"
#include "ifrit.internal/vkrhi2/adapter/TextureUtils.h"
#include "ifrit/vkrhi2/adapter/CommandContext.h"
#include "ifrit/vkrhi2/adapter/CommandBuffer.h"
#include "ifrit.internal/vkrhi2/adapter/CmdHelpersTexClear.h"
#include "ifrit.internal/vkrhi2/adapter/CmdHelpersBarrier.h"
#include "ifrit/rhi/common/RhiCommandList.h"
#include "ifrit/core/algo/BuddyAllocator.h"
#include "ifrit/core/console/ConsoleObject.h"

#include <vma/vk_mem_alloc.h>

namespace Ifrit::RHI::VulkanRHI2
{

    static TConsoleVariable<u64> cvVulkanStagingBufferBlockSize(
        "cv.VulkanRHI2.StagingBufferBlockSize", 1048576, "Vulkan Staging Buffer Block Size", CVF_ReadOnly);
    static TConsoleVariable<u32> cvVulkanStagingBufferGranularity(
        "cv.VulkanRHI2.StagingBufferGranularity", 32, "Vulkan Staging Buffer Granularity", CVF_ReadOnly);

    struct VA_BufferInternal
    {
        VkBuffer          mBuffer;
        VmaAllocation     mAllocation;
        VmaAllocationInfo mAllocInfo;
        u64               mDeviceAddress = 0;
        char*             mMappedMemory  = nullptr;
        bool              mCreated       = false;

        bool              mLockState = false;
        VA_StagingBuffer  mStagingBuffer;
    };

    struct VA_TextureInternal
    {
        VkImage           mImage;
        VmaAllocation     mAllocation;
        VmaAllocationInfo mAllocInfo;
        bool              mCreated = false;
    };

    // ===== VA Buffer =====
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

    void* VA_Buffer::MapMemory(RhiCommandListBase* cmdList, ERhiBufferMapType mapType)
    {
        IF_LOG_ASSERTION("VulkanRHI2", !mData->mLockState, "Buffer is already mapped");
        IF_LOG_ASSERTION("VulkanRHI2", mapType == ERhiBufferMapType::CPUWriteOnly,
            "Only CPUWriteOnly map type is supported currently");
        mData->mLockState = true;

        auto stagingBufferManager = CheckedCast<VA_Device>(mContext)->GetStagingBufferManager();
        mData->mStagingBuffer     = stagingBufferManager->AllocateStagingBuffer(mDesc.mSize);
        return mData->mStagingBuffer.mMappedPtr;
    }
    void VA_Buffer::UnmapMemory(RhiCommandListBase* cmdList)
    {
        IF_LOG_ASSERTION("VulkanRHI2", mData->mLockState, "Buffer is not mapped");
        mData->mLockState = false;

        cmdList->EnqueueLambda([buffer = this, stagingBuffer = mData->mStagingBuffer](RhiCommandListBase* cmdList) {
            auto         cmdCtx    = CheckedCast<VA_CommandListContext>(cmdList->GetActiveContext());
            auto         cmdBuf    = cmdCtx->GetCommandBuffer();
            auto         cmdNative = cmdBuf->GetCmd();
            auto         device    = CheckedCast<VA_Device>(buffer->mContext);

            // Copy from staging buffer to the actual buffer
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = stagingBuffer.mInBlockOffset;
            copyRegion.dstOffset = 0;
            copyRegion.size      = buffer->mDesc.mSize;
            vkCmdCopyBuffer(cmdNative, stagingBuffer.mHandle, buffer->mData->mBuffer, 1, &copyRegion);

            CmdLegacyGlobalPipelineBarrier(cmdNative, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);

            cmdCtx->AddCompletionCallback([stagingBuffer, device]() {
                auto stagingBufferManager = device->GetStagingBufferManager();
                stagingBufferManager->FreeStagingBuffer(stagingBuffer);
            });
        });
    }

    RhiDeviceAddr VA_Buffer::GetDeviceAddress() const { return mData->mDeviceAddress; }
    RhiRawHandle  VA_Buffer::GetRawHandle() const { return reinterpret_cast<RhiRawHandle>(mData->mBuffer); }

    // ===== VA Texture =====
    VA_Texture::VA_Texture(RhiDevice* device, const RhiTextureDesc& desc) : RhiTexture(desc)
    {
        mContext                    = device;
        mData                       = new VA_TextureInternal();
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
        auto desiredUsage = TranslateCreateFlags(castedCtx, desc.mUsage);
        imageCI.usage     = desiredUsage;

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
            imageCI.usage &= ~VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        }
        if (!HasFlagBit(tilingProps, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
        {
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
        switch (desc.mInitialState)
        {
            case ERhiResourceState::Undefined:
                imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                break;
            case ERhiResourceState::Common:
                imageCI.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
                break;
            case ERhiResourceState::ColorRT:
                imageCI.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                break;
            case ERhiResourceState::DepthStencilRT:
                imageCI.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                break;
            case ERhiResourceState::ShaderRead:
                imageCI.initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                break;
            case ERhiResourceState::UnorderedAccess:
                imageCI.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
                break;
            case ERhiResourceState::CopySrc:
                imageCI.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                break;
            case ERhiResourceState::CopyDst:
                imageCI.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                break;
        }
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

        // IF_LOG_INFO("VulkanRHI2", "Texture handle: {}", (void*)mData->mImage);

        auto cmdList        = RHI::GetCommandListExecutor()->GetImmediateCmdList();
        bool isRenderTarget = HasFlagBit(desc.mUsage, ERhiImageUsageFlag::RenderTarget)
            || HasFlagBit(desc.mUsage, ERhiImageUsageFlag::Depth);
        SetInitialState(cmdList, imageCI.initialLayout, isRenderTarget, desc.mClearValue);

        mData->mCreated = true;
    }

    VkImageSubresourceRange VA_Texture::GetAllSubresourceRange()
    {
        VkImageSubresourceRange range{};
        range.aspectMask =
            HasFlagBit(mDesc.mUsage, ERhiImageUsageFlag::Depth) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        range.baseMipLevel   = 0;
        range.levelCount     = mDesc.mMips;
        range.baseArrayLayer = 0;
        range.layerCount     = mDesc.mArraySize;
        return range;
    }

    void VA_Texture::SetInitialState(
        RhiCommandListBase* cmdList, VkImageLayout desiredInitLayout, bool doInitClear, RhiClearValue2 clearValue)
    {
        cmdList->EnqueueLambda([this, desiredInitLayout, doInitClear, clearValue](RhiCommandListBase* cmdList) {
            auto uploadCtx   = cmdList->GetUploadContext();
            auto uploadVkCtx = CheckedCast<VA_CommandListContext>(uploadCtx);
            auto nativeCmd   = uploadVkCtx->GetCommandBuffer()->GetCmd();

            auto currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            if (doInitClear)
            {
                bool isRenderTarget = HasFlagBit(mDesc.mUsage, ERhiImageUsageFlag::RenderTarget);
                bool isDepth        = HasFlagBit(mDesc.mUsage, ERhiImageUsageFlag::Depth);
                auto subresRange    = GetAllSubresourceRange();
                bool requireClear   = (isRenderTarget || isDepth);

                if (requireClear)
                {
                    VA_PipelineBarriers barriers;
                    barriers.AddImageTransition(
                        mData->mImage, currentLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, GetAllSubresourceRange());
                    barriers.ExecuteNative(nativeCmd);
                    currentLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                }
                if (isRenderTarget)
                {
                    IF_LOG_ASSERTION("VulkanRHI2", clearValue.m_Type == ERhiClearValueType::Color,
                        "Initial clear value type must be color for render target");
                    CmdClearColorTexture(nativeCmd, mData->mImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        clearValue.m_Color, subresRange);
                }
                else if (isDepth)
                {
                    IF_LOG_ASSERTION("VulkanRHI2", clearValue.m_Type == ERhiClearValueType::DepthStencil,
                        "Initial clear value type must be depth-stencil for depth texture");
                    CmdClearDepthStencilTexture(nativeCmd, mData->mImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        clearValue.m_DepthStencil.m_Depth, clearValue.m_DepthStencil.m_Stencil, subresRange);
                }
            }
            if (currentLayout != desiredInitLayout && desiredInitLayout != VK_IMAGE_LAYOUT_UNDEFINED)
            {
                VA_PipelineBarriers barriers;
                barriers.AddImageTransition(mData->mImage, currentLayout, desiredInitLayout, GetAllSubresourceRange());
                barriers.ExecuteNative(nativeCmd);
                IF_LOG_INFO("VulkanRHI2", "Texture initial layout transition executed");
            }
        });

        // IF_LOG_INFO("VulkanRHI2", "Texture created with initial state");
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

    RhiRawHandle VA_Texture::GetRawHandle() const { return reinterpret_cast<RhiRawHandle>(mData->mImage); }

    VkFormat     VA_Texture::GetVkFormat() const
    {
        auto fnFmtTranslator = [](ERhiImageFormat fmt) -> VkFormat {
            IF_LOG_ASSERTION("VulkanRHI2", fmt != ERhiImageFormat::Undefined, "Invalid image format");
            return static_cast<VkFormat>(fmt);
        };
        return static_cast<VkFormat>(mDesc.mFormat);
    }

    // ===== Staging Buffer Manager =====

    struct VA_StagingAllocResult
    {
        bool mSuccess = false;
        u64  mOffset  = ~0ull;
    };

    struct VA_StagingBufferBlocks : public NonCopyable
    {
        bool                  mSingleAllocation = false;
        bool                  mSingleAvailable  = true;
        BuddyAddressAllocator mAllocator;
        u64                   mSize;
        VkBuffer              mBuffer;
        VmaAllocation         mAllocation;
        Mutex                 mMutex;
        VA_Device*            mDevice;

        void*                 mMappedAddress;

        VA_StagingBufferBlocks(u64 size, bool singleAlloc, VA_Device* device)
            : mSingleAllocation(singleAlloc)
            , mAllocator(cvVulkanStagingBufferGranularity.GetValue(), size)
            , mSize(size)
            , mBuffer(VK_NULL_HANDLE)
            , mAllocation(VK_NULL_HANDLE)
            , mDevice(device)
        {
            VkBufferCreateInfo bufferCI{};
            bufferCI.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferCI.size        = size;
            bufferCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocCI{};
            allocCI.usage = VMA_MEMORY_USAGE_AUTO;
            allocCI.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VA_AssertResult(vmaCreateBuffer(mDevice->GetAllocator()->mAllocator, &bufferCI, &allocCI, &mBuffer,
                                &mAllocation, nullptr),
                "Failed to create staging buffer");

            IF_LOG_DEBUG("VulkanRHI2", "Created staging buffer block of size {}", size);

            // get mapped address
            vmaMapMemory(mDevice->GetAllocator()->mAllocator, mAllocation, &mMappedAddress);
        }

        ~VA_StagingBufferBlocks()
        {
            if (mBuffer != VK_NULL_HANDLE)
            {
                vmaUnmapMemory(mDevice->GetAllocator()->mAllocator, mAllocation);
                vmaDestroyBuffer(mDevice->GetAllocator()->mAllocator, mBuffer, mAllocation);
                mBuffer     = VK_NULL_HANDLE;
                mAllocation = VK_NULL_HANDLE;
            }
        }

        void* GetOffsetedPtr(u64 offset)
        {
            return reinterpret_cast<void*>(reinterpret_cast<u64>(mMappedAddress) + offset);
        }

        VA_StagingAllocResult Allocate(u64 size)
        {
            ScopedLock            lock(mMutex);
            VA_StagingAllocResult ret;
            auto                  alloc = mAllocator.Allocate(size);
            ret.mSuccess                = alloc.mSuccess;
            ret.mOffset                 = alloc.mOffset;
            return ret;
        }

        void Free(u64 offset)
        {
            ScopedLock lock(mMutex);
            mAllocator.Free(offset);
        }
    };

    struct VA_StagingBufferManagerInternal : public NonCopyable
    {
        VA_Device*                         mDevice;
        Vec<Owner<VA_StagingBufferBlocks>> mSmallBlocks;
        Vec<Owner<VA_StagingBufferBlocks>> mSingleBlocks;
        Mutex                              mMutex;
    };

    VA_StagingBufferManager::VA_StagingBufferManager(VA_Device* device)
    {
        mData          = new VA_StagingBufferManagerInternal();
        mData->mDevice = device;
    }

    VA_StagingBufferManager::~VA_StagingBufferManager()
    {
        delete mData;
        mData = nullptr;
    }

    VA_StagingBuffer VA_StagingBufferManager::AllocateStagingBuffer(u64 size)
    {
        ScopedLock lock(mData->mMutex);
        if (size <= cvVulkanStagingBufferBlockSize.GetValue())
        {
            for (int i = 0; i < mData->mSmallBlocks.size(); ++i)
            {
                auto& block = mData->mSmallBlocks[i];
                auto  res   = block->Allocate(size);
                if (res.mSuccess)
                {
                    auto stagingBuffer           = VA_StagingBuffer();
                    stagingBuffer.mBlockIdx      = i;
                    stagingBuffer.mIsSingle      = 0;
                    stagingBuffer.mInBlockOffset = res.mOffset;
                    stagingBuffer.mSize          = size;
                    stagingBuffer.mMappedPtr     = block->GetOffsetedPtr(res.mOffset);
                    stagingBuffer.mHandle        = block->mBuffer;
                    return stagingBuffer;
                }
            }
            AddSmallBlock();
            auto& block = mData->mSmallBlocks.back();
            auto  res   = block->Allocate(size);
            IF_LOG_ASSERTION("VulkanRHI2", res.mSuccess, "Failed to allocate staging buffer from new small block");
            auto stagingBuffer           = VA_StagingBuffer();
            stagingBuffer.mBlockIdx      = static_cast<u32>(mData->mSmallBlocks.size() - 1);
            stagingBuffer.mIsSingle      = 0;
            stagingBuffer.mInBlockOffset = res.mOffset;
            stagingBuffer.mSize          = size;
            stagingBuffer.mMappedPtr     = block->GetOffsetedPtr(res.mOffset);
            stagingBuffer.mHandle        = block->mBuffer;
            return stagingBuffer;
        }
        else
        {
            for (int i = 0; i < mData->mSingleBlocks.size(); ++i)
            {
                auto& block = mData->mSingleBlocks[i];
                if (block->mSize >= size && block->mSingleAvailable)
                {
                    block->mSingleAvailable      = false;
                    auto stagingBuffer           = VA_StagingBuffer();
                    stagingBuffer.mBlockIdx      = i;
                    stagingBuffer.mIsSingle      = 1;
                    stagingBuffer.mInBlockOffset = 0;
                    stagingBuffer.mSize          = size;
                    stagingBuffer.mMappedPtr     = block->GetOffsetedPtr(0);
                    stagingBuffer.mHandle        = block->mBuffer;
                    return stagingBuffer;
                }
            }
            AddSingleBlock(size);
            auto& block                  = mData->mSingleBlocks.back();
            block->mSingleAvailable      = false;
            auto stagingBuffer           = VA_StagingBuffer();
            stagingBuffer.mBlockIdx      = static_cast<u32>(mData->mSingleBlocks.size() - 1);
            stagingBuffer.mIsSingle      = 1;
            stagingBuffer.mInBlockOffset = 0;
            stagingBuffer.mSize          = size;
            stagingBuffer.mMappedPtr     = block->GetOffsetedPtr(0);
            stagingBuffer.mHandle        = block->mBuffer;
            return stagingBuffer;
        }
    }

    void VA_StagingBufferManager::FreeStagingBuffer(const VA_StagingBuffer& buffer)
    {
        ScopedLock lock(mData->mMutex);
        if (buffer.mIsSingle)
        {
            IF_LOG_ASSERTION("VulkanRHI2", buffer.mBlockIdx < mData->mSingleBlocks.size(), "Invalid staging buffer");
            auto& block             = mData->mSingleBlocks[buffer.mBlockIdx];
            block->mSingleAvailable = true;
        }
        else
        {
            IF_LOG_ASSERTION("VulkanRHI2", buffer.mBlockIdx < mData->mSmallBlocks.size(), "Invalid staging buffer");
            auto& block = mData->mSmallBlocks[buffer.mBlockIdx];
            block->Free(buffer.mInBlockOffset);
        }
    }

    void VA_StagingBufferManager::AddSingleBlock(u64 size)
    {
        auto newBlock = MakeOwner<VA_StagingBufferBlocks>(size, true, mData->mDevice);
        mData->mSingleBlocks.push_back(std::move(newBlock));
    }

    void VA_StagingBufferManager::AddSmallBlock()
    {
        auto newBlock =
            MakeOwner<VA_StagingBufferBlocks>(cvVulkanStagingBufferBlockSize.GetValue(), false, mData->mDevice);
        mData->mSmallBlocks.push_back(std::move(newBlock));
    }

    // ===== Sampler State =====
    struct VA_SamplerInternal
    {
        VkSampler  mSampler = VK_NULL_HANDLE;
        VA_Device* mDevice  = nullptr;
    };

    VA_Sampler::VA_Sampler(VA_Device* device, const RhiSamplerDesc& state) : RhiSampler(state)
    {
        mContext        = device;
        mData           = new VA_SamplerInternal();
        mData->mDevice  = device;
        mData->mSampler = VK_NULL_HANDLE;

        VkSamplerCreateInfo samplerCI{};
        samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

        // filter
        switch (state.mFilterMode)
        {
            case ERhiSamplerFilter::Nearest:
                samplerCI.minFilter  = VK_FILTER_NEAREST;
                samplerCI.magFilter  = VK_FILTER_NEAREST;
                samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                break;
            case ERhiSamplerFilter::Linear:
                samplerCI.minFilter  = VK_FILTER_LINEAR;
                samplerCI.magFilter  = VK_FILTER_LINEAR;
                samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                break;
            default:
                IF_LOG_ASSERTION("VulkanRHI2", false, "Unsupported min filter");
        }

        // address mode
        auto translateAddrMode = [](ERhiSamplerWrapMode mode) -> VkSamplerAddressMode {
            switch (mode)
            {
                case ERhiSamplerWrapMode::Repeat:
                    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
                case ERhiSamplerWrapMode::Clamp:
                    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                default:
                    IF_LOG_ASSERTION("VulkanRHI2", false, "Unsupported address mode");
                    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            }
        };
        samplerCI.addressModeU = translateAddrMode(state.mWrapModeU);
        samplerCI.addressModeV = translateAddrMode(state.mWrapModeV);
        samplerCI.addressModeW = translateAddrMode(state.mWrapModeW);

        samplerCI.mipLodBias              = 0.0f;
        samplerCI.anisotropyEnable        = VK_TRUE;
        samplerCI.maxAnisotropy           = static_cast<f32>(state.mMaxAnisotropy);
        samplerCI.compareEnable           = VK_FALSE;
        samplerCI.compareOp               = VK_COMPARE_OP_ALWAYS;
        samplerCI.minLod                  = state.mMinLod;
        samplerCI.maxLod                  = static_cast<f32>(state.mMaxLod);
        samplerCI.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerCI.unnormalizedCoordinates = VK_FALSE;
        VA_AssertResult(vkCreateSampler(device->GetVulkanDevice(), &samplerCI, nullptr, &mData->mSampler),
            "Failed to create sampler");

        RegisterSamplerDescriptor();
    }

    VA_Sampler::~VA_Sampler()
    {
        UnregisterSamplerDescriptor();
        if (mData->mSampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(mData->mDevice->GetVulkanDevice(), mData->mSampler, nullptr);
            mData->mSampler = VK_NULL_HANDLE;
        }
        delete mData;
        mData = nullptr;
    }

    RhiRawHandle VA_Sampler::GetRawHandle() const { return reinterpret_cast<RhiRawHandle>(mData->mSampler); }

    // ===== Sampler Registry =====
    struct VA_SamplerRegistryInternal : public NonCopyable
    {
        VA_Device*                                  mDevice;
        HashMap<ERhiGlobalSamplerType, VA_Sampler*> mSamplerMap;
        Mutex                                       mMutex;
    };
    VA_SamplerRegistry::VA_SamplerRegistry(VA_Device* device)
    {
        mData          = new VA_SamplerRegistryInternal();
        mData->mDevice = device;

        CreateGlobalSampler(ERhiGlobalSamplerType::PointClamp);
        CreateGlobalSampler(ERhiGlobalSamplerType::PointWrap);
        CreateGlobalSampler(ERhiGlobalSamplerType::BilinearClamp);
        CreateGlobalSampler(ERhiGlobalSamplerType::BilinearWrap);
        CreateGlobalSampler(ERhiGlobalSamplerType::TrilinearClamp);
        CreateGlobalSampler(ERhiGlobalSamplerType::TrilinearWrap);
    }
    VA_SamplerRegistry::~VA_SamplerRegistry()
    {
        ScopedLock lock(mData->mMutex);
        for (auto& [desc, sampler] : mData->mSamplerMap)
        {
            delete sampler;
        }
        mData->mSamplerMap.clear();
        delete mData;
        mData = nullptr;
    }
    VA_Sampler* VA_SamplerRegistry::GetGlobalSampler(ERhiGlobalSamplerType type)
    {
        ScopedLock lock(mData->mMutex);
        auto       it = mData->mSamplerMap.find(type);
        if (it != mData->mSamplerMap.end())
        {
            return it->second;
        }
        return CreateGlobalSampler(type);
    }
    VA_Sampler* VA_SamplerRegistry::CreateGlobalSampler(ERhiGlobalSamplerType type)
    {
        RhiSamplerDesc desc;
        switch (type)
        {
            case ERhiGlobalSamplerType::PointClamp:
                desc.mFilterMode = ERhiSamplerFilter::Nearest;
                desc.mWrapModeU  = ERhiSamplerWrapMode::Clamp;
                desc.mWrapModeV  = ERhiSamplerWrapMode::Clamp;
                desc.mWrapModeW  = ERhiSamplerWrapMode::Clamp;
                break;
            case ERhiGlobalSamplerType::PointWrap:
                desc.mFilterMode = ERhiSamplerFilter::Nearest;
                desc.mWrapModeU  = ERhiSamplerWrapMode::Repeat;
                desc.mWrapModeV  = ERhiSamplerWrapMode::Repeat;
                desc.mWrapModeW  = ERhiSamplerWrapMode::Repeat;
                break;
            case ERhiGlobalSamplerType::BilinearClamp:
                desc.mFilterMode = ERhiSamplerFilter::Linear;
                desc.mWrapModeU  = ERhiSamplerWrapMode::Clamp;
                desc.mWrapModeV  = ERhiSamplerWrapMode::Clamp;
                desc.mWrapModeW  = ERhiSamplerWrapMode::Clamp;
                break;
            case ERhiGlobalSamplerType::BilinearWrap:
                desc.mFilterMode = ERhiSamplerFilter::Linear;
                desc.mWrapModeU  = ERhiSamplerWrapMode::Repeat;
                desc.mWrapModeV  = ERhiSamplerWrapMode::Repeat;
                desc.mWrapModeW  = ERhiSamplerWrapMode::Repeat;
                break;
            case ERhiGlobalSamplerType::TrilinearClamp:
                desc.mFilterMode    = ERhiSamplerFilter::Linear;
                desc.mWrapModeU     = ERhiSamplerWrapMode::Clamp;
                desc.mWrapModeV     = ERhiSamplerWrapMode::Clamp;
                desc.mWrapModeW     = ERhiSamplerWrapMode::Clamp;
                desc.mMinLod        = 0.0f;
                desc.mMaxLod        = FLT_MAX;
                desc.mMaxAnisotropy = 16;
                break;
            case ERhiGlobalSamplerType::TrilinearWrap:
                desc.mFilterMode    = ERhiSamplerFilter::Linear;
                desc.mWrapModeU     = ERhiSamplerWrapMode::Repeat;
                desc.mWrapModeV     = ERhiSamplerWrapMode::Repeat;
                desc.mWrapModeW     = ERhiSamplerWrapMode::Repeat;
                desc.mMinLod        = 0.0f;
                desc.mMaxLod        = FLT_MAX;
                desc.mMaxAnisotropy = 16;
                break;
            default:
                IF_LOG_ASSERTION("VulkanRHI2", false, "Unsupported global sampler type");
                return nullptr;
        }
        auto sampler             = new VA_Sampler(mData->mDevice, desc);
        mData->mSamplerMap[type] = sampler;
        return sampler;
    }

} // namespace Ifrit::RHI::VulkanRHI2