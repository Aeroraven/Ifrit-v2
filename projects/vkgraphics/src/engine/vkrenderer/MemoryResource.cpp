
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/vkgraphics/utility/Logger.h"

using namespace Ifrit;

namespace Ifrit::Graphics::VulkanGraphics
{
    IFRIT_APIDECL void SingleBuffer::Init()
    {
        VkBufferCreateInfo bufferCI{};
        bufferCI.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCI.size        = m_createInfo.size;
        bufferCI.usage       = m_createInfo.usage;
        bufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        bool                    hasDeviceSideAddr = m_createInfo.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_AUTO;
        if (m_createInfo.hostVisible)
        {
            allocCI.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }
        vkrVulkanAssert(
            vmaCreateBuffer(m_context->GetAllocator(), &bufferCI, &allocCI, &m_buffer, &m_allocation, &m_allocInfo),
            "Failed to create buffer");

        // Query device address
        VkBufferDeviceAddressInfo bufferDeviceAI{};
        bufferDeviceAI.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bufferDeviceAI.buffer = m_buffer;
        if (hasDeviceSideAddr)
            m_deviceAddress = vkGetBufferDeviceAddress(m_context->GetDevice(), &bufferDeviceAI);
        else
            m_deviceAddress = 0;
    }
    IFRIT_APIDECL SingleBuffer::~SingleBuffer() { vmaDestroyBuffer(m_context->GetAllocator(), m_buffer, m_allocation); }
    IFRIT_APIDECL void SingleBuffer::MapMemory()
    {
        vkrAssert(m_createInfo.hostVisible, "Buffer must be host visible to map");
        vmaMapMemory(m_context->GetAllocator(), m_allocation, (void**)&m_mappedMemory);
    }
    IFRIT_APIDECL void SingleBuffer::UnmapMemory()
    {
        vkrAssert(m_mappedMemory, "Buffer must be mapped to unmap");
        vmaUnmapMemory(m_context->GetAllocator(), m_allocation);
        m_mappedMemory = nullptr;
    }
    IFRIT_APIDECL void SingleBuffer::ReadBuffer(void* data, u32 size, u32 offset)
    {
        vkrAssert(m_mappedMemory, "Buffer must be mapped to copy data");
        memcpy(m_mappedMemory + offset, data, size);
    }
    IFRIT_APIDECL void SingleBuffer::WriteBuffer(const void* data, u32 size, u32 offset)
    {
        vkrAssert(m_mappedMemory, "Buffer must be mapped to copy data");
        memcpy((void*)(m_mappedMemory + offset), data, size);
    }
    IFRIT_APIDECL void SingleBuffer::FlushBuffer()
    {
        vmaFlushAllocation(m_context->GetAllocator(), m_allocation, 0, VK_WHOLE_SIZE);
    }

    IFRIT_APIDECL Rhi::RhiDeviceAddr SingleBuffer::GetDeviceAddress() const { return m_deviceAddress; }

    // Class: MultiBuffer
    IFRIT_APIDECL MultiBuffer::MultiBuffer(EngineContext* ctx, const BufferCreateInfo& ci, u32 numCopies)
        : Rhi::RhiMultiBuffer(ctx->GetDeleteQueue()), m_context(ctx), m_createInfo(ci)
    {
        for (u32 i = 0; i < numCopies; i++)
        {
            auto bufferPtr = new SingleBuffer(ctx, ci);
            auto bufferRef = MakeCountRef<Rhi::RhiBuffer>(bufferPtr);
            m_buffersOwning.push_back(bufferRef);
            m_buffers.push_back(bufferPtr);
        }
    }

    IFRIT_APIDECL SingleBuffer* MultiBuffer::GetBuffer(u32 index) { return m_buffers[index]; }

    // Class: Image
    IFRIT_APIDECL VkFormat      SingleDeviceImage::GetFormat() const { return m_format; }
    IFRIT_APIDECL VkImage       SingleDeviceImage::GetImage() const { return m_image; }
    IFRIT_APIDECL VkImageView   SingleDeviceImage::GetImageView() { return m_imageView; }

    IFRIT_APIDECL VkImageView   SingleDeviceImage::GetImageViewMipLayer(
        u32 mipLevel, u32 layer, u32 mipRange, u32 layerRange)
    {
        auto key = (static_cast<uint64_t>(mipLevel) << 16) | (static_cast<uint64_t>(layer) << 32)
            | (static_cast<uint64_t>(mipRange) << 48) | (static_cast<uint64_t>(layerRange) << 56);
        if (m_managedImageViews.count(key))
        {
            return m_managedImageViews[key];
        }

        // create image view
        auto&                 ci = m_createInfo;
        VkImageViewCreateInfo imageViewCI{};
        imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCI.image = m_image;
        if (ci.type == ImageType::ImageCube)
        {
            imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        }
        else if (ci.type == ImageType::Image2D)
        {
            imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        }
        else if (ci.type == ImageType::Image3D)
        {
            imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_3D;
        }
        imageViewCI.format = ci.format;
        if (ci.aspect == ImageAspect::Color)
        {
            imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        else if (ci.aspect == ImageAspect::Depth)
        {
            imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        }
        else if (ci.aspect == ImageAspect::Stencil)
        {
            imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        imageViewCI.subresourceRange.baseMipLevel   = mipLevel;
        imageViewCI.subresourceRange.levelCount     = mipRange;
        imageViewCI.subresourceRange.baseArrayLayer = layer;
        imageViewCI.subresourceRange.layerCount     = layerRange;

        VkImageView imageView;
        vkrVulkanAssert(vkCreateImageView(m_context->GetDevice(), &imageViewCI, nullptr, &imageView),
            "Failed to create image view");
        m_managedImageViews[key] = imageView;
        return imageView;
    }

    IFRIT_APIDECL SingleDeviceImage::~SingleDeviceImage()
    {
        if (m_created)
        {
            vmaDestroyImage(m_context->GetAllocator(), m_image, m_allocation);
            for (auto& [k, v] : m_managedImageViews)
            {
                vkDestroyImageView(m_context->GetDevice(), v, nullptr);
            }
        }
    }
    IFRIT_APIDECL SingleDeviceImage::SingleDeviceImage(EngineContext* ctx, const ImageCreateInfo& ci)
        : Rhi::RhiTexture(ctx->GetDeleteQueue())
    {
        m_context    = ctx;
        m_createInfo = ci;
        VkImageCreateInfo imageCI{};
        imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;

        if (ci.type == ImageType::Image2D)
        {
            imageCI.imageType = VK_IMAGE_TYPE_2D;
        }
        else if (ci.type == ImageType::Image3D)
        {
            imageCI.imageType = VK_IMAGE_TYPE_3D;
        }
        else if (ci.type == ImageType::ImageCube)
        {
            imageCI.imageType = VK_IMAGE_TYPE_2D;
            imageCI.flags     = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        }

        imageCI.format        = ci.format;
        imageCI.extent.width  = ci.width;
        imageCI.extent.height = ci.height;
        imageCI.extent.depth  = ci.depth;
        imageCI.mipLevels     = ci.mipLevels;
        imageCI.arrayLayers   = ci.arrayLayers;
        imageCI.samples       = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage         = ci.usage;
        imageCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageCI.flags         = 0;

        // Resolve samples
        switch (ci.samples)
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
                vkrError("Unsupported sample count");
        }

        m_format = ci.format;
        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_AUTO;
        if (ci.hostVisible)
        {
            allocCI.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }
        vkrVulkanAssert(vmaCreateImage(ctx->GetAllocator(), &imageCI, &allocCI, &m_image, &m_allocation, &m_allocInfo),
            "Failed to create image");
        m_imageView = this->GetImageViewMipLayer(0, 0, ci.mipLevels, ci.arrayLayers);
        m_created   = true;
    }

    IFRIT_APIDECL u32 SingleDeviceImage::GetSize()
    {
        auto pixelNums = m_createInfo.width * m_createInfo.height * m_createInfo.depth;
        auto pixelSize = 0;

#define FORMAT_MAP(format, pxSize) \
    case format:                   \
        pixelSize = pxSize;        \
        break;
        switch (m_createInfo.format)
        {
            FORMAT_MAP(VK_FORMAT_R8G8B8A8_UNORM, 4)
            FORMAT_MAP(VK_FORMAT_R8G8B8A8_SRGB, 4)
            FORMAT_MAP(VK_FORMAT_R32G32B32A32_SFLOAT, 16)
            FORMAT_MAP(VK_FORMAT_R32G32B32_SFLOAT, 12)
            FORMAT_MAP(VK_FORMAT_R32G32_SFLOAT, 8)
            FORMAT_MAP(VK_FORMAT_R32_SFLOAT, 4)
            FORMAT_MAP(VK_FORMAT_R32_UINT, 4)
            FORMAT_MAP(VK_FORMAT_R32G32B32A32_UINT, 16)
            FORMAT_MAP(VK_FORMAT_R32G32B32_UINT, 12)
            FORMAT_MAP(VK_FORMAT_R32G32_UINT, 8)
            FORMAT_MAP(VK_FORMAT_R32G32B32A32_SINT, 16)
            FORMAT_MAP(VK_FORMAT_R32G32B32_SINT, 12)
            FORMAT_MAP(VK_FORMAT_R32G32_SINT, 8)
            FORMAT_MAP(VK_FORMAT_R32_SINT, 4)

            default:
                vkrError("Unsupported format");
        }
#undef FORMAT_MAP
        return pixelNums * pixelSize;
    }

    // Class: Sampler
    IFRIT_APIDECL Sampler::Sampler(EngineContext* ctx, const SamplerCreateInfo& ci)
        : Rhi::RhiSampler(ctx->GetDeleteQueue())
    {
        m_context = ctx;
        VkSamplerCreateInfo samplerCI{};
        samplerCI.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCI.magFilter    = ci.magFilter;
        samplerCI.minFilter    = ci.minFilter;
        samplerCI.addressModeU = ci.addressModeU;
        samplerCI.addressModeV = ci.addressModeV;
        samplerCI.addressModeW = ci.addressModeW;
        if (ctx->GetPhysicalDeviceProperties().limits.maxSamplerAnisotropy < 1.0f)
        {
            samplerCI.anisotropyEnable = VK_FALSE;
            samplerCI.maxAnisotropy    = 1.0f;
            if (ci.anisotropyEnable)
            {
                vkrLog("Anisotropy not supported. Feature disabled");
            }
        }
        else
        {
            samplerCI.anisotropyEnable = VK_TRUE;
            samplerCI.maxAnisotropy    = ci.maxAnisotropy;
            if (samplerCI.maxAnisotropy < 0.0f)
            {
                samplerCI.maxAnisotropy = ctx->GetPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
            }
        }

        samplerCI.borderColor             = ci.borderColor;
        samplerCI.unnormalizedCoordinates = ci.unNormalizedCoordinates;
        samplerCI.compareEnable           = ci.compareEnable;
        samplerCI.compareOp               = ci.compareOp;
        samplerCI.mipmapMode              = ci.mipmapMode;
        samplerCI.mipLodBias              = ci.mipLodBias;
        samplerCI.minLod                  = ci.minLod;
        samplerCI.maxLod                  = ci.maxLod;

        vkrVulkanAssert(vkCreateSampler(ctx->GetDevice(), &samplerCI, nullptr, &m_sampler), "Failed to create sampler");
    }

    IFRIT_APIDECL Sampler::~Sampler() { vkDestroySampler(m_context->GetDevice(), m_sampler, nullptr); }

    // Class: ResourceManager
    IFRIT_APIDECL std::shared_ptr<MultiBuffer> ResourceManager::CreateMultipleBuffer(
        const BufferCreateInfo& ci, u32 numCopies)
    {
        if (numCopies == UINT32_MAX)
        {
            if (m_defaultCopies == -1)
            {
                vkrError("No default copies set for multiple buffer");
            }
            numCopies = m_defaultCopies;
        }
        auto buffer = std::make_shared<MultiBuffer>(m_context, ci, numCopies);
        m_multiBuffer.push_back(buffer);
        return buffer;
    }

    IFRIT_APIDECL Rhi::RhiTextureRef ResourceManager::CreateSimpleImageUnmanaged(const ImageCreateInfo& ci)
    {
        auto imagePtr = new SingleDeviceImage(m_context, ci);
        auto image    = MakeCountRef<Rhi::RhiTexture>(imagePtr);
        return image;
    }

    IFRIT_APIDECL Rhi::RhiBufferRef ResourceManager::CreateSimpleBufferUnmanaged(const BufferCreateInfo& ci)
    {
        auto bufferPtr = new SingleBuffer(m_context, ci);
        auto buffer    = MakeCountRef<Rhi::RhiBuffer>(bufferPtr);
        return buffer;
    }

    IFRIT_APIDECL std::shared_ptr<MultiBuffer> ResourceManager::CreateTracedMultipleBuffer(
        const BufferCreateInfo& ci, u32 numCopies)
    {
        if (numCopies == UINT32_MAX)
        {
            if (m_defaultCopies == -1)
            {
                vkrError("No default copies set for multiple buffer");
            }
            numCopies = m_defaultCopies;
        }
        auto buffer = std::make_shared<MultiBuffer>(m_context, ci, numCopies);
        auto ptr    = buffer.get();
        m_multiBuffer.push_back(buffer);
        m_multiBufferTraced.push_back(SizeCast<int>(m_multiBuffer.size()) - 1);
        return buffer;
    }

    IFRIT_APIDECL void ResourceManager::SetActiveFrame(u32 frame)
    {
        for (int i = 0; i < m_multiBufferTraced.size(); i++)
        {
            m_multiBuffer[m_multiBufferTraced[i]]->SetActiveFrame(frame);
        }
    }

    // Shortcut methods
    // IFRIT_APIDECL MultiBuffer *
    // ResourceManager::CreateStorageBufferShared(u32 size, bool hostVisible,
    //                                            VkFlags extraFlags) {
    //   BufferCreateInfo ci{};
    //   ci.size = size;
    //   ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extraFlags;
    //   ci.hostVisible = hostVisible;
    //   return CreateTracedMultipleBuffer(ci);
    // }

    // IFRIT_APIDECL MultiBuffer *
    // ResourceManager::createUniformBufferShared(u32 size, bool hostVisible,
    //                                            VkFlags extraFlags) {
    //   BufferCreateInfo ci{};
    //   ci.size = size;
    //   ci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | extraFlags;
    //   ci.hostVisible = hostVisible;
    //   return CreateTracedMultipleBuffer(ci);
    // }

    IFRIT_APIDECL Rhi::RhiTextureRef ResourceManager::CreateDepthAttachment(
        u32 width, u32 height, VkFormat format, VkImageUsageFlags extraUsage)
    {
        ImageCreateInfo ci{};
        ci.aspect      = ImageAspect::Depth;
        ci.format      = format;
        ci.width       = width;
        ci.height      = height;
        ci.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | extraUsage;
        ci.hostVisible = false;
        return CreateSimpleImageUnmanaged(ci);
    }

    IFRIT_APIDECL Rhi::RhiTextureRef ResourceManager::CreateTexture2DDeviceUnmanaged(
        u32 width, u32 height, VkFormat format, VkImageUsageFlags extraUsage, u32 samples)
    {
        ImageCreateInfo ci{};
        ci.aspect      = ImageAspect::Color;
        ci.format      = format;
        ci.width       = width;
        ci.height      = height;
        ci.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | extraUsage;
        ci.hostVisible = false;
        ci.samples     = samples;
        return CreateSimpleImageUnmanaged(ci);
    }

    IFRIT_APIDECL Rhi::RhiTextureRef ResourceManager::CreateRenderTargetTexture(
        u32 width, u32 height, VkFormat format, VkImageUsageFlags extraUsage)
    {
        ImageCreateInfo ci{};
        ci.aspect      = ImageAspect::Color;
        ci.format      = format;
        ci.width       = width;
        ci.height      = height;
        ci.usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | extraUsage;
        ci.hostVisible = false;
        return CreateSimpleImageUnmanaged(ci);
    }

    IFRIT_APIDECL Rhi::RhiTextureRef ResourceManager::CreateTexture3D(
        u32 width, u32 height, u32 depth, VkFormat format, VkImageUsageFlags extraUsage)
    {
        ImageCreateInfo ci{};
        ci.aspect = ImageAspect::Color;
        if (depth == 1)
            ci.type = ImageType::Image2D;
        else
            ci.type = ImageType::Image3D;
        ci.format      = format;
        ci.width       = width;
        ci.height      = height;
        ci.depth       = depth;
        ci.usage       = extraUsage;
        ci.hostVisible = false;
        return CreateSimpleImageUnmanaged(ci);
    }

    IFRIT_APIDECL Rhi::RhiTextureRef ResourceManager::createMipTexture(
        u32 width, u32 height, u32 mips, VkFormat format, VkImageUsageFlags extraUsage)
    {
        ImageCreateInfo ci{};
        ci.aspect      = ImageAspect::Color;
        ci.format      = format;
        ci.width       = width;
        ci.height      = height;
        ci.usage       = extraUsage;
        ci.hostVisible = false;
        ci.mipLevels   = mips;
        return CreateSimpleImageUnmanaged(ci);
    }

    IFRIT_APIDECL Rhi::RhiSamplerRef ResourceManager::CreateTrivialRenderTargetSampler()
    {
        SamplerCreateInfo ci{};
        ci.magFilter               = VK_FILTER_LINEAR;
        ci.minFilter               = VK_FILTER_LINEAR;
        ci.addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.anisotropyEnable        = false;
        ci.maxAnisotropy           = 1.0f;
        ci.borderColor             = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        ci.unNormalizedCoordinates = VK_FALSE;
        ci.compareEnable           = VK_FALSE;
        ci.compareOp               = VK_COMPARE_OP_ALWAYS;
        ci.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        ci.mipLodBias              = 0.0f;
        ci.minLod                  = 0.0f;
        ci.maxLod                  = 0.0f;
        auto samplerPtr            = new Sampler(m_context, ci);
        auto sampler               = MakeCountRef<Rhi::RhiSampler>(samplerPtr);
        return sampler;
    }

    IFRIT_APIDECL Rhi::RhiSamplerRef ResourceManager::CreateTrivialBilinearSampler(bool repeat)
    {
        SamplerCreateInfo ci{};
        ci.magFilter = VK_FILTER_LINEAR;
        ci.minFilter = VK_FILTER_LINEAR;
        if (repeat)
        {
            ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        }
        else
        {
            ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        }

        ci.anisotropyEnable        = false;
        ci.maxAnisotropy           = 1.0f;
        ci.borderColor             = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        ci.unNormalizedCoordinates = VK_FALSE;
        ci.compareEnable           = VK_FALSE;
        ci.compareOp               = VK_COMPARE_OP_ALWAYS;
        ci.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        ci.mipLodBias              = 0.0f;
        ci.minLod                  = 0.0f;
        ci.maxLod                  = 0.0f;
        auto samplerPtr            = new Sampler(m_context, ci);
        auto sampler               = MakeCountRef<Rhi::RhiSampler>(samplerPtr);
        return sampler;
    }

    IFRIT_APIDECL
    Rhi::RhiSamplerRef ResourceManager::CreateTrivialNearestSampler(bool repeat)
    {
        SamplerCreateInfo ci{};
        ci.magFilter = VK_FILTER_NEAREST;
        ci.minFilter = VK_FILTER_NEAREST;
        if (repeat)
        {
            ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        }
        else
        {
            ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        }

        ci.anisotropyEnable        = false;
        ci.maxAnisotropy           = 1.0f;
        ci.borderColor             = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        ci.unNormalizedCoordinates = VK_FALSE;
        ci.compareEnable           = VK_FALSE;
        ci.compareOp               = VK_COMPARE_OP_ALWAYS;
        ci.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        ci.mipLodBias              = 0.0f;
        ci.minLod                  = 0.0f;
        ci.maxLod                  = 0.0f;
        auto samplerPtr            = new Sampler(m_context, ci);
        auto sampler               = MakeCountRef<Rhi::RhiSampler>(samplerPtr);
        return sampler;
    }

} // namespace Ifrit::Graphics::VulkanGraphics