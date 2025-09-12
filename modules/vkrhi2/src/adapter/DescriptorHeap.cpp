#include "ifrit/vkrhi2/adapter/DescriptorHeap.h"
#include "ifrit/core/console/ConsoleObject.h"
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/vkrhi2/adapter/MemoryResource.h"

namespace Ifrit::RHI::VulkanRHI2
{

    static TConsoleVariable<u32> cvVulkanBindlessDescriptorMaxSize(
        "cv.VulkanRHI2.BindlessDescriptorMaxSize", 65536, "Vulkan Max Descriptor Bindings in Heap", CVF_ReadOnly);

    // ===== Resource View =====
    struct VA_ResourceViewInternal
    {
        VA_Device*          mDevice    = nullptr;
        VkImageView         mImageView = VK_NULL_HANDLE;
        EVA_ViewType        mType;
        VA_Texture*         mTexture = nullptr;
        VA_Buffer*          mBuffer  = nullptr;
        RhiResourceViewDesc mDesc;

        u32                 mInHeapElemId = ~0u;
    };
    IFRIT_VKRHI2_API VA_ResourceView::VA_ResourceView(
        const RhiResourceViewDesc& desc, VA_Texture* texture, VA_Device* device, EVA_ViewType type)

    {
        mInternal           = new VA_ResourceViewInternal();
        mInternal->mDevice  = device;
        mInternal->mType    = type;
        mInternal->mTexture = texture;
        mInternal->mDesc    = desc;

        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType                 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image                 = static_cast<VkImage>(texture->GetRawHandle());
        switch (texture->GetDimension())
        {
            case ERhiImageDimension::Texture2D:
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                break;
            case ERhiImageDimension::Texture2DArray:
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                break;
            case ERhiImageDimension::Texture3D:
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
                break;
            case ERhiImageDimension::TextureCube:
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
                break;
            case ERhiImageDimension::TextureCubeArray:
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
                break;
            default:
                IF_LOG_ASSERTION("VulkanRHI2", false, "Unsupported texture dimension");
                break;
        }
        viewInfo.format                          = texture->GetVkFormat();
        viewInfo.subresourceRange.baseArrayLayer = desc.mTextureView.mSubResource.arrayLayer;
        viewInfo.subresourceRange.layerCount     = desc.mTextureView.mSubResource.layerCount;
        viewInfo.subresourceRange.baseMipLevel   = desc.mTextureView.mSubResource.mipLevel;
        viewInfo.subresourceRange.levelCount     = desc.mTextureView.mSubResource.mipCount;
        if (texture->IsDepthTexture())
        {
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        }
        else
        {
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        auto result = vkCreateImageView(device->GetVulkanDevice(), &viewInfo, nullptr, &mInternal->mImageView);
    }

    IFRIT_VKRHI2_API VA_ResourceView::VA_ResourceView(
        const RhiResourceViewDesc& desc, VA_Buffer* buffer, VA_Device* device, EVA_ViewType type)
    {
        mInternal          = new VA_ResourceViewInternal();
        mInternal->mDevice = device;
        mInternal->mType   = type;
        mInternal->mBuffer = buffer;
        mInternal->mDesc   = desc;
    }

    IFRIT_VKRHI2_API VA_ResourceView::~VA_ResourceView()
    {
        if (mInternal->mImageView != VK_NULL_HANDLE)
        {
            vkDestroyImageView(mInternal->mDevice->GetVulkanDevice(), mInternal->mImageView, nullptr);
            mInternal->mImageView = VK_NULL_HANDLE;
        }
        delete mInternal;
        mInternal = nullptr;
    }

    void VA_ResourceView::AcquireHandleInternal(void* ptr)
    {
        auto device = mInternal->mDevice;
        auto heap   = device->GetBindlessDescriptorHeap();
        if (mInternal->mInHeapElemId != ~0u)
        {
            return;
        }
        if (mInternal->mDesc.mType == ERhiResourceViewedType::Texture)
        {
            if (mInternal->mType == EVA_ViewType::SRV)
            {
                auto srvIdx              = heap->RegisterSRV(*static_cast<VA_ResourceViewSRV*>(ptr));
                mInternal->mInHeapElemId = srvIdx;
            }
            else if (mInternal->mType == EVA_ViewType::UAV)
            {
                auto uavIdx              = heap->RegisterUAV(*static_cast<VA_ResourceViewUAV*>(ptr));
                mInternal->mInHeapElemId = uavIdx;
            }
        }
        else if (mInternal->mDesc.mType == ERhiResourceViewedType::Buffer)
        {
            if (mInternal->mType == EVA_ViewType::SRV)
            {
                auto srvIdx              = heap->RegisterSRV(*static_cast<VA_ResourceViewSRV*>(ptr));
                mInternal->mInHeapElemId = srvIdx;
            }
            else if (mInternal->mType == EVA_ViewType::UAV)
            {
                auto uavIdx              = heap->RegisterUAV(*static_cast<VA_ResourceViewUAV*>(ptr));
                mInternal->mInHeapElemId = uavIdx;
            }
        }
    }
    void VA_ResourceView::ReleaseHandleInternal(void* ptr)
    {
        auto device = mInternal->mDevice;
        auto heap   = device->GetBindlessDescriptorHeap();
        if (mInternal->mInHeapElemId != ~0u)
        {
            if (mInternal->mDesc.mType == ERhiResourceViewedType::Texture)
            {
                if (mInternal->mType == EVA_ViewType::SRV)
                {
                    heap->FreeSRVImage(mInternal->mInHeapElemId);
                }
                else if (mInternal->mType == EVA_ViewType::UAV)
                {
                    heap->FreeUAVImage(mInternal->mInHeapElemId);
                }
            }
            else if (mInternal->mDesc.mType == ERhiResourceViewedType::Buffer)
            {
                if (mInternal->mType == EVA_ViewType::SRV)
                {
                    heap->FreeSRVBuffer(mInternal->mInHeapElemId);
                }
                else if (mInternal->mType == EVA_ViewType::UAV)
                {
                    heap->FreeUAVBuffer(mInternal->mInHeapElemId);
                }
            }
            mInternal->mInHeapElemId = ~0u;
        }
    }
    RhiRawHandle     VA_ResourceView::GetRawHandleInternal() const { return mInternal->mImageView; }

    // ===== Specific View Types =====
    IFRIT_VKRHI2_API VA_ResourceViewSRV::VA_ResourceViewSRV(
        const RhiResourceViewDesc& desc, VA_Texture* texture, VA_Device* device)
        : VA_ResourceView(desc, texture, device, EVA_ViewType::SRV), RhiShaderReadView(texture, desc)
    {
        AcquireHandle();
        mHandle.mType  = ERhiDescriptorHeapType::SampledImage;
        mHandle.mIndex = mInternal->mInHeapElemId;
        mContext       = device;
    }

    IFRIT_VKRHI2_API VA_ResourceViewSRV::VA_ResourceViewSRV(
        const RhiResourceViewDesc& desc, VA_Buffer* buffer, VA_Device* device)
        : VA_ResourceView(desc, buffer, device, EVA_ViewType::SRV), RhiShaderReadView(buffer, desc)
    {
        AcquireHandle();
        mHandle.mType  = ERhiDescriptorHeapType::ReadOnlyStorageBuffer;
        mHandle.mIndex = mInternal->mInHeapElemId;
        mContext       = device;
    }

    IFRIT_VKRHI2_API VA_ResourceViewUAV::VA_ResourceViewUAV(
        const RhiResourceViewDesc& desc, VA_Texture* texture, VA_Device* device)
        : VA_ResourceView(desc, texture, device, EVA_ViewType::UAV), RhiUnorderedAccessView(texture, desc)
    {
        AcquireHandle();
        mHandle.mType  = ERhiDescriptorHeapType::StorageImage;
        mHandle.mIndex = mInternal->mInHeapElemId;
        mContext       = device;
    }

    IFRIT_VKRHI2_API VA_ResourceViewUAV::VA_ResourceViewUAV(
        const RhiResourceViewDesc& desc, VA_Buffer* buffer, VA_Device* device)
        : VA_ResourceView(desc, buffer, device, EVA_ViewType::UAV), RhiUnorderedAccessView(buffer, desc)
    {
        AcquireHandle();
        mHandle.mType  = ERhiDescriptorHeapType::StorageBuffer;
        mHandle.mIndex = mInternal->mInHeapElemId;
        mContext       = device;
    }

    IFRIT_VKRHI2_API           VA_ResourceViewSRV::~VA_ResourceViewSRV() { ReleaseHandle(); }
    IFRIT_VKRHI2_API           VA_ResourceViewUAV::~VA_ResourceViewUAV() { ReleaseHandle(); }

    // ===== Descriptor Heap =====

    constexpr VkDescriptorType kDescriptorTypes[] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER };

    constexpr u32              kMaxDescriptorType = sizeof(kDescriptorTypes) / sizeof(kDescriptorTypes[0]);

    struct VA_DescHeapSpec
    {
        u32      mHeapId;
        u32      mDescriptorTypeCount;
        Vec<u32> mFreeSlots;
        u32      mAllocatedSlots = 0;
        Mutex    mMutex;

        u32      Allocate()
        {
            ScopedLock lock(mMutex);
            if (!mFreeSlots.empty())
            {
                u32 slot = mFreeSlots.back();
                mFreeSlots.pop_back();
                return slot;
            }
            else
            {
                IF_LOG_ASSERTION("VulkanRHI2", mAllocatedSlots < mDescriptorTypeCount, "No more free descriptor slots");
                return mAllocatedSlots++;
            }
        }
        void Free(u32 slot)
        {
            ScopedLock lock(mMutex);
            IF_LOG_ASSERTION("VulkanRHI2", slot < mAllocatedSlots, "Invalid descriptor slot to free");
            mFreeSlots.push_back(slot);
        }
    };

    struct VA_BindlessDescriptorHeapInternal
    {
        VA_Device*                                              mDevice = nullptr;

        VA_DescHeapSpec                                         mSampledImageHeap{ 6 };
        VA_DescHeapSpec                                         mStorageImageHeap{ 4 };
        VA_DescHeapSpec                                         mSamplerHeap{ 3 };
        VA_DescHeapSpec                                         mRWStorageBufferHeap{ 2 };
        VA_DescHeapSpec                                         mReadOnlyStorageBufferHeap{ 1 };

        // Just Placeholder for future use
        VA_DescHeapSpec                                         mUniformBufferHeap{ 0 };
        VA_DescHeapSpec                                         mCombinedImageSamplerHeap{ 5 };

        Array<VkDescriptorSetLayoutBinding, kMaxDescriptorType> mLayoutBindings;
        Array<VkDescriptorBindingFlagsEXT, kMaxDescriptorType>  mBindingFlags;

        VkDescriptorSetLayout                                   mDescriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool                                        mDescriptorPool      = VK_NULL_HANDLE;
        VkDescriptorSet                                         mDescriptorSet       = VK_NULL_HANDLE;
    };

    IFRIT_VKRHI2_API VA_BindlessDescriptorHeap::VA_BindlessDescriptorHeap(VA_Device* device)
    {
        mInternal          = new VA_BindlessDescriptorHeapInternal();
        mInternal->mDevice = device;

        auto deviceProperty = device->GetProperties();

        mInternal->mSampledImageHeap.mDescriptorTypeCount =
            std::min(cvVulkanBindlessDescriptorMaxSize.GetValue(), deviceProperty.mMaxDescriptorsSetSRVImage);
        mInternal->mStorageImageHeap.mDescriptorTypeCount =
            std::min(cvVulkanBindlessDescriptorMaxSize.GetValue(), deviceProperty.mMaxDescriptorsSetUAVImage);
        mInternal->mSamplerHeap.mDescriptorTypeCount =
            std::min(cvVulkanBindlessDescriptorMaxSize.GetValue(), deviceProperty.mMaxDescriptorsSetSampler);
        mInternal->mRWStorageBufferHeap.mDescriptorTypeCount =
            std::min(cvVulkanBindlessDescriptorMaxSize.GetValue(), deviceProperty.mMaxDescriptorsSetUAVBuffer);
        mInternal->mReadOnlyStorageBufferHeap.mDescriptorTypeCount =
            std::min(cvVulkanBindlessDescriptorMaxSize.GetValue(), deviceProperty.mMaxDescriptorsSetSRVBuffer);

        mInternal->mUniformBufferHeap.mDescriptorTypeCount        = 1;
        mInternal->mCombinedImageSamplerHeap.mDescriptorTypeCount = 1;

        // Setup Layout
        Array<VA_DescHeapSpec*, kMaxDescriptorType> heaps = { &mInternal->mUniformBufferHeap,
            &mInternal->mRWStorageBufferHeap, &mInternal->mReadOnlyStorageBufferHeap,
            &mInternal->mCombinedImageSamplerHeap, &mInternal->mStorageImageHeap, &mInternal->mSampledImageHeap,
            &mInternal->mSamplerHeap };

        for (u32 i = 0; i < kMaxDescriptorType; ++i)
        {
            mInternal->mLayoutBindings[i].binding            = i;
            mInternal->mLayoutBindings[i].descriptorType     = static_cast<VkDescriptorType>(kDescriptorTypes[i]);
            mInternal->mLayoutBindings[i].descriptorCount    = heaps[i]->mDescriptorTypeCount;
            mInternal->mLayoutBindings[i].stageFlags         = VK_SHADER_STAGE_ALL;
            mInternal->mLayoutBindings[i].pImmutableSamplers = nullptr;
            if (i != 0)
                mInternal->mBindingFlags[i] =
                    VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
            else
                mInternal->mBindingFlags[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        }

        VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCI{};
        bindingFlagsCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        bindingFlagsCI.bindingCount  = kMaxDescriptorType;
        bindingFlagsCI.pBindingFlags = mInternal->mBindingFlags.data();

        VkDescriptorSetLayoutCreateInfo layoutCI{};
        layoutCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCI.bindingCount = kMaxDescriptorType;
        layoutCI.pBindings    = mInternal->mLayoutBindings.data();
        layoutCI.pNext        = &bindingFlagsCI;
        layoutCI.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;

        VA_AssertResult(vkCreateDescriptorSetLayout(
                            device->GetVulkanDevice(), &layoutCI, nullptr, &mInternal->mDescriptorSetLayout),
            "Failed to create descriptor set layout");

        // Setup Pool
        Array<VkDescriptorPoolSize, kMaxDescriptorType> poolSizes;
        for (u32 i = 0; i < kMaxDescriptorType; ++i)
        {
            poolSizes[i].type            = kDescriptorTypes[i];
            poolSizes[i].descriptorCount = heaps[i]->mDescriptorTypeCount;
        }

        VkDescriptorPoolCreateInfo poolCI{};
        poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        poolCI.maxSets       = 1;
        poolCI.poolSizeCount = kMaxDescriptorType;
        poolCI.pPoolSizes    = poolSizes.data();

        VA_AssertResult(
            vkCreateDescriptorPool(device->GetVulkanDevice(), &poolCI, nullptr, &mInternal->mDescriptorPool),
            "Failed to create descriptor pool");

        // Allocate Set
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool     = mInternal->mDescriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts        = &mInternal->mDescriptorSetLayout;
        VA_AssertResult(vkAllocateDescriptorSets(device->GetVulkanDevice(), &allocInfo, &mInternal->mDescriptorSet),
            "Failed to allocate descriptor set");
    }

    IFRIT_VKRHI2_API u32 VA_BindlessDescriptorHeap::RegisterSRV(VA_ResourceViewSRV& srv)
    {

        if (srv.IsBufferView())
        {
            IF_LOG_INFO("VVV", "IS buffer view");
            auto                   buffer = srv.GetUnderlyingBuffer()->GetRawHandle();
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = reinterpret_cast<VkBuffer>(buffer);
            bufferInfo.offset = srv.GetDesc().mBufferView.mOffset;
            bufferInfo.range  = srv.GetDesc().mBufferView.mSize != ~0u ? srv.GetDesc().mBufferView.mSize
                                                                       : srv.GetDesc().mBufferView.mSize;
            VkWriteDescriptorSet write{};
            write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet          = mInternal->mDescriptorSet;
            write.dstBinding      = 1; // SRV Buffer binding
            write.dstArrayElement = mInternal->mReadOnlyStorageBufferHeap.Allocate();
            write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo     = &bufferInfo;
            vkUpdateDescriptorSets(mInternal->mDevice->GetVulkanDevice(), 1, &write, 0, nullptr);
            return write.dstArrayElement;
        }
        else if (srv.IsTextureView())
        {
            auto                  texture = srv.GetUnderlyingTexture()->GetRawHandle();
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView   = reinterpret_cast<VkImageView>(srv.GetRawHandle());
            imageInfo.sampler     = VK_NULL_HANDLE;
            VkWriteDescriptorSet write{};
            write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet          = mInternal->mDescriptorSet;
            write.dstBinding      = 5; // SRV Image binding
            write.dstArrayElement = mInternal->mSampledImageHeap.Allocate();
            write.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            write.descriptorCount = 1;
            write.pImageInfo      = &imageInfo;
            vkUpdateDescriptorSets(mInternal->mDevice->GetVulkanDevice(), 1, &write, 0, nullptr);
            return write.dstArrayElement;
        }
        IF_LOG_CRITICAL("VulkanRHI2", "Invalid SRV resource view");
        return 0;
    }
    IFRIT_VKRHI2_API u32 VA_BindlessDescriptorHeap::RegisterUAV(VA_ResourceViewUAV& uav)
    {

        if (uav.IsBufferView())
        {
            auto                   buffer = uav.GetUnderlyingBuffer()->GetRawHandle();
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = reinterpret_cast<VkBuffer>(buffer);
            bufferInfo.offset = uav.GetDesc().mBufferView.mOffset;
            bufferInfo.range  = uav.GetDesc().mBufferView.mSize != ~0u ? uav.GetDesc().mBufferView.mSize : VK_WHOLE_SIZE;
            VkWriteDescriptorSet write{};
            write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet          = mInternal->mDescriptorSet;
            write.dstBinding      = 2; // UAV Buffer binding
            write.dstArrayElement = mInternal->mRWStorageBufferHeap.Allocate();
            write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo     = &bufferInfo;
            vkUpdateDescriptorSets(mInternal->mDevice->GetVulkanDevice(), 1, &write, 0, nullptr);
            return write.dstArrayElement;
        }
        else if (uav.IsTextureView())
        {
            auto                  texture = uav.GetUnderlyingTexture()->GetRawHandle();
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            imageInfo.imageView   = reinterpret_cast<VkImageView>(uav.GetRawHandle());
            imageInfo.sampler     = VK_NULL_HANDLE;
            VkWriteDescriptorSet write{};
            write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet          = mInternal->mDescriptorSet;
            write.dstBinding      = 4; // UAV Image binding
            write.dstArrayElement = mInternal->mStorageImageHeap.Allocate();
            write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write.descriptorCount = 1;
            write.pImageInfo      = &imageInfo;
            vkUpdateDescriptorSets(mInternal->mDevice->GetVulkanDevice(), 1, &write, 0, nullptr);
            return write.dstArrayElement;
        }
        IF_LOG_CRITICAL("VulkanRHI2", "Invalid UAV resource view");
        return 0;
    }
    IFRIT_VKRHI2_API u32 VA_BindlessDescriptorHeap::RegisterSampler(const RhiSampler& sampler)
    {
        u32                   retId = 0;
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView   = VK_NULL_HANDLE;
        imageInfo.sampler     = reinterpret_cast<VkSampler>(sampler.GetRawHandle());

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = mInternal->mDescriptorSet;
        write.dstBinding      = 6; // Sampler binding
        write.dstArrayElement = mInternal->mSamplerHeap.Allocate();
        write.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo      = &imageInfo;
        vkUpdateDescriptorSets(mInternal->mDevice->GetVulkanDevice(), 1, &write, 0, nullptr);
        retId = write.dstArrayElement;
        return retId;
    }

    IFRIT_VKRHI2_API u32 VA_BindlessDescriptorHeap::FreeSRVImage(u32 handleId)
    {
        mInternal->mSampledImageHeap.Free(handleId);
        return handleId;
    }
    IFRIT_VKRHI2_API u32 VA_BindlessDescriptorHeap::FreeUAVImage(u32 handleId)
    {
        mInternal->mStorageImageHeap.Free(handleId);
        return handleId;
    }
    IFRIT_VKRHI2_API u32 VA_BindlessDescriptorHeap::FreeSampler(u32 handleId)
    {
        mInternal->mSamplerHeap.Free(handleId);
        return handleId;
    }
    IFRIT_VKRHI2_API u32 VA_BindlessDescriptorHeap::FreeCBVBuffer(u32 handleId)
    {
        mInternal->mUniformBufferHeap.Free(handleId);
        return handleId;
    }
    IFRIT_VKRHI2_API u32 VA_BindlessDescriptorHeap::FreeSRVBuffer(u32 handleId)
    {
        mInternal->mReadOnlyStorageBufferHeap.Free(handleId);
        return handleId;
    }
    IFRIT_VKRHI2_API u32 VA_BindlessDescriptorHeap::FreeUAVBuffer(u32 handleId)
    {
        mInternal->mRWStorageBufferHeap.Free(handleId);
        return handleId;
    }

    IFRIT_VKRHI2_API VA_BindlessDescriptorHeap::~VA_BindlessDescriptorHeap()
    {

        if (mInternal->mDescriptorSetLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(
                mInternal->mDevice->GetVulkanDevice(), mInternal->mDescriptorSetLayout, nullptr);
            mInternal->mDescriptorSetLayout = VK_NULL_HANDLE;
        }
        if (mInternal->mDescriptorPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(mInternal->mDevice->GetVulkanDevice(), mInternal->mDescriptorPool, nullptr);
            mInternal->mDescriptorPool = VK_NULL_HANDLE;
        }
        delete mInternal;
    }

    IFRIT_VKRHI2_API VkDescriptorSet VA_BindlessDescriptorHeap::GetDescriptorSet() const
    {
        return mInternal->mDescriptorSet;
    }

    IFRIT_VKRHI2_API VkDescriptorSetLayout VA_BindlessDescriptorHeap::GetDescriptorSetLayout() const
    {
        return mInternal->mDescriptorSetLayout;
    }

} // namespace Ifrit::RHI::VulkanRHI2