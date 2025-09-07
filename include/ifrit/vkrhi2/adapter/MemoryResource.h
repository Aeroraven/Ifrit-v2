
#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/core/algo/Parallel.h"
#include "ifrit/rhi/common/RhiCommandList.h"
#include <vulkan/vulkan.h>
#ifdef _WIN32
    #include <Windows.h>
#endif

namespace Ifrit::RHI::VulkanRHI2
{
    class VA_Device;

    struct VA_BufferInternal;
    class IFRIT_VKRHI2_API VA_Buffer : public RhiBuffer
    {
    public:
        VA_Buffer(RhiDevice* device, const RhiBufferDesc& desc);
        virtual ~VA_Buffer();

        void*                 MapMemory(RhiCommandListBase* cmdList, ERhiBufferMapType mapType);
        void                  UnmapMemory(RhiCommandListBase* cmdList);

        virtual RhiDeviceAddr GetDeviceAddress() const override;
        virtual RhiRawHandle  GetRawHandle() const override;

    private:
        VA_BufferInternal* mData;
    };

    struct VA_TextureInternal;
    class IFRIT_VKRHI2_API VA_Texture : public RhiTexture
    {
    public:
        VA_Texture(RhiDevice* device, const RhiTextureDesc& desc);
        virtual ~VA_Texture();

        virtual RhiRawHandle GetRawHandle() const override;

        // Vulkan Specific
        VkFormat             GetVkFormat() const;

    private:
        void SetInitialState(
            RhiCommandListBase* cmdList, VkImageLayout desiredInitLayout, bool doInitClear, RhiClearValue2 clearValue);
        VkImageSubresourceRange GetAllSubresourceRange();

    private:
        VA_TextureInternal* mData;
    };

    // ===== Staging Buffer =====

    struct VA_StagingBufferDesc
    {
        u64 mSize = 0;
    };

    struct VA_StagingBuffer
    {
        u64      mInBlockOffset = 0;
        u32      mBlockIdx      = 0;
        u32      mIsSingle      = 0;

        u64      mSize      = 0;
        void*    mMappedPtr = nullptr;

        VkBuffer mHandle = VK_NULL_HANDLE;
    };

    struct VA_StagingBufferManagerInternal;
    class VA_StagingBufferManager
    {
    public:
        VA_StagingBufferManager(VA_Device* device);
        ~VA_StagingBufferManager();

        VA_StagingBuffer AllocateStagingBuffer(u64 size);
        void             FreeStagingBuffer(const VA_StagingBuffer& buffer);

    protected:
        void AddSingleBlock(u64 size);
        void AddSmallBlock();

    private:
        VA_StagingBufferManagerInternal* mData;
    };

    // ===== Sampler State =====
    struct VA_SamplerInternal;
    class IFRIT_VKRHI2_API VA_Sampler : public RhiSampler
    {
    public:
        VA_Sampler(VA_Device* device, const RhiSamplerDesc& desc);
        virtual ~VA_Sampler();

        virtual RhiRawHandle GetRawHandle() const override;
        RhiDescriptorHandle  GetDescriptorHandle() const { return mHandle; }

    private:
        void RegisterSamplerDescriptor();
        void UnregisterSamplerDescriptor();

    private:
        VA_SamplerInternal* mData;
        RhiDescriptorHandle mHandle;
    };

    struct VA_SamplerRegistryInternal;
    class IFRIT_VKRHI2_API VA_SamplerRegistry : public NonCopyable
    {
    public:
        VA_SamplerRegistry(VA_Device* device);
        ~VA_SamplerRegistry();

        VA_Sampler* GetGlobalSampler(ERhiGlobalSamplerType type);

    private:
        VA_Sampler* CreateGlobalSampler(ERhiGlobalSamplerType type);

    private:
        VA_SamplerRegistryInternal* mData;
    };

} // namespace Ifrit::RHI::VulkanRHI2