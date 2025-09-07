#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/adapter/MemoryResource.h"

#include <vulkan/vulkan.h>

namespace Ifrit::RHI::VulkanRHI2
{
    // ===== Descriptor =====
    enum class EVA_ViewType
    {
        SRV,
        UAV,
        CBV,
        Sampler,
    };

    struct VA_ResourceViewInternal;
    class IFRIT_VKRHI2_API VA_ResourceView
    {
    public:
        VA_ResourceView(const RhiResourceViewDesc& desc, VA_Texture* texture, VA_Device* device, EVA_ViewType type);
        VA_ResourceView(const RhiResourceViewDesc& desc, VA_Buffer* buffer, VA_Device* device, EVA_ViewType type);
        ~VA_ResourceView();

        virtual void         AcquireHandleInternal(void* ptr);
        virtual void         ReleaseHandleInternal(void* ptr);
        virtual RhiRawHandle GetRawHandleInternal() const;

    protected:
        VA_ResourceViewInternal* mInternal;
    };

    class IFRIT_VKRHI2_API VA_ResourceViewSRV : protected VA_ResourceView, public RhiShaderReadView
    {
    public:
        VA_ResourceViewSRV(const RhiResourceViewDesc& desc, VA_Texture* texture, VA_Device* device);
        VA_ResourceViewSRV(const RhiResourceViewDesc& desc, VA_Buffer* buffer, VA_Device* device);
        ~VA_ResourceViewSRV();

        virtual void         AcquireHandle() override { VA_ResourceView::AcquireHandleInternal(this); }
        virtual void         ReleaseHandle() override { VA_ResourceView::ReleaseHandleInternal(this); }
        virtual RhiRawHandle GetRawHandle() const override { return VA_ResourceView::GetRawHandleInternal(); }
    };

    class IFRIT_VKRHI2_API VA_ResourceViewUAV : protected VA_ResourceView, public RhiUnorderedAccessView
    {
    public:
        VA_ResourceViewUAV(const RhiResourceViewDesc& desc, VA_Texture* texture, VA_Device* device);
        VA_ResourceViewUAV(const RhiResourceViewDesc& desc, VA_Buffer* buffer, VA_Device* device);
        ~VA_ResourceViewUAV();

        virtual void         AcquireHandle() override { VA_ResourceView::AcquireHandleInternal(this); }
        virtual void         ReleaseHandle() override { VA_ResourceView::ReleaseHandleInternal(this); }
        virtual RhiRawHandle GetRawHandle() const override { return VA_ResourceView::GetRawHandleInternal(); }
    };

    // ===== Descriptor Heap =====
    struct VA_BindlessDescriptorHeapInternal;
    class IFRIT_VKRHI2_API VA_BindlessDescriptorHeap
    {
    public:
        VA_BindlessDescriptorHeap(VA_Device* device);
        ~VA_BindlessDescriptorHeap();

    public:
        u32                   RegisterSRV(VA_ResourceViewSRV& srv);
        u32                   RegisterUAV(VA_ResourceViewUAV& uav);
        u32                   RegisterSampler(const RhiSampler& sampler);

        u32                   FreeSRVImage(u32 handleId);
        u32                   FreeUAVImage(u32 handleId);
        u32                   FreeSampler(u32 handleId);
        u32                   FreeCBVBuffer(u32 handleId);
        u32                   FreeSRVBuffer(u32 handleId);
        u32                   FreeUAVBuffer(u32 handleId);

        VkDescriptorSet       GetDescriptorSet() const;
        VkDescriptorSetLayout GetDescriptorSetLayout() const;

    private:
        VA_BindlessDescriptorHeapInternal* mInternal;
    };

} // namespace Ifrit::RHI::VulkanRHI2