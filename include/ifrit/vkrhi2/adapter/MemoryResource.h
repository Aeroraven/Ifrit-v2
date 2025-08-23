
#pragma once
#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/core/algo/Parallel.h"

#include <vulkan/vulkan.h>
#ifdef _WIN32
    #include <Windows.h>
#endif

namespace Ifrit::RHI::VulkanRHI2
{
    struct VA_BufferInternal;
    class IFRIT_VKRHI2_API VA_Buffer : public RhiBuffer
    {
    public:
        VA_Buffer(RhiDevice* device, const RhiBufferDesc& desc);
        virtual ~VA_Buffer();

    private:
        VA_BufferInternal* mData;
    };

    struct VA_TextureInternal;
    class IFRIT_VKRHI2_API VA_Texture : public RhiTexture
    {
    public:
        VA_Texture(RhiDevice* device, const RhiTextureDesc& desc);
        virtual ~VA_Texture();

    private:
        VA_TextureInternal* mData;
    };
} // namespace Ifrit::RHI::VulkanRHI2