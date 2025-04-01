
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

#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"
#include <array>
#include <map>
#include <memory>

namespace Ifrit::Graphics::VulkanGraphics
{

    struct DescriptorTypeDetails
    {
        VkDescriptorType type;
        u32              maxDescriptors;
    };

    IF_CONSTEXPR u32 cMaxDescriptorType =
        static_cast<typename std::underlying_type<Rhi::RhiDescriptorType>::type>(Rhi::RhiDescriptorType::MaxEnum);

    IF_CONSTEXPR Array<DescriptorTypeDetails, cMaxDescriptorType> cDescriptorTypeDetails = {
        { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 40000 }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 40000 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 40000 }, { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 40000 } }
    };

    struct DescriptorBindRange
    {
        u32 rangeId;
        u32 rangeOffset;
    };

    struct DescriptorBindRangeData
    {
        struct Range
        {
            Vec<char> data;
            u32       offset;
            u32       bytes;
        };
        u32                m_currentOffset = 0;
        Vec<Range>         m_ranges;
        VkDescriptorPool   m_pool{};
        VkDescriptorSet    m_set{};

        Uref<SingleBuffer> m_buffer;
    };

    class IFRIT_APIDECL DescriptorManager
    {
    private:
        EngineContext*                                          m_context;
        Array<VkDescriptorSetLayoutBinding, cMaxDescriptorType> m_bindings;
        Array<VkDescriptorBindingFlagsEXT, cMaxDescriptorType>  m_bindingFlags;
        VkDescriptorSetLayout                                   m_bindlessLayout;
        VkDescriptorPool                                        m_bindlessPool;
        VkDescriptorSet                                         m_bindlessSet;

        Vec<VkBuffer>                                           m_uniformBuffers;
        Vec<VkBuffer>                                           m_storageBuffers;
        Vec<Pair<VkImage, VkSampler>>                           m_combinedImageSamplers;

        Vec<Pair<VkImage, Rhi::RhiImageSubResource>>            m_storageImages;

        u32                                                     m_minUniformBufferAlignment = 0;

        Vec<Uref<DescriptorBindRangeData>>                      m_bindRanges;
        Uref<DescriptorBindRangeData>                           m_currentBindRange;

        VkDescriptorSetLayout                                   m_layoutShared = VK_NULL_HANDLE;
        VkDescriptorSetLayoutBinding                            m_bindingShared{};

    protected:
        void Destructor();

    public:
        DescriptorManager(EngineContext* ctx);
        DescriptorManager(const DescriptorManager& p)            = delete;
        DescriptorManager& operator=(const DescriptorManager& p) = delete;
        virtual ~DescriptorManager() { Destructor(); }

        u32                    RegisterUniformBuffer(SingleBuffer* buffer);
        u32                    RegisterCombinedImageSampler(SingleDeviceImage* image, Sampler* sampler);
        u32                    RegisterStorageBuffer(SingleBuffer* buffer);
        u32                    RegisterStorageImage(SingleDeviceImage* image, Rhi::RhiImageSubResource subResource);
        DescriptorBindRange    RegisterBindlessParameterRaw(const char* data, u32 size);

        void                   BuildBindlessParameter();

        inline VkDescriptorSet GetBindlessSet() const { return m_bindlessSet; }
        inline VkDescriptorSetLayout GetBindlessLayout() const { return m_bindlessLayout; }
        inline VkDescriptorSet       GetParameterDescriptorSet(u32 rangeId)
        {
            if (rangeId >= m_bindRanges.size())
            {
                BuildBindlessParameter();
            }
            return m_bindRanges[rangeId]->m_set;
        }
        template <typename T> DescriptorBindRange RegisterBindlessParameter(const T& data)
        {
            return RegisterBindlessParameterRaw((char*)&data, sizeof(T));
        }
        inline VkDescriptorSetLayout GetParameterDescriptorSetLayout() { return m_layoutShared; }
    };

    class IFRIT_APIDECL DescriptorBindlessIndices : public Rhi::RhiBindlessDescriptorRef
    {
    private:
        EngineContext*           m_context;
        DescriptorManager*       m_descriptorManager;
        Vec<VkDescriptorSet>     m_set;
        Vec<DescriptorBindRange> m_bindRange;

        Vec<Map<u32, u32>>       m_indices;
        u32                      numCopies;
        u32                      activeFrame = 0;

    public:
        DescriptorBindlessIndices(EngineContext* ctx, DescriptorManager* manager, u32 copies)
            : m_context(ctx), m_descriptorManager(manager), numCopies(copies)
        {
            m_indices.resize(copies);
        }

        virtual void AddUniformBuffer(Rhi::RhiMultiBuffer* buffer, u32 loc) override;
        virtual void AddStorageBuffer(Rhi::RhiMultiBuffer* buffer, u32 loc) override;
        virtual void AddStorageBuffer(Rhi::RhiBuffer* buffer, u32 loc) override;
        virtual void AddCombinedImageSampler(Rhi::RhiTexture* texture, Rhi::RhiSampler* sampler, u32 loc) override;
        virtual void AddUAVImage(Rhi::RhiTexture* texture, Rhi::RhiImageSubResource subResource, u32 loc) override;
        void         BuildRanges();

        inline virtual VkDescriptorSet GetRangeSet(u32 frame)
        {
            BuildRanges();
            return m_descriptorManager->GetParameterDescriptorSet(m_bindRange[frame].rangeId);
        }

        inline u32 GetRangeOffset(u32 frame)
        {
            BuildRanges();
            return m_bindRange[frame].rangeOffset;
        }

        inline void            SetActiveFrame(u32 frame) { activeFrame = frame; }
        inline u32             GetActiveRangeOffset() { return GetRangeOffset(activeFrame); }
        inline VkDescriptorSet GetActiveRangeSet() { return GetRangeSet(activeFrame); }
    };

} // namespace Ifrit::Graphics::VulkanGraphics