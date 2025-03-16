
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"
#include <array>
#include <map>
#include <memory>

namespace Ifrit::GraphicsBackend::VulkanGraphics {

struct DescriptorTypeDetails {
  VkDescriptorType type;
  u32 maxDescriptors;
};

IF_CONSTEXPR u32 cMaxDescriptorType =
    static_cast<typename std::underlying_type<Rhi::RhiDescriptorType>::type>(Rhi::RhiDescriptorType::MaxEnum);

IF_CONSTEXPR std::array<DescriptorTypeDetails, cMaxDescriptorType> cDescriptorTypeDetails = {
    {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 40000},
     {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 40000},
     {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 40000},
     {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 40000}}};

struct DescriptorBindRange {
  u32 rangeId;
  u32 rangeOffset;
};

struct DescriptorBindRangeData {
  struct Range {
    std::vector<char> data;
    u32 offset;
    u32 bytes;
  };
  u32 m_currentOffset = 0;
  std::vector<Range> m_ranges;
  VkDescriptorPool m_pool{};
  VkDescriptorSet m_set{};

  std::unique_ptr<SingleBuffer> m_buffer;
};

class IFRIT_APIDECL DescriptorManager {
private:
  EngineContext *m_context;
  std::array<VkDescriptorSetLayoutBinding, cMaxDescriptorType> m_bindings;
  std::array<VkDescriptorBindingFlagsEXT, cMaxDescriptorType> m_bindingFlags;
  VkDescriptorSetLayout m_bindlessLayout;
  VkDescriptorPool m_bindlessPool;
  VkDescriptorSet m_bindlessSet;

  std::vector<VkBuffer> m_uniformBuffers;
  std::vector<VkBuffer> m_storageBuffers;
  std::vector<std::pair<VkImage, VkSampler>> m_combinedImageSamplers;

  std::vector<std::pair<VkImage, Rhi::RhiImageSubResource>> m_storageImages;

  u32 m_minUniformBufferAlignment = 0;

  std::vector<std::unique_ptr<DescriptorBindRangeData>> m_bindRanges;
  std::unique_ptr<DescriptorBindRangeData> m_currentBindRange;

  VkDescriptorSetLayout m_layoutShared = VK_NULL_HANDLE;
  VkDescriptorSetLayoutBinding m_bindingShared{};

protected:
  void destructor();

public:
  DescriptorManager(EngineContext *ctx);
  DescriptorManager(const DescriptorManager &p) = delete;
  DescriptorManager &operator=(const DescriptorManager &p) = delete;
  virtual ~DescriptorManager() { destructor(); }

  u32 registerUniformBuffer(SingleBuffer *buffer);
  u32 registerCombinedImageSampler(SingleDeviceImage *image, Sampler *sampler);
  u32 registerStorageBuffer(SingleBuffer *buffer);
  u32 registerStorageImage(SingleDeviceImage *image, Rhi::RhiImageSubResource subResource);

  DescriptorBindRange registerBindlessParameterRaw(const char *data, u32 size);
  template <typename T> DescriptorBindRange registerBindlessParameter(const T &data) {
    return registerBindlessParameterRaw((char *)&data, sizeof(T));
  }
  void buildBindlessParameter();

  inline VkDescriptorSet getBindlessSet() const { return m_bindlessSet; }
  inline VkDescriptorSetLayout getBindlessLayout() const { return m_bindlessLayout; }
  inline VkDescriptorSet getParameterDescriptorSet(u32 rangeId) {
    if (rangeId >= m_bindRanges.size()) {
      buildBindlessParameter();
    }
    return m_bindRanges[rangeId]->m_set;
  }
  inline VkDescriptorSetLayout getParameterDescriptorSetLayout() { return m_layoutShared; }
};

class IFRIT_APIDECL DescriptorBindlessIndices : public Rhi::RhiBindlessDescriptorRef {
private:
  EngineContext *m_context;
  DescriptorManager *m_descriptorManager;
  std::vector<VkDescriptorSet> m_set;
  std::vector<DescriptorBindRange> m_bindRange;

  std::vector<std::map<u32, u32>> m_indices;
  u32 numCopies;
  u32 activeFrame = 0;

public:
  DescriptorBindlessIndices(EngineContext *ctx, DescriptorManager *manager, u32 copies)
      : m_context(ctx), m_descriptorManager(manager), numCopies(copies) {
    m_indices.resize(copies);
  }

  virtual void addUniformBuffer(Rhi::RhiMultiBuffer *buffer, u32 loc) override;
  virtual void addStorageBuffer(Rhi::RhiMultiBuffer *buffer, u32 loc) override;
  virtual void addStorageBuffer(Rhi::RhiBuffer *buffer, u32 loc) override;
  virtual void addCombinedImageSampler(Rhi::RhiTexture *texture, Rhi::RhiSampler *sampler, u32 loc) override;
  virtual void addUAVImage(Rhi::RhiTexture *texture, Rhi::RhiImageSubResource subResource, u32 loc) override;

  void buildRanges();

  inline virtual VkDescriptorSet getRangeSet(u32 frame) {
    buildRanges();
    return m_descriptorManager->getParameterDescriptorSet(m_bindRange[frame].rangeId);
  }

  inline u32 getRangeOffset(u32 frame) {
    buildRanges();
    return m_bindRange[frame].rangeOffset;
  }

  inline void setActiveFrame(u32 frame) { activeFrame = frame; }
  inline u32 getActiveRangeOffset() { return getRangeOffset(activeFrame); }
  inline VkDescriptorSet getActiveRangeSet() { return getRangeSet(activeFrame); }
};

} // namespace Ifrit::GraphicsBackend::VulkanGraphics