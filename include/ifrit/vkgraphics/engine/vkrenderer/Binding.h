
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
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"
#include <array>
#include <map>
#include <memory>

namespace Ifrit::GraphicsBackend::VulkanGraphics {

struct DescriptorTypeDetails {
  VkDescriptorType type;
  uint32_t maxDescriptors;
};

constexpr uint32_t cMaxDescriptorType =
    static_cast<typename std::underlying_type<Rhi::RhiDescriptorType>::type>(
        Rhi::RhiDescriptorType::MaxEnum);

constexpr std::array<DescriptorTypeDetails, cMaxDescriptorType>
    cDescriptorTypeDetails = {
        {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 40000},
         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 40000},
         {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 40000},
         {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 40000}}};

struct DescriptorBindRange {
  uint32_t rangeId;
  uint32_t rangeOffset;
};

struct DescriptorBindRangeData {
  struct Range {
    std::vector<char> data;
    uint32_t offset;
    uint32_t bytes;
  };
  uint32_t m_currentOffset = 0;
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

  std::vector<SingleBuffer *> m_uniformBuffers;
  std::vector<SingleBuffer *> m_storageBuffers;
  std::vector<std::pair<SingleDeviceImage *, Sampler *>>
      m_combinedImageSamplers;

  std::vector<std::pair<SingleDeviceImage *, Rhi::RhiImageSubResource>>
      m_storageImages;

  uint32_t m_minUniformBufferAlignment = 0;

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

  uint32_t registerUniformBuffer(SingleBuffer *buffer);
  uint32_t registerCombinedImageSampler(SingleDeviceImage *image,
                                        Sampler *sampler);
  uint32_t registerStorageBuffer(SingleBuffer *buffer);
  uint32_t registerStorageImage(SingleDeviceImage *image,
                                Rhi::RhiImageSubResource subResource);

  DescriptorBindRange registerBindlessParameterRaw(const char *data,
                                                   uint32_t size);
  template <typename T>
  DescriptorBindRange registerBindlessParameter(const T &data) {
    return registerBindlessParameterRaw((char *)&data, sizeof(T));
  }
  void buildBindlessParameter();

  inline VkDescriptorSet getBindlessSet() const { return m_bindlessSet; }
  inline VkDescriptorSetLayout getBindlessLayout() const {
    return m_bindlessLayout;
  }
  inline VkDescriptorSet getParameterDescriptorSet(uint32_t rangeId) {
    if (rangeId >= m_bindRanges.size()) {
      buildBindlessParameter();
    }
    return m_bindRanges[rangeId]->m_set;
  }
  inline VkDescriptorSetLayout getParameterDescriptorSetLayout() {
    return m_layoutShared;
  }
};

class IFRIT_APIDECL DescriptorBindlessIndices
    : public Rhi::RhiBindlessDescriptorRef {
private:
  EngineContext *m_context;
  DescriptorManager *m_descriptorManager;
  std::vector<VkDescriptorSet> m_set;
  std::vector<DescriptorBindRange> m_bindRange;

  std::vector<std::map<uint32_t, uint32_t>> m_indices;
  uint32_t numCopies;
  uint32_t activeFrame = 0;

public:
  DescriptorBindlessIndices(EngineContext *ctx, DescriptorManager *manager,
                            uint32_t copies)
      : m_context(ctx), m_descriptorManager(manager) {
    m_indices.resize(copies);
    numCopies = copies;
  }

  inline virtual void addUniformBuffer(Rhi::RhiMultiBuffer *buffer,
                                       uint32_t loc) override {
    auto buf = Ifrit::Common::Utility::checked_cast<MultiBuffer>(buffer);
    for (uint32_t i = 0; i < numCopies; i++) {
      auto p = m_descriptorManager->registerUniformBuffer(buf->getBuffer(i));
      m_indices[i][loc] = p;
    }
  }

  inline virtual void addStorageBuffer(Rhi::RhiMultiBuffer *buffer,
                                       uint32_t loc) override {
    auto buf = Ifrit::Common::Utility::checked_cast<MultiBuffer>(buffer);
    for (uint32_t i = 0; i < numCopies; i++) {
      auto p = m_descriptorManager->registerStorageBuffer(buf->getBuffer(i));
      m_indices[i][loc] = p;
    }
  }

  inline virtual void addStorageBuffer(Rhi::RhiBuffer *buffer,
                                       uint32_t loc) override {
    auto buf = Ifrit::Common::Utility::checked_cast<SingleBuffer>(buffer);
    for (uint32_t i = 0; i < numCopies; i++) {
      auto p = m_descriptorManager->registerStorageBuffer(buf);
      m_indices[i][loc] = p;
    }
  }

  inline virtual void addCombinedImageSampler(Rhi::RhiTexture *texture,
                                              Rhi::RhiSampler *sampler,
                                              uint32_t loc) override {
    auto tex = Ifrit::Common::Utility::checked_cast<SingleDeviceImage>(texture);
    auto sam = Ifrit::Common::Utility::checked_cast<Sampler>(sampler);
    for (uint32_t i = 0; i < numCopies; i++) {
      auto p = m_descriptorManager->registerCombinedImageSampler(tex, sam);
      m_indices[i][loc] = p;
    }
  }

  inline virtual void addUAVImage(Rhi::RhiTexture *texture,
                                  Rhi::RhiImageSubResource subResource,
                                  uint32_t loc) override {
    auto tex = Ifrit::Common::Utility::checked_cast<SingleDeviceImage>(texture);
    for (uint32_t i = 0; i < numCopies; i++) {
      auto p = m_descriptorManager->registerStorageImage(tex, subResource);
      m_indices[i][loc] = p;
    }
  }

  inline void buildRanges() {
    using Ifrit::Common::Utility::size_cast;
    if (m_bindRange.size() == 0) {
      m_bindRange.resize(numCopies);
      for (uint32_t i = 0; i < numCopies; i++) {
        std::vector<uint32_t> uniformData;
        auto numKeys = m_indices[i].size();
        uniformData.resize(numKeys);
        for (auto &[k, v] : m_indices[i]) {
          uniformData[k] = v;
        }
        auto ptr = reinterpret_cast<const char *>(uniformData.data());
        m_bindRange[i] = m_descriptorManager->registerBindlessParameterRaw(
            ptr, size_cast<uint32_t>(numKeys * sizeof(uint32_t)));
      }
    }
  }

  inline virtual VkDescriptorSet getRangeSet(uint32_t frame) {
    buildRanges();
    return m_descriptorManager->getParameterDescriptorSet(
        m_bindRange[frame].rangeId);
  }

  inline uint32_t getRangeOffset(uint32_t frame) {
    buildRanges();
    return m_bindRange[frame].rangeOffset;
  }

  inline void setActiveFrame(uint32_t frame) { activeFrame = frame; }

  inline uint32_t getActiveRangeOffset() { return getRangeOffset(activeFrame); }
  inline VkDescriptorSet getActiveRangeSet() {
    return getRangeSet(activeFrame);
  }
};

} // namespace Ifrit::GraphicsBackend::VulkanGraphics