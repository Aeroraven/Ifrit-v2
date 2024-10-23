#pragma once
#include <array>
#include <memory>
#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>
#include <vkrenderer/include/engine/vkrenderer/MemoryResource.h>

namespace Ifrit::Engine::VkRenderer {

enum class DescriptorType {
  UniformBuffer,
  StorageBuffer,
  CombinedImageSampler,
  StorageImage,
  MaxEnum
};

struct DescriptorTypeDetails {
  VkDescriptorType type;
  uint32_t maxDescriptors;
};

constexpr uint32_t cMaxDescriptorType =
    static_cast<typename std::underlying_type<DescriptorType>::type>(
        DescriptorType::MaxEnum);

constexpr std::array<DescriptorTypeDetails, cMaxDescriptorType>
    cDescriptorTypeDetails = {
        {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10000},
         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10000},
         {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10000},
         {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10000}}};

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
  VkDescriptorSetLayout m_layout{};
  VkDescriptorSetLayoutBinding m_binding{};
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

  uint32_t m_minUniformBufferAlignment = 0;

  std::vector<std::unique_ptr<DescriptorBindRangeData>> m_bindRanges;
  std::unique_ptr<DescriptorBindRangeData> m_currentBindRange;

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
  inline VkDescriptorSet getParameterDescriptorSet(uint32_t rangeId) const {
    return m_bindRanges[rangeId]->m_set;
  }
  inline VkDescriptorSetLayout
  getParameterDescriptorSetLayout(uint32_t rangeId) const {
    return m_bindRanges[rangeId]->m_layout;
  }
};
} // namespace Ifrit::Engine::VkRenderer