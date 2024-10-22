#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <algorithm>
#include <vkrenderer/include/engine/vkrenderer/Binding.h>
#include <vkrenderer/include/utility/Logger.h>

namespace Ifrit::Engine::VkRenderer {
template <typename E>
constexpr typename std::underlying_type<E>::type getUnderlying(E e) noexcept {
  return static_cast<typename std::underlying_type<E>::type>(e);
}

IFRIT_APIDECL void DescriptorManager::destructor() {
  vkDestroyDescriptorPool(m_context->getDevice(), m_bindlessPool, nullptr);
  vkDestroyDescriptorSetLayout(m_context->getDevice(), m_bindlessLayout,
                               nullptr);
  // iterates m_bindRages and delete descriptor pool and layout
  for (auto &range : m_bindRanges) {
    vkDestroyDescriptorPool(m_context->getDevice(), range->m_pool, nullptr);
    vkDestroyDescriptorSetLayout(m_context->getDevice(), range->m_layout,
                                 nullptr);
  }

  if (m_currentBindRange) {
    if (m_currentBindRange->m_pool != VK_NULL_HANDLE)
      vkDestroyDescriptorPool(m_context->getDevice(),
                              m_currentBindRange->m_pool, nullptr);
    if (m_currentBindRange->m_layout != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(m_context->getDevice(),
                                   m_currentBindRange->m_layout, nullptr);
  }
}
IFRIT_APIDECL DescriptorManager::DescriptorManager(EngineContext *ctx)
    : m_context(ctx) {
  for (int i = 0; i < cMaxDescriptorType; i++) {
    m_bindings[i].binding = i;
    m_bindings[i].descriptorType = cDescriptorTypeDetails[i].type;
    m_bindings[i].descriptorCount = cDescriptorTypeDetails[i].maxDescriptors;
    m_bindings[i].stageFlags = VK_SHADER_STAGE_ALL;
    m_bindings[i].pImmutableSamplers = nullptr;

    m_bindingFlags[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
  }
  VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCI{};
  bindingFlagsCI.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
  bindingFlagsCI.bindingCount = cMaxDescriptorType;
  bindingFlagsCI.pBindingFlags = m_bindingFlags.data();

  VkDescriptorSetLayoutCreateInfo layoutCI{};
  layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutCI.bindingCount = cMaxDescriptorType;
  layoutCI.pBindings = m_bindings.data();
  layoutCI.pNext = &bindingFlagsCI;
  layoutCI.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;

  vkrVulkanAssert(vkCreateDescriptorSetLayout(m_context->getDevice(), &layoutCI,
                                              nullptr, &m_bindlessLayout),
                  "Failed to create descriptor set layout");

  std::array<VkDescriptorPoolSize, cMaxDescriptorType> poolSizes;
  for (int i = 0; i < cMaxDescriptorType; i++) {
    poolSizes[i].type = cDescriptorTypeDetails[i].type;
    poolSizes[i].descriptorCount = cDescriptorTypeDetails[i].maxDescriptors;
  }
  // Create pool
  VkDescriptorPoolCreateInfo poolCI{};
  poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCI.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
  poolCI.maxSets = 1;
  poolCI.poolSizeCount = cMaxDescriptorType;
  poolCI.pPoolSizes = poolSizes.data();

  vkrVulkanAssert(vkCreateDescriptorPool(m_context->getDevice(), &poolCI,
                                         nullptr, &m_bindlessPool),
                  "Failed to create descriptor pool");

  // Allocate descriptor set
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = m_bindlessPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &m_bindlessLayout;

  vkrVulkanAssert(vkAllocateDescriptorSets(m_context->getDevice(), &allocInfo,
                                           &m_bindlessSet),
                  "Failed to allocate descriptor set");

  // Query alignment for dynamic uniform buffer
  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(m_context->getPhysicalDevice(), &properties);
  m_minUniformBufferAlignment =
      properties.limits.minUniformBufferOffsetAlignment;

  m_currentBindRange = std::make_unique<DescriptorBindRangeData>();
}

IFRIT_APIDECL uint32_t
DescriptorManager::registerUniformBuffer(SingleBuffer *buffer) {
  // check if the buffer is already registered
  for (int i = 0; i < m_uniformBuffers.size(); i++) {
    if (m_uniformBuffers[i] == buffer) {
      return i;
    }
  }
  printf("Add uniform buffer\n");

  auto handleId = m_uniformBuffers.size();
  m_uniformBuffers.push_back(buffer);

  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = buffer->getBuffer();
  bufferInfo.offset = 0;
  bufferInfo.range = VK_WHOLE_SIZE;

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = m_bindlessSet;
  write.dstBinding = getUnderlying(DescriptorType::UniformBuffer);
  write.dstArrayElement = handleId;
  write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  write.descriptorCount = 1;
  write.pBufferInfo = &bufferInfo;

  vkUpdateDescriptorSets(m_context->getDevice(), 1, &write, 0, nullptr);
  return handleId;
}

IFRIT_APIDECL DescriptorBindRange
DescriptorManager::registerBindlessParameterRaw(const char *data,
                                                uint32_t size) {
  auto offset = m_currentBindRange->m_currentOffset;
  auto rangeId = m_currentBindRange->m_ranges.size();

  auto paddedSize = size;
  if (size % m_minUniformBufferAlignment != 0) {
    paddedSize +=
        m_minUniformBufferAlignment - (size % m_minUniformBufferAlignment);
  }
  m_currentBindRange->m_currentOffset += paddedSize;

  DescriptorBindRangeData::Range range;
  range.offset = offset;
  range.bytes = size;
  range.data.resize(size);
  memcpy(range.data.data(), data, size);
  m_currentBindRange->m_ranges.push_back(range);

  return {static_cast<uint32_t>(m_bindRanges.size()), offset};
}

IFRIT_APIDECL void DescriptorManager::buildBindlessParameter() {
  BufferCreateInfo ci{};
  ci.size = m_currentBindRange->m_currentOffset;
  ci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  ci.hostVisible = true;
  m_currentBindRange->m_buffer = std::make_unique<SingleBuffer>(m_context, ci);

  m_currentBindRange->m_buffer->map();
  for (int i = 0; i < m_currentBindRange->m_ranges.size(); i++) {
    auto &range = m_currentBindRange->m_ranges[i];
    m_currentBindRange->m_buffer->copyFromBuffer(range.data.data(), range.bytes,
                                                 range.offset);
  }
  m_currentBindRange->m_buffer->flush();
  m_currentBindRange->m_buffer->unmap();

  m_currentBindRange->m_binding.binding = 0;
  m_currentBindRange->m_binding.descriptorType =
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  m_currentBindRange->m_binding.descriptorCount = 1;
  m_currentBindRange->m_binding.stageFlags = VK_SHADER_STAGE_ALL;
  m_currentBindRange->m_binding.pImmutableSamplers = nullptr;

  // Create descriptor set layout
  VkDescriptorSetLayoutCreateInfo layoutCI{};
  layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutCI.bindingCount = 1;
  layoutCI.pBindings = &m_currentBindRange->m_binding;

  vkrVulkanAssert(vkCreateDescriptorSetLayout(m_context->getDevice(), &layoutCI,
                                              nullptr,
                                              &m_currentBindRange->m_layout),
                  "Failed to create descriptor set layout");

  // Create pool
  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  poolSize.descriptorCount = 1;

  VkDescriptorPoolCreateInfo poolCI{};
  poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCI.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
  poolCI.maxSets = 1;
  poolCI.poolSizeCount = 1;
  poolCI.pPoolSizes = &poolSize;

  vkrVulkanAssert(vkCreateDescriptorPool(m_context->getDevice(), &poolCI,
                                         nullptr, &m_currentBindRange->m_pool),
                  "Failed to create descriptor pool");

  // Allocate descriptor set
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = m_currentBindRange->m_pool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &m_currentBindRange->m_layout;

  vkrVulkanAssert(vkAllocateDescriptorSets(m_context->getDevice(), &allocInfo,
                                           &m_currentBindRange->m_set),
                  "Failed to allocate descriptor set");

  // Update descriptor set
  uint32_t maxRange = 0;
  for (int i = 0; i < m_currentBindRange->m_ranges.size(); i++) {
    maxRange = std::max(maxRange, m_currentBindRange->m_ranges[i].bytes);
  }
  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = m_currentBindRange->m_buffer->getBuffer();
  bufferInfo.offset = 0;
  bufferInfo.range = maxRange;

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = m_currentBindRange->m_set;
  write.dstBinding = 0;
  write.dstArrayElement = 0;
  write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  write.descriptorCount = 1;
  write.pBufferInfo = &bufferInfo;

  vkUpdateDescriptorSets(m_context->getDevice(), 1, &write, 0, nullptr);

  m_bindRanges.push_back(std::move(m_currentBindRange));
  m_currentBindRange = std::make_unique<DescriptorBindRangeData>();
}

} // namespace Ifrit::Engine::VkRenderer