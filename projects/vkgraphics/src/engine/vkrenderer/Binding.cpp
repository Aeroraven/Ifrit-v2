
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

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "ifrit/vkgraphics/engine/vkrenderer/Binding.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/utility/Logger.h"
#include <algorithm>

using namespace Ifrit::Common::Utility;

namespace Ifrit::GraphicsBackend::VulkanGraphics {
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
  }

  if (m_currentBindRange) {
    if (m_currentBindRange->m_pool != VK_NULL_HANDLE)
      vkDestroyDescriptorPool(m_context->getDevice(),
                              m_currentBindRange->m_pool, nullptr);
  }
  if (m_layoutShared != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(m_context->getDevice(), m_layoutShared,
                                 nullptr);
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
      size_cast<int>(properties.limits.minUniformBufferOffsetAlignment);

  m_currentBindRange = std::make_unique<DescriptorBindRangeData>();

  // Descriptor layout for bindless parameter
  m_bindingShared.binding = 0;
  m_bindingShared.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  m_bindingShared.descriptorCount = 1;
  m_bindingShared.stageFlags = VK_SHADER_STAGE_ALL;
  m_bindingShared.pImmutableSamplers = nullptr;

  // Create descriptor set layout
  VkDescriptorSetLayoutCreateInfo layoutCI2{};
  layoutCI2.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutCI2.bindingCount = 1;
  layoutCI2.pBindings = &m_bindingShared;

  vkrVulkanAssert(vkCreateDescriptorSetLayout(m_context->getDevice(),
                                              &layoutCI2, nullptr,
                                              &m_layoutShared),
                  "Failed to create descriptor set layout");
}

IFRIT_APIDECL uint32_t
DescriptorManager::registerUniformBuffer(SingleBuffer *buffer) {
  // check if the buffer is already registered
  for (int i = 0; i < m_uniformBuffers.size(); i++) {
    if (m_uniformBuffers[i] == buffer->getBuffer()) {
      return i;
    }
  }
  auto handleId = size_cast<int>(m_uniformBuffers.size());
  m_uniformBuffers.push_back(buffer->getBuffer());

  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = buffer->getBuffer();
  bufferInfo.offset = 0;
  bufferInfo.range = VK_WHOLE_SIZE;

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = m_bindlessSet;
  write.dstBinding = getUnderlying(Rhi::RhiDescriptorType::UniformBuffer);
  write.dstArrayElement = handleId;
  write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  write.descriptorCount = 1;
  write.pBufferInfo = &bufferInfo;

  vkUpdateDescriptorSets(m_context->getDevice(), 1, &write, 0, nullptr);
  return handleId;
}

IFRIT_APIDECL
uint32_t DescriptorManager::registerStorageBuffer(SingleBuffer *buffer) {
  for (int i = 0; i < m_storageBuffers.size(); i++) {
    if (m_storageBuffers[i] == buffer->getBuffer()) {
      return i;
    }
  }
  auto handleId = m_storageBuffers.size();
  m_storageBuffers.push_back(buffer->getBuffer());

  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = buffer->getBuffer();
  bufferInfo.offset = 0;
  bufferInfo.range = VK_WHOLE_SIZE;

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = m_bindlessSet;
  write.dstBinding = getUnderlying(Rhi::RhiDescriptorType::StorageBuffer);
  write.dstArrayElement = size_cast<uint32_t>(handleId);
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.descriptorCount = 1;
  write.pBufferInfo = &bufferInfo;

  vkUpdateDescriptorSets(m_context->getDevice(), 1, &write, 0, nullptr);
  return size_cast<uint32_t>(handleId);
}

IFRIT_APIDECL
uint32_t
DescriptorManager::registerStorageImage(SingleDeviceImage *image,
                                        Rhi::RhiImageSubResource subResource) {
  for (int i = 0; i < m_storageImages.size(); i++) {
    if (m_storageImages[i].first == image->getImage()) {
      auto &sub = m_storageImages[i].second;
      if (sub.mipLevel == subResource.mipLevel &&
          sub.arrayLayer == subResource.arrayLayer &&
          sub.mipCount == subResource.mipCount &&
          sub.layerCount == subResource.layerCount) {
        return i;
      }
    }
  }
  auto handle = m_storageImages.size();
  VkDescriptorImageInfo imageInfo{};
  imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  imageInfo.imageView =
      image->getImageViewMipLayer(subResource.mipLevel, subResource.arrayLayer,
                                  subResource.mipCount, subResource.layerCount);
  imageInfo.sampler = VK_NULL_HANDLE;

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = m_bindlessSet;
  write.dstBinding = getUnderlying(Rhi::RhiDescriptorType::StorageImage);
  write.dstArrayElement = size_cast<uint32_t>(handle);
  ;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  write.descriptorCount = 1;
  write.pImageInfo = &imageInfo;

  vkUpdateDescriptorSets(m_context->getDevice(), 1, &write, 0, nullptr);
  m_storageImages.push_back({image->getImage(), subResource});
  return size_cast<uint32_t>(handle);
}

IFRIT_APIDECL
uint32_t
DescriptorManager::registerCombinedImageSampler(SingleDeviceImage *image,
                                                Sampler *sampler) {
  for (int i = 0; i < m_combinedImageSamplers.size(); i++) {
    if (m_combinedImageSamplers[i].first == image->getImage() &&
        m_combinedImageSamplers[i].second == sampler->getSampler()) {
      return i;
    }
  }
  auto handleId = m_combinedImageSamplers.size();
  m_combinedImageSamplers.push_back({image->getImage(), sampler->getSampler()});

  VkDescriptorImageInfo imageInfo{};
  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  imageInfo.imageView = image->getImageView();
  imageInfo.sampler = sampler->getSampler();

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = m_bindlessSet;
  write.dstBinding =
      getUnderlying(Rhi::RhiDescriptorType::CombinedImageSampler);
  write.dstArrayElement = size_cast<uint32_t>(handleId);
  write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write.descriptorCount = 1;
  write.pImageInfo = &imageInfo;

  vkUpdateDescriptorSets(m_context->getDevice(), 1, &write, 0, nullptr);
  return size_cast<uint32_t>(handleId);
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
    m_currentBindRange->m_buffer->writeBuffer(range.data.data(), range.bytes,
                                              range.offset);
  }
  m_currentBindRange->m_buffer->flush();
  m_currentBindRange->m_buffer->unmap();

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
  allocInfo.pSetLayouts = &m_layoutShared;

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

} // namespace Ifrit::GraphicsBackend::VulkanGraphics