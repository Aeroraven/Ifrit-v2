
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

#include "ifrit/vkgraphics/engine/vkrenderer/Backend.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/engine/vkrenderer/RenderGraph.h"
#include "ifrit/vkgraphics/engine/vkrenderer/RenderTargets.h"
#include "ifrit/vkgraphics/engine/vkrenderer/StagedMemoryResource.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Timer.h"

#include "ifrit/common/logging/Logging.h"
#include "ifrit/vkgraphics/engine/fsr2extension/FSR2Processor.h"

using namespace Ifrit::Common::Utility;

namespace Ifrit::GraphicsBackend::VulkanGraphics {
inline VkFormat toVkFormat(Rhi::RhiImageFormat format) { return static_cast<VkFormat>(format); }

struct RhiVulkanBackendImplDetails : public NonCopyable {
  std::unique_ptr<CommandExecutor> m_commandExecutor;
  std::unique_ptr<DescriptorManager> m_descriptorManager;
  std::unique_ptr<ResourceManager> m_resourceManager;
  std::vector<std::unique_ptr<StagedSingleBuffer>> m_stagedSingleBuffer;
  std::vector<std::unique_ptr<ShaderModule>> m_shaderModule;
  std::unique_ptr<PipelineCache> m_pipelineCache;

  std::unique_ptr<RegisteredResourceMapper> m_mapper;

  // managed passes
  std::vector<std::unique_ptr<ComputePass>> m_computePasses;
  std::vector<std::unique_ptr<GraphicsPass>> m_graphicsPasses;
  std::vector<std::unique_ptr<DescriptorBindlessIndices>> m_bindlessIndices;

  // managed descriptors
  std::vector<std::shared_ptr<Rhi::RhiBindlessIdRef>> m_bindlessIdRefs;

  // some utility buffers
  Rhi::RhiBufferRef m_fullScreenQuadVertexBuffer;
  std::shared_ptr<VertexBufferDescriptor> m_fullScreenQuadVertexBufferDescriptor;

  // timers
  std::vector<std::shared_ptr<DeviceTimer>> m_deviceTimers;
};

IFRIT_APIDECL
RhiVulkanBackend::RhiVulkanBackend(const Rhi::RhiInitializeArguments &args) {
  m_device = std::make_unique<EngineContext>(args);
  auto engineContext = checked_cast<EngineContext>(m_device.get());
  m_swapChain = std::make_unique<Swapchain>(engineContext);
  auto swapchain = checked_cast<Swapchain>(m_swapChain.get());
  m_implDetails = new RhiVulkanBackendImplDetails();
  m_implDetails->m_descriptorManager = std::make_unique<DescriptorManager>(engineContext);
  m_implDetails->m_resourceManager = std::make_unique<ResourceManager>(engineContext);
  m_implDetails->m_commandExecutor = std::make_unique<CommandExecutor>(
      engineContext, swapchain, m_implDetails->m_descriptorManager.get(), m_implDetails->m_resourceManager.get());
  m_implDetails->m_pipelineCache = std::make_unique<PipelineCache>(engineContext);
  m_implDetails->m_mapper = std::make_unique<RegisteredResourceMapper>();
  m_implDetails->m_commandExecutor->setQueues(1, args.m_expectedGraphicsQueueCount, args.m_expectedComputeQueueCount,
                                              args.m_expectedTransferQueueCount);

  // All done, then make a full screen quad buffer
  BufferCreateInfo ci{}; // One Triangle,
  ci.size = 3 * 2 * sizeof(float);
  ci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  ci.hostVisible = false;
  m_implDetails->m_fullScreenQuadVertexBuffer = m_implDetails->m_resourceManager->createSimpleBufferUnmanaged(ci);

  auto singleBufferPtr = checked_cast<SingleBuffer>(m_implDetails->m_fullScreenQuadVertexBuffer.get());
  StagedSingleBuffer stagedQuadBuffer(engineContext, singleBufferPtr);

  auto transferQueue = m_implDetails->m_commandExecutor->getQueue(QueueRequirement::Transfer);
  transferQueue->runSyncCommand([&](const Rhi::RhiCommandList *cmd) {
    float data[] = {
        0.0f, 0.0f, //
        4.0f, 0.0f, //
        0.0f, 4.0f, //
    };
    stagedQuadBuffer.cmdCopyToDevice(cmd, data, sizeof(data), 0);
  });
  m_implDetails->m_fullScreenQuadVertexBufferDescriptor = std::make_shared<VertexBufferDescriptor>();
  m_implDetails->m_fullScreenQuadVertexBufferDescriptor->addBinding(
      {0}, {Rhi::RhiImageFormat::RHI_FORMAT_R32G32_SFLOAT}, {0}, 2 * sizeof(float));
}

IFRIT_APIDECL void RhiVulkanBackend::waitDeviceIdle() {
  auto p = checked_cast<EngineContext>(m_device.get());
  p->waitIdle();
}

IFRIT_APIDECL std::shared_ptr<Rhi::RhiDeviceTimer> RhiVulkanBackend::createDeviceTimer() {
  auto swapchain = checked_cast<Swapchain>(m_swapChain.get());
  auto numFrameInFlight = swapchain->getNumBackbuffers();
  auto p = std::make_shared<DeviceTimer>(checked_cast<EngineContext>(m_device.get()), numFrameInFlight);
  m_implDetails->m_deviceTimers.push_back(p);
  return p;
}

IFRIT_APIDECL Rhi::RhiBufferRef RhiVulkanBackend::createBuffer(const String &name, u32 size, u32 usage,
                                                               bool hostVisible) const {
  BufferCreateInfo ci{};
  ci.size = size;
  ci.usage = usage;
  ci.hostVisible = hostVisible;
  auto p = m_implDetails->m_resourceManager->createSimpleBufferUnmanaged(ci);
  p->setDebugName(name);
  return p;
}

IFRIT_APIDECL Rhi::RhiBufferRef RhiVulkanBackend::getFullScreenQuadVertexBuffer() const {
  return m_implDetails->m_fullScreenQuadVertexBuffer;
}

IFRIT_APIDECL Rhi::RhiBufferRef RhiVulkanBackend::createBufferDevice(const String &name, u32 size, u32 usage) const {
  BufferCreateInfo ci{};
  ci.size = size;
  ci.usage = usage;
  ci.hostVisible = false;
  auto p = m_implDetails->m_resourceManager->createSimpleBufferUnmanaged(ci);
  p->setDebugName(name);
  return p;
}
IFRIT_APIDECL std::shared_ptr<Rhi::RhiMultiBuffer> RhiVulkanBackend::createBufferCoherent(u32 size, u32 usage,
                                                                                          u32 numCopies) const {
  BufferCreateInfo ci{};
  ci.size = size;
  ci.usage = usage;
  ci.hostVisible = true;
  if (numCopies == ~0u) {
    // Use num backbuffers
    auto swapchain = checked_cast<Swapchain>(m_swapChain.get());
    numCopies = swapchain->getNumBackbuffers();
  }
  return m_implDetails->m_resourceManager->createTracedMultipleBuffer(ci, numCopies);
}

IFRIT_APIDECL std::shared_ptr<Rhi::RhiStagedSingleBuffer>
RhiVulkanBackend::createStagedSingleBuffer(Rhi::RhiBuffer *target) {
  // TODO: release memory, (not managed)
  auto buffer = checked_cast<SingleBuffer>(target);
  auto engineContext = checked_cast<EngineContext>(m_device.get());
  auto ptr = std::make_shared<StagedSingleBuffer>(engineContext, buffer);
  return ptr;
}

IFRIT_APIDECL Rhi::RhiQueue *RhiVulkanBackend::getQueue(Rhi::RhiQueueCapability req) {
  QueueRequirement reqs;
  if (req == Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT) {
    reqs = QueueRequirement::Graphics;
  } else if (req == Rhi::RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT) {
    reqs = QueueRequirement::Compute;
  } else if (req == Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT) {
    reqs = QueueRequirement::Transfer;
  } else if (req ==
             (Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT | Rhi::RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT)) {
    reqs = QueueRequirement::Graphics_Compute;
  } else if (req ==
             (Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT | Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT)) {
    reqs = QueueRequirement::Graphics_Transfer;
  } else if (req ==
             (Rhi::RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT | Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT)) {
    reqs = QueueRequirement::Compute_Transfer;
  } else if (req == (Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT | Rhi::RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT |
                     Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT)) {
    reqs = QueueRequirement::Universal;
  }
  auto s = m_implDetails->m_commandExecutor->getQueue(reqs);
  if (s == nullptr) {
    throw std::runtime_error("Queue not found");
  }
  return s;
}

IFRIT_APIDECL Rhi::RhiShader *RhiVulkanBackend::createShader(const std::string &name, const std::vector<char> &code,
                                                             const std::string &entry, Rhi::RhiShaderStage stage,
                                                             Rhi::RhiShaderSourceType sourceType) {
  ShaderModuleCI ci{};
  ci.code = code;
  ci.entryPoint = entry;
  ci.stage = stage;
  ci.sourceType = sourceType;
  ci.fileName = name;
  auto shaderModule = std::make_unique<ShaderModule>(checked_cast<EngineContext>(m_device.get()), ci);
  auto ptr = shaderModule.get();
  m_implDetails->m_shaderModule.push_back(std::move(shaderModule));
  return ptr;
}

IFRIT_APIDECL Rhi::RhiTextureRef RhiVulkanBackend::createTexture2D(const String &name, u32 width, u32 height,
                                                                   Rhi::RhiImageFormat format, u32 extraFlags) {
  auto p =
      m_implDetails->m_resourceManager->createTexture2DDeviceUnmanaged(width, height, toVkFormat(format), extraFlags);
  p->setDebugName(name);
  return p;
}

IFRIT_APIDECL Rhi::RhiTextureRef RhiVulkanBackend::createDepthTexture(const String &name, u32 width, u32 height) {
  auto p = m_implDetails->m_resourceManager->createDepthAttachment(width, height);
  p->setDebugName(name);
  return p;
}

IFRIT_APIDECL Rhi::RhiTextureRef RhiVulkanBackend::createTexture3D(const String &name, u32 width, u32 height, u32 depth,
                                                                   Rhi::RhiImageFormat format, u32 extraFlags) {
  auto p = m_implDetails->m_resourceManager->createTexture3D(width, height, depth, toVkFormat(format), extraFlags);
  p->setDebugName(name);
  return p;
};

IFRIT_APIDECL Rhi::RhiTextureRef RhiVulkanBackend::createMipMapTexture(const String &name, u32 width, u32 height,
                                                                       u32 mips, Rhi::RhiImageFormat format,
                                                                       u32 extraFlags) {
  auto p = m_implDetails->m_resourceManager->createMipTexture(width, height, mips, toVkFormat(format), extraFlags);
  p->setDebugName(name);
  return p;
}

IFRIT_APIDECL Rhi::RhiSamplerRef RhiVulkanBackend::createTrivialSampler() {
  return m_implDetails->m_resourceManager->createTrivialRenderTargetSampler();
}

IFRIT_APIDECL Rhi::RhiSamplerRef RhiVulkanBackend::createTrivialBilinearSampler(bool repeat) {
  return m_implDetails->m_resourceManager->createTrivialBilinearSampler(repeat);
}

IFRIT_APIDECL Rhi::RhiSamplerRef RhiVulkanBackend::createTrivialNearestSampler(bool repeat) {
  return m_implDetails->m_resourceManager->createTrivialNearestSampler(repeat);
}

IFRIT_APIDECL Rhi::RhiComputePass *RhiVulkanBackend::createComputePass() {
  auto pass =
      std::make_unique<ComputePass>(checked_cast<EngineContext>(m_device.get()), m_implDetails->m_pipelineCache.get(),
                                    m_implDetails->m_descriptorManager.get(), m_implDetails->m_mapper.get());
  auto ptr = pass.get();
  ptr->setDefaultNumMultiBuffers(m_swapChain->getNumBackbuffers());
  m_implDetails->m_computePasses.push_back(std::move(pass));
  return ptr;
}

IFRIT_APIDECL Rhi::RhiGraphicsPass *RhiVulkanBackend::createGraphicsPass() {
  auto pass =
      std::make_unique<GraphicsPass>(checked_cast<EngineContext>(m_device.get()), m_implDetails->m_pipelineCache.get(),
                                     m_implDetails->m_descriptorManager.get(), m_implDetails->m_mapper.get());
  auto ptr = pass.get();
  ptr->setDefaultNumMultiBuffers(m_swapChain->getNumBackbuffers());
  m_implDetails->m_graphicsPasses.push_back(std::move(pass));
  return ptr;
}

IFRIT_APIDECL Rhi::RhiTexture *RhiVulkanBackend::getSwapchainImage() {
  return m_implDetails->m_commandExecutor->getSwapchainImageResource();
}

IFRIT_APIDECL void RhiVulkanBackend::beginFrame() {
  m_implDetails->m_commandExecutor->beginFrame();
  m_implDetails->m_resourceManager->setActiveFrame(m_swapChain->getCurrentImageIndex());
  for (auto &desc : m_implDetails->m_bindlessIndices) {
    desc->setActiveFrame(m_swapChain->getCurrentImageIndex());
  }
  for (auto &pass : m_implDetails->m_computePasses) {
    pass->setActiveFrame(m_swapChain->getCurrentImageIndex());
  }
  for (auto &pass : m_implDetails->m_graphicsPasses) {
    pass->setActiveFrame(m_swapChain->getCurrentImageIndex());
  }
  for (auto &idRef : m_implDetails->m_bindlessIdRefs) {
    idRef->activeFrame = m_swapChain->getCurrentImageIndex();
  }
  for (auto &timer : m_implDetails->m_deviceTimers) {
    timer->frameProceed();
  }

  // get engine context
  auto engineContext = checked_cast<EngineContext>(m_device.get());
  auto deleteList = engineContext->getDeleteQueue();
  auto nums = deleteList->processDeleteQueue();
  if (nums > 0) {
    iDebug("Deleted {} resources", nums);
  }
}
IFRIT_APIDECL void RhiVulkanBackend::endFrame() { m_implDetails->m_commandExecutor->endFrame(); }
IFRIT_APIDECL std::unique_ptr<Rhi::RhiTaskSubmission> RhiVulkanBackend::getSwapchainFrameReadyEventHandler() {
  auto swapchain = checked_cast<Swapchain>(m_swapChain.get());
  auto sema = swapchain->getImageAvailableSemaphoreCurrentFrame();
  TimelineSemaphoreWait wait;
  wait.m_isSwapchainSemaphore = true;
  wait.m_semaphore = sema;
  return std::make_unique<TimelineSemaphoreWait>(wait);
}
IFRIT_APIDECL std::unique_ptr<Rhi::RhiTaskSubmission> RhiVulkanBackend::getSwapchainRenderDoneEventHandler() {
  auto swapchain = checked_cast<Swapchain>(m_swapChain.get());
  auto sema = swapchain->getRenderingFinishSemaphoreCurrentFrame();
  auto fence = swapchain->getCurrentFrameFence();
  TimelineSemaphoreWait wait;
  wait.m_isSwapchainSemaphore = true;
  wait.m_semaphore = sema;
  wait.m_fence = fence;
  return std::make_unique<TimelineSemaphoreWait>(wait);
}

std::shared_ptr<Rhi::RhiColorAttachment> RhiVulkanBackend::createRenderTarget(Rhi::RhiTexture *renderTarget,
                                                                              Rhi::RhiClearValue clearValue,
                                                                              Rhi::RhiRenderTargetLoadOp loadOp,
                                                                              u32 mips, u32 layers) {
  auto attachment = std::make_shared<ColorAttachment>(renderTarget, clearValue, loadOp, mips, layers);
  return attachment;
}

std::shared_ptr<Rhi::RhiDepthStencilAttachment>
RhiVulkanBackend::createRenderTargetDepthStencil(Rhi::RhiTexture *renderTarget, Rhi::RhiClearValue clearValue,
                                                 Rhi::RhiRenderTargetLoadOp loadOp) {
  auto attachment = std::make_shared<DepthStencilAttachment>(renderTarget, clearValue, loadOp);
  return attachment;
}

std::shared_ptr<Rhi::RhiRenderTargets> RhiVulkanBackend::createRenderTargets() {
  auto ctx = checked_cast<EngineContext>(m_device.get());
  return std::make_shared<RenderTargets>(ctx);
}

IFRIT_APIDECL RhiVulkanBackend::~RhiVulkanBackend() { delete m_implDetails; }

IFRIT_APIDECL Rhi::RhiBindlessDescriptorRef *RhiVulkanBackend::createBindlessDescriptorRef() {
  auto ref = std::make_unique<DescriptorBindlessIndices>(checked_cast<EngineContext>(m_device.get()),
                                                         m_implDetails->m_descriptorManager.get(),
                                                         m_swapChain->getNumBackbuffers());
  auto ptr = ref.get();
  m_implDetails->m_bindlessIndices.push_back(std::move(ref));
  return ptr;
}

IFRIT_APIDECL std::shared_ptr<Rhi::RhiBindlessIdRef>
RhiVulkanBackend::registerUniformBuffer(Rhi::RhiMultiBuffer *buffer) {
  std::vector<u32> ids;
  auto descriptorManager = m_implDetails->m_descriptorManager.get();
  auto multiBuffer = checked_cast<MultiBuffer>(buffer);
  auto numBackbuffers = m_swapChain->getNumBackbuffers();
  for (u32 i = 0; i < numBackbuffers; i++) {
    auto id = descriptorManager->registerUniformBuffer(multiBuffer->getBuffer(i));
    ids.push_back(id);
  }
  auto p = std::make_shared<Rhi::RhiBindlessIdRef>();
  p->ids = ids;
  p->activeFrame = m_swapChain->getCurrentImageIndex();
  m_implDetails->m_bindlessIdRefs.push_back(p);
  return p;
}

std::shared_ptr<Rhi::RhiBindlessIdRef> RhiVulkanBackend::registerCombinedImageSampler(Rhi::RhiTexture *texture,
                                                                                      Rhi::RhiSampler *sampler) {
  auto descriptorManager = m_implDetails->m_descriptorManager.get();
  auto tex = checked_cast<SingleDeviceImage>(texture);
  auto sam = checked_cast<Sampler>(sampler);
  auto id = descriptorManager->registerCombinedImageSampler(tex, sam);
  auto p = std::make_shared<Rhi::RhiBindlessIdRef>();
  p->ids.push_back(id);
  p->activeFrame = 0;
  return p;
}

IFRIT_APIDECL std::shared_ptr<Rhi::RhiBindlessIdRef>
RhiVulkanBackend::registerUAVImage(Rhi::RhiTexture *texture, Rhi::RhiImageSubResource subResource) {
  auto descriptorManager = m_implDetails->m_descriptorManager.get();
  auto tex = checked_cast<SingleDeviceImage>(texture);
  auto id = descriptorManager->registerStorageImage(tex, subResource);
  auto p = std::make_shared<Rhi::RhiBindlessIdRef>();
  p->ids.push_back(id);
  p->activeFrame = 0;
  return p;
}

IFRIT_APIDECL std::shared_ptr<Rhi::RhiBindlessIdRef> RhiVulkanBackend::registerStorageBuffer(Rhi::RhiBuffer *buffer) {
  auto descriptorManager = m_implDetails->m_descriptorManager.get();
  auto buf = checked_cast<SingleBuffer>(buffer);
  auto id = descriptorManager->registerStorageBuffer(buf);
  auto p = std::make_shared<Rhi::RhiBindlessIdRef>();
  p->ids.push_back(id);
  p->activeFrame = 0;
  return p;
}

IFRIT_APIDECL std::shared_ptr<Rhi::RhiBindlessIdRef>
RhiVulkanBackend::registerStorageBufferShared(Rhi::RhiMultiBuffer *buffer) {
  // TODO
  std::vector<u32> ids;
  auto descriptorManager = m_implDetails->m_descriptorManager.get();
  auto multiBuffer = checked_cast<MultiBuffer>(buffer);
  auto numBackbuffers = m_swapChain->getNumBackbuffers();
  for (u32 i = 0; i < numBackbuffers; i++) {
    auto id = descriptorManager->registerStorageBuffer(multiBuffer->getBuffer(i));
    ids.push_back(id);
  }
  auto p = std::make_shared<Rhi::RhiBindlessIdRef>();
  p->ids = ids;
  p->activeFrame = m_swapChain->getCurrentImageIndex();
  m_implDetails->m_bindlessIdRefs.push_back(p);
  return p;
}

IFRIT_APIDECL std::shared_ptr<Rhi::RhiVertexBufferView> RhiVulkanBackend::createVertexBufferView() {
  auto view = std::make_shared<VertexBufferDescriptor>();
  return view;
}

IFRIT_APIDECL std::shared_ptr<Rhi::RhiVertexBufferView> RhiVulkanBackend::getFullScreenQuadVertexBufferView() const {
  return m_implDetails->m_fullScreenQuadVertexBufferDescriptor;
}

IFRIT_APIDECL std::unique_ptr<Rhi::FSR2::RhiFsr2Processor> RhiVulkanBackend::createFsr2Processor() {
  auto ctx = checked_cast<EngineContext>(m_device.get());
  return std::make_unique<VulkanGraphics::FSR2::FSR2Processor>(ctx);
}

IFRIT_APIDECL void RhiVulkanBackend::setCacheDirectory(const std::string &dir) {
  auto engineContext = checked_cast<EngineContext>(m_device.get());
  engineContext->setCacheDirectory(dir);
}
IFRIT_APIDECL std::string RhiVulkanBackend::getCacheDirectory() const {
  auto engineContext = checked_cast<EngineContext>(m_device.get());
  return engineContext->getCacheDirectory();
}

IFRIT_APIDECL std::unique_ptr<Rhi::RhiBackend>
RhiVulkanBackendBuilder::createBackend(const Rhi::RhiInitializeArguments &args) {
  return std::make_unique<RhiVulkanBackend>(args);
}

IFRIT_APIDECL void getRhiBackendBuilder_Vulkan(std::unique_ptr<Rhi::RhiBackendFactory> &ptr) {
  ptr = std::make_unique<RhiVulkanBackendBuilder>();
}
} // namespace Ifrit::GraphicsBackend::VulkanGraphics