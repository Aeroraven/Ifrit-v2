#include "ifrit/vkgraphics/engine/vkrenderer/Backend.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/engine/vkrenderer/RenderGraph.h"
#include "ifrit/vkgraphics/engine/vkrenderer/StagedMemoryResource.h"

using namespace Ifrit::Common::Utility;

namespace Ifrit::GraphicsBackend::VulkanGraphics {
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
};

IFRIT_APIDECL
RhiVulkanBackend::RhiVulkanBackend(const Rhi::RhiInitializeArguments &args) {
  m_device = std::make_unique<EngineContext>(args);
  auto engineContext = checked_cast<EngineContext>(m_device.get());
  m_swapChain = std::make_unique<Swapchain>(engineContext);
  auto swapchain = checked_cast<Swapchain>(m_swapChain.get());
  m_implDetails = new RhiVulkanBackendImplDetails();
  m_implDetails->m_descriptorManager =
      std::make_unique<DescriptorManager>(engineContext);
  m_implDetails->m_resourceManager =
      std::make_unique<ResourceManager>(engineContext);
  m_implDetails->m_commandExecutor = std::make_unique<CommandExecutor>(
      engineContext, swapchain, m_implDetails->m_descriptorManager.get(),
      m_implDetails->m_resourceManager.get());
  m_implDetails->m_pipelineCache =
      std::make_unique<PipelineCache>(engineContext);
  m_implDetails->m_mapper = std::make_unique<RegisteredResourceMapper>();
  m_implDetails->m_commandExecutor->setQueues(
      1, args.m_expectedGraphicsQueueCount, args.m_expectedComputeQueueCount,
      args.m_expectedTransferQueueCount);
}

IFRIT_APIDECL void RhiVulkanBackend::waitDeviceIdle() {
  auto p = checked_cast<EngineContext>(m_device.get());
  p->waitIdle();
}

IFRIT_APIDECL Rhi::RhiBuffer *
RhiVulkanBackend::createBuffer(uint32_t size, uint32_t usage,
                               bool hostVisible) const {
  BufferCreateInfo ci{};
  ci.size = size;
  ci.usage = usage;
  ci.hostVisible = hostVisible;
  auto p = m_implDetails->m_resourceManager->createSimpleBuffer(ci);
  return p;
}
IFRIT_APIDECL Rhi::RhiBuffer *
RhiVulkanBackend::createIndirectMeshDrawBufferDevice(uint32_t drawCalls) {
  return m_implDetails->m_resourceManager->createIndirectMeshDrawBufferDevice(
      drawCalls);
}
IFRIT_APIDECL Rhi::RhiBuffer *
RhiVulkanBackend::createStorageBufferDevice(uint32_t size, uint32_t usage) {
  return m_implDetails->m_resourceManager->createStorageBufferDevice(size,
                                                                     usage);
}
IFRIT_APIDECL Rhi::RhiMultiBuffer *
RhiVulkanBackend::createMultiBuffer(uint32_t size, uint32_t usage,
                                    uint32_t numCopies) {
  BufferCreateInfo ci{};
  ci.size = size;
  ci.usage = usage;
  ci.hostVisible = true;
  return m_implDetails->m_resourceManager->createMultipleBuffer(ci, numCopies);
}
IFRIT_APIDECL Rhi::RhiMultiBuffer *
RhiVulkanBackend::createUniformBufferShared(uint32_t size, bool hostVisible,
                                            uint32_t extraFlags) {
  return m_implDetails->m_resourceManager->createUniformBufferShared(
      size, hostVisible, extraFlags);
}
IFRIT_APIDECL Rhi::RhiMultiBuffer *
RhiVulkanBackend::createStorageBufferShared(uint32_t size, bool hostVisible,
                                            uint32_t extraFlags) {
  return m_implDetails->m_resourceManager->createStorageBufferShared(
      size, hostVisible, extraFlags);
}
IFRIT_APIDECL Rhi::RhiStagedSingleBuffer *
RhiVulkanBackend::createStagedSingleBuffer(Rhi::RhiBuffer *target) {
  // TODO: release memory, (not managed)
  auto buffer = checked_cast<SingleBuffer>(target);
  auto engineContext = checked_cast<EngineContext>(m_device.get());
  auto unique = std::make_unique<StagedSingleBuffer>(engineContext, buffer);
  auto ptr = unique.get();
  m_implDetails->m_stagedSingleBuffer.push_back(std::move(unique));
  return ptr;
}

IFRIT_APIDECL Rhi::RhiQueue *
RhiVulkanBackend::getQueue(Rhi::RhiQueueCapability req) {
  QueueRequirement reqs;
  if (req == Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT) {
    reqs = QueueRequirement::Graphics;
  } else if (req == Rhi::RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT) {
    reqs = QueueRequirement::Compute;
  } else if (req == Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT) {
    reqs = QueueRequirement::Transfer;
  } else if (req == Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT |
             Rhi::RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT) {
    reqs = QueueRequirement::Graphics_Compute;
  } else if (req == Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT |
             Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT) {
    reqs = QueueRequirement::Graphics_Transfer;
  } else if (req == Rhi::RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT |
             Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT) {
    reqs = QueueRequirement::Compute_Transfer;
  } else if (req == Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT |
             Rhi::RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT |
             Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT) {
    reqs = QueueRequirement::Universal;
  }
  auto s = m_implDetails->m_commandExecutor->getQueue(reqs);
  if (s == nullptr) {
    throw std::runtime_error("Queue not found");
  }
  return s;
}

IFRIT_APIDECL Rhi::RhiShader *
RhiVulkanBackend::createShader(const std::vector<char> &code, std::string entry,
                               Rhi::RhiShaderStage stage) {
  auto shaderModule = std::make_unique<ShaderModule>(
      checked_cast<EngineContext>(m_device.get()), code, entry, stage);
  auto ptr = shaderModule.get();
  m_implDetails->m_shaderModule.push_back(std::move(shaderModule));
  return ptr;
}

IFRIT_APIDECL Rhi::RhiTexture *
RhiVulkanBackend::createDepthRenderTexture(uint32_t width, uint32_t height) {
  return m_implDetails->m_resourceManager->createDepthAttachment(width, height);
}

IFRIT_APIDECL Rhi::RhiComputePass *RhiVulkanBackend::createComputePass() {
  auto pass = std::make_unique<ComputePass>(
      checked_cast<EngineContext>(m_device.get()),
      m_implDetails->m_pipelineCache.get(),
      m_implDetails->m_descriptorManager.get(), m_implDetails->m_mapper.get());
  auto ptr = pass.get();
  ptr->setDefaultNumMultiBuffers(m_swapChain->getNumBackbuffers());
  m_implDetails->m_computePasses.push_back(std::move(pass));
  return ptr;
}

IFRIT_APIDECL Rhi::RhiGraphicsPass *RhiVulkanBackend::createGraphicsPass() {
  auto pass = std::make_unique<GraphicsPass>(
      checked_cast<EngineContext>(m_device.get()),
      m_implDetails->m_pipelineCache.get(),
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
  m_implDetails->m_resourceManager->setActiveFrame(
      m_swapChain->getCurrentImageIndex());
  for (auto &pass : m_implDetails->m_computePasses) {
    pass->setActiveFrame(m_swapChain->getCurrentImageIndex());
  }
  for (auto &pass : m_implDetails->m_graphicsPasses) {
    pass->setActiveFrame(m_swapChain->getCurrentImageIndex());
  }
}
IFRIT_APIDECL void RhiVulkanBackend::endFrame() {
  m_implDetails->m_commandExecutor->endFrame();
}
IFRIT_APIDECL std::unique_ptr<Rhi::RhiTaskSubmission>
RhiVulkanBackend::getSwapchainFrameReadyEventHandler() {
  auto swapchain = checked_cast<Swapchain>(m_swapChain.get());
  auto sema = swapchain->getImageAvailableSemaphoreCurrentFrame();
  TimelineSemaphoreWait wait;
  wait.m_isSwapchainSemaphore = true;
  wait.m_semaphore = sema;
  return std::make_unique<TimelineSemaphoreWait>(wait);
}
IFRIT_APIDECL std::unique_ptr<Rhi::RhiTaskSubmission>
RhiVulkanBackend::getSwapchainRenderDoneEventHandler() {
  auto swapchain = checked_cast<Swapchain>(m_swapChain.get());
  auto sema = swapchain->getRenderingFinishSemaphoreCurrentFrame();
  auto fence = swapchain->getCurrentFrameFence();
  TimelineSemaphoreWait wait;
  wait.m_isSwapchainSemaphore = true;
  wait.m_semaphore = sema;
  wait.m_fence = fence;
  return std::make_unique<TimelineSemaphoreWait>(wait);
}

IFRIT_APIDECL RhiVulkanBackend::~RhiVulkanBackend() { delete m_implDetails; }

IFRIT_APIDECL std::unique_ptr<Rhi::RhiBackend>
RhiVulkanBackendBuilder::createBackend(
    const Rhi::RhiInitializeArguments &args) {
  return std::make_unique<RhiVulkanBackend>(args);
}

} // namespace Ifrit::GraphicsBackend::VulkanGraphics