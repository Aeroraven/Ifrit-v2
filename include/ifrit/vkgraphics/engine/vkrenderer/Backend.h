#pragma once
#include "ifrit/common/core/ApiConv.h"
#include "ifrit/common/core/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <memory>

namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics {

struct RhiVulkanBackendImplDetails;
class IFRIT_APIDECL RhiVulkanBackend : public Rhi::RhiBackend {
protected:
  // Note that destructor order matters here
  // https://isocpp.org/wiki/faq/dtors#order-dtors-for-members
  std::unique_ptr<Rhi::RhiDevice> m_device;
  std::unique_ptr<Rhi::RhiSwapchain> m_swapChain;
  RhiVulkanBackendImplDetails *m_implDetails;
  
public:
  RhiVulkanBackend(const Rhi::RhiInitializeArguments &args);
  ~RhiVulkanBackend();

  void waitDeviceIdle() override;
  Rhi::RhiBuffer *createBuffer(uint32_t size, uint32_t usage,
                               bool hostVisible) const override;
  Rhi::RhiBuffer *
  createIndirectMeshDrawBufferDevice(uint32_t drawCalls) override;
  Rhi::RhiBuffer *createStorageBufferDevice(uint32_t size,
                                            uint32_t usage) override;
  Rhi::RhiMultiBuffer *createMultiBuffer(uint32_t size, uint32_t usage,
                                         uint32_t numCopies) override;
  Rhi::RhiMultiBuffer *createUniformBufferShared(uint32_t size,
                                                 bool hostVisible,
                                                 uint32_t extraFlags) override;
  Rhi::RhiMultiBuffer *createStorageBufferShared(uint32_t size,
                                                 bool hostVisible,
                                                 uint32_t extraFlags) override;
  Rhi::RhiStagedSingleBuffer *
  createStagedSingleBuffer(Rhi::RhiBuffer *target) override;

  // Command execution
  Rhi::RhiQueue *getQueue(Rhi::RhiQueueCapability req) override;

  // Shader
  Rhi::RhiShader *createShader(const std::vector<char> &code, std::string entry,
                               Rhi::RhiShaderStage stage) override;

  // Texture
  Rhi::RhiTexture *createDepthRenderTexture(uint32_t width,
                                            uint32_t height) override;

  // Pass
  Rhi::RhiComputePass *createComputePass() override;

  Rhi::RhiGraphicsPass *createGraphicsPass() override;

  // Swapchain
  Rhi::RhiTexture *getSwapchainImage() override;
  void beginFrame() override;
  void endFrame() override;
  std::unique_ptr<Rhi::RhiTaskSubmission>
  getSwapchainFrameReadyEventHandler() override;
  std::unique_ptr<Rhi::RhiTaskSubmission>
  getSwapchainRenderDoneEventHandler() override;
};

class IFRIT_APIDECL RhiVulkanBackendBuilder : public Rhi::RhiBackendFactory,
                                              public Common::Core::NonCopyable {
public:
  std::unique_ptr<Rhi::RhiBackend>
  createBackend(const Rhi::RhiInitializeArguments &args) override;
};
} // namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics