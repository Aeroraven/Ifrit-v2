#pragma once
#include "ifrit/common/util/ApiConv.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <memory>

namespace Ifrit::GraphicsBackend::VulkanGraphics {

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
  Rhi::RhiBuffer *createIndirectMeshDrawBufferDevice(uint32_t drawCalls,
                                                     uint32_t usage) override;
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
  std::shared_ptr<Rhi::RhiStagedSingleBuffer>
  createStagedSingleBuffer(Rhi::RhiBuffer *target) override;

  std::shared_ptr<Rhi::RhiBuffer>
  getFullScreenQuadVertexBuffer() const override;

  // Command execution
  Rhi::RhiQueue *getQueue(Rhi::RhiQueueCapability req) override;

  // Shader
  Rhi::RhiShader *createShader(const std::vector<char> &code, std::string entry,
                               Rhi::RhiShaderStage stage,
                               Rhi::RhiShaderSourceType sourceType) override;

  // Texture
  Rhi::RhiTexture *createDepthRenderTexture(uint32_t width,
                                            uint32_t height) override;

  std::shared_ptr<Rhi::RhiTexture>
  createRenderTargetTexture(uint32_t width, uint32_t height,
                            Rhi::RhiImageFormat format) override;

  std::shared_ptr<Rhi::RhiSampler> createTrivialSampler() override;

  std::shared_ptr<Rhi::RhiTexture>
  createRenderTargetMipTexture(uint32_t width, uint32_t height, uint32_t mips,
                               Rhi::RhiImageFormat format,
                               uint32_t extraFlags) override;

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

  // Descriptor
  virtual Rhi::RhiBindlessDescriptorRef *createBindlessDescriptorRef() override;

  virtual std::shared_ptr<Rhi::RhiBindlessIdRef>
  registerUniformBuffer(Rhi::RhiMultiBuffer *buffer) override;

  virtual std::shared_ptr<Rhi::RhiBindlessIdRef>
  registerStorageBuffer(Rhi::RhiBuffer *buffer) override;

  virtual std::shared_ptr<Rhi::RhiBindlessIdRef>
  registerUAVImage(Rhi::RhiTexture *texture,
                   Rhi::RhiImageSubResource subResource) override;

  // Render targets
  virtual std::shared_ptr<Rhi::RhiColorAttachment>
  createRenderTarget(Rhi::RhiTexture *renderTarget,
                     Rhi::RhiClearValue clearValue,
                     Rhi::RhiRenderTargetLoadOp loadOp, uint32_t mips,
                     uint32_t layers) override;

  virtual std::shared_ptr<Rhi::RhiDepthStencilAttachment>
  createRenderTargetDepthStencil(Rhi::RhiTexture *renderTarget,
                                 Rhi::RhiClearValue clearValue,
                                 Rhi::RhiRenderTargetLoadOp loadOp) override;

  virtual std::shared_ptr<Rhi::RhiRenderTargets> createRenderTargets() override;

  // Vertex buffer
  virtual std::shared_ptr<Rhi::RhiVertexBufferView>
  createVertexBufferView() override;
  virtual std::shared_ptr<Rhi::RhiVertexBufferView>
  getFullScreenQuadVertexBufferView() const override;
};

class IFRIT_APIDECL RhiVulkanBackendBuilder
    : public Rhi::RhiBackendFactory,
      public Common::Utility::NonCopyable {
public:
  std::unique_ptr<Rhi::RhiBackend>
  createBackend(const Rhi::RhiInitializeArguments &args) override;
};
} // namespace Ifrit::GraphicsBackend::VulkanGraphics