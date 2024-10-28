#pragma once
#include "ifrit/common/core/ApiConv.h"
#include "ifrit/common/core/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <memory>

namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics {

struct RhiVulkanBackendImplDetails;
class RhiVulkanBackend : public Rhi::RhiBackend {
protected:
  std::unique_ptr<Rhi::RhiSwapchain> m_swapChain;
  std::unique_ptr<Rhi::RhiDevice> m_device;
  RhiVulkanBackendImplDetails *m_implDetails;

public:
  RhiVulkanBackend(const Rhi::RhiInitializeArguments &args);
  ~RhiVulkanBackend();

  Rhi::RhiBuffer *createBuffer(uint32_t size, uint32_t usage,
                               bool hostVisible) const override;
  Rhi::RhiBuffer *
  createIndirectMeshDrawBufferDevice(uint32_t drawCalls) override;
  Rhi::RhiBuffer *createStorageBufferDevice(uint32_t size, uint32_t usage) = 0;
  Rhi::RhiMultiBuffer *createMultiBuffer(uint32_t size, uint32_t usage,
                                         uint32_t numCopies) = 0;
  Rhi::RhiMultiBuffer *createUniformBufferShared(uint32_t size,
                                                 bool hostVisible,
                                                 uint32_t extraFlags) = 0;
  Rhi::RhiMultiBuffer *createStorageBufferShared(uint32_t size,
                                                 bool hostVisible,
                                                 uint32_t extraFlags) = 0;
  Rhi::RhiStagedSingleBuffer *
  createStagedSingleBuffer(Rhi::RhiBuffer *target) = 0;
};

class RhiVulkanBackendBuilder : public Rhi::RhiBackendFactory,
                                public Common::Core::NonCopyable {
public:
  std::unique_ptr<Rhi::RhiBackend>
  createBackend(const Rhi::RhiInitializeArguments &args) override;
};
} // namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics