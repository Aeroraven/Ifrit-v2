#pragma once
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"

namespace Ifrit::GraphicsBackend::VulkanGraphics {

// Referenced from: https://pavelsmejkal.net/Posts/GPUTimingBasics

class IFRIT_APIDECL DeviceTimer
    : public Ifrit::GraphicsBackend::Rhi::RhiDeviceTimer {
private:
  uint32_t m_numFrameInFlight;
  EngineContext *m_context;
  std::vector<VkQueryPool> m_queryPools;
  std::vector<uint64_t> m_timestampsStart;
  std::vector<uint64_t> m_timestampsEnd;
  uint32_t m_currentFrame = 0;
  float m_elapsedMs = 0;

public:
  DeviceTimer(EngineContext *ctx, uint32_t numFrameInFlight);
  virtual ~DeviceTimer();
  virtual void start(const Rhi::RhiCommandBuffer *cmd) override;
  virtual void stop(const Rhi::RhiCommandBuffer *cmd) override;
  virtual float getElapsedMs() override;
  virtual void frameProceed();
};

} // namespace Ifrit::GraphicsBackend::VulkanGraphics