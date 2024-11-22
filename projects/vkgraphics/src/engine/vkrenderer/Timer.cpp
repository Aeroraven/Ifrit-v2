#include "ifrit/vkgraphics/engine/vkrenderer/Timer.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Command.h"
#include "ifrit/vkgraphics/utility/Logger.h"

namespace Ifrit::GraphicsBackend::VulkanGraphics {
IFRIT_APIDECL DeviceTimer::DeviceTimer(EngineContext *ctx,
                                       uint32_t numFrameInFlight)
    : m_context(ctx), m_numFrameInFlight(numFrameInFlight) {
  m_queryPools.resize(numFrameInFlight);
  m_timestampsStart.resize(numFrameInFlight);
  m_timestampsEnd.resize(numFrameInFlight);
  VkQueryPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
  poolInfo.queryCount = 2;
  for (uint32_t i = 0; i < numFrameInFlight; i++) {
    vkrVulkanAssert(vkCreateQueryPool(ctx->getDevice(), &poolInfo, nullptr,
                                      &m_queryPools[i]),
                    "Failed to create query pool");
  }
}

IFRIT_APIDECL DeviceTimer::~DeviceTimer() {
  for (auto &pool : m_queryPools) {
    vkDestroyQueryPool(m_context->getDevice(), pool, nullptr);
  }
}

IFRIT_APIDECL void DeviceTimer::start(const Rhi::RhiCommandBuffer *cmd) {
  auto cmdBuf = Ifrit::Common::Utility::checked_cast<CommandBuffer>(cmd);
  auto curFrame = m_currentFrame;
  vkCmdWriteTimestamp(cmdBuf->getCommandBuffer(),
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_queryPools[curFrame],
                      0);
}

IFRIT_APIDECL void DeviceTimer::stop(const Rhi::RhiCommandBuffer *cmd) {
  auto cmdBuf = Ifrit::Common::Utility::checked_cast<CommandBuffer>(cmd);
  auto curFrame = m_currentFrame;
  vkCmdWriteTimestamp(cmdBuf->getCommandBuffer(),
                      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      m_queryPools[curFrame], 1);
}

IFRIT_APIDECL float DeviceTimer::getElapsedMs() { return m_elapsedMs; }

IFRIT_APIDECL void DeviceTimer::frameProceed() {
  m_currentFrame = (m_currentFrame + 1) % m_numFrameInFlight;

  auto device = m_context->getDevice();
  auto timeStampPeriod =
      m_context->getPhysicalDeviceProperties().limits.timestampPeriod;
  auto curFrame = m_currentFrame;
  uint64_t ts[2];
  vkGetQueryPoolResults(device, m_queryPools[curFrame], 0, 2,
                        sizeof(uint64_t) * 2, ts, sizeof(uint64_t),
                        VK_QUERY_RESULT_64_BIT);
  float nanoToMs = 1.0f / 1000000.0f;
  m_elapsedMs = static_cast<float>(ts[1] - ts[0]) * timeStampPeriod * nanoToMs;
  vkResetQueryPool(m_context->getDevice(), m_queryPools[m_currentFrame], 0, 2);
}

} // namespace Ifrit::GraphicsBackend::VulkanGraphics