
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