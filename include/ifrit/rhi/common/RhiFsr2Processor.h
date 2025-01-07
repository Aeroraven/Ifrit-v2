
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
#include "RhiForwardingTypes.h"
#include "ifrit/common/util/ApiConv.h"
#include <cstdint>
namespace Ifrit::GraphicsBackend::Rhi::FSR2 {

struct RhiFSR2InitialzeArgs {
  uint32_t maxRenderWidth;
  uint32_t maxRenderHeight;
  uint32_t displayWidth;
  uint32_t displayHeight;
};

struct RhiFSR2DispatchArgs {
  Rhi::RhiTexture *color;
  Rhi::RhiTexture *depth;
  Rhi::RhiTexture *motion;
  Rhi::RhiTexture *exposure;
  Rhi::RhiTexture *reactiveMask;
  Rhi::RhiTexture *transparencyMask;
  Rhi::RhiTexture *output;
  float deltaTime;
  float jitterX;
  float jitterY;
  float camNear;
  float camFar;
  float camFovY;
};

class IFRIT_APIDECL RhiFsr2Processor {
public:
  virtual ~RhiFsr2Processor() = default;
  virtual void init(const Rhi::FSR2::RhiFSR2InitialzeArgs &args) = 0;
  virtual void getJitters(float *jitterX, float *jitterY, uint32_t frameIdx,
                          uint32_t rtWidth, uint32_t rtHeight) = 0;
  virtual void dispatch(const Rhi::RhiCommandBuffer *cmd,
                        const Rhi::FSR2::RhiFSR2DispatchArgs &args) = 0;
};

} // namespace Ifrit::GraphicsBackend::Rhi::FSR2