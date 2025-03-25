
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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

#include "RhiBaseTypes.h"

namespace Ifrit::GraphicsBackend::Rhi {
class IFRIT_APIDECL RhiDevice {
protected:
  virtual int _polymorphismPlaceHolder() { return 0; }
};

class IFRIT_APIDECL RhiSwapchain {
protected:
  RhiDevice *m_context;

public:
  virtual ~RhiSwapchain() = default;
  virtual void present() = 0;
  virtual u32 acquireNextImage() = 0;
  virtual u32 getNumBackbuffers() const = 0;
  virtual u32 getCurrentFrameIndex() const = 0;
  virtual u32 getCurrentImageIndex() const = 0;
};
} // namespace Ifrit::GraphicsBackend::Rhi