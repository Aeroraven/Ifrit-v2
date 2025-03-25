
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
#include "ifrit/display/presentation/window/WindowProvider.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Core {
class IApplication {
public:
  virtual void onStart() = 0;
  virtual void onUpdate() = 0;
  virtual void onEnd() = 0;

  virtual Ifrit::GraphicsBackend::Rhi::RhiBackend *getRhiLayer() = 0;
  virtual Ifrit::Display::Window::WindowProvider *getWindowProvider() = 0;
  virtual String getCacheDirectory() const = 0;
};
} // namespace Ifrit::Core