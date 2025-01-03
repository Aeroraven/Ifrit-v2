
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


#include "ifrit/display/presentation/window/WindowSelector.h"
#include "ifrit/display/presentation/window/GLFWWindowProvider.h"

namespace Ifrit::Display::Window {
IFRIT_APIDECL std::unique_ptr<WindowProvider>
WindowSelector::createWindowProvider(WindowProviderType type,
                                     const WindowProviderSetupArgs &args) {
  GLFWWindowProviderInitArgs initArgs;
  initArgs.vulkanMode = args.useVulkan;
  switch (type) {
  case WindowProviderType::GLFW:
    return std::make_unique<GLFWWindowProvider>(GLFWWindowProvider(initArgs));
  default:
    return nullptr;
  }
}
} // namespace Ifrit::Display::Window