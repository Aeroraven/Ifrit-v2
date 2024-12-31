
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
#include "ifrit/common/logging/Logging.h"
#include <stdexcept>
namespace Ifrit::GraphicsBackend::VulkanGraphics {
inline void vkrAssert(bool condition, const char *message) {
  if (!condition) {
    iError(message);
    throw std::runtime_error(message);
  }
}
inline void vkrDebug(const char *message) { iDebug(message); }
inline void vkrVulkanAssert(VkResult result, const char *message) {
  if (result != VK_SUCCESS) {
    iError("Error code: {}", static_cast<uint32_t>(result));
    throw std::runtime_error(message);
  }
}
inline void vkrLog(const char *message) { iInfo(message); }
inline void vkrError(const char *message) {
  iError(message);
  throw std::runtime_error(message);
}
} // namespace Ifrit::GraphicsBackend::VulkanGraphics