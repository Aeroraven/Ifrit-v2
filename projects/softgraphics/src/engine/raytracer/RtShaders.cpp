
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


#include "ifrit/softgraphics/engine/raytracer/RtShaders.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer {
IFRIT_HOST void RaytracerShaderExecutionStack::pushStack(const RayInternal &ray,
                                                         const RayHit &rayHit,
                                                         void *pPayload) {
  RaytracerShaderStackElement el;
  el.ray = ray;
  el.payloadPtr = pPayload;
  el.rayHit = rayHit;
  execStack.push_back(el);
  this->onStackPushComplete();
}
IFRIT_HOST void RaytracerShaderExecutionStack::popStack() {
  execStack.pop_back();
  this->onStackPopComplete();
}
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer