
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/base/TypeDescriptor.h"
#include "ifrit/softgraphics/engine/base/VaryingStore.h"

// Include order is strictly required here
#include "ifrit/common/math/simd/SimdVectors.h"

namespace Ifrit::GraphicsBackend::SoftGraphics {
struct VertexShaderResultContext {
  std::vector<Vector4f> position;
  std::vector<std::vector<Ifrit::Math::SIMD::SVector4f>> varyings;
  std::vector<TypeDescriptor> varyingDescriptors;
};
class IFRIT_APIDECL VertexShaderResult {
private:
  VertexShaderResultContext *context;
  u32 vertexCount;
  u32 rawCounter = 0;

public:
  VertexShaderResult(u32 vertexCount, u32 varyingCount);
  ~VertexShaderResult();
  Vector4f *getPositionBuffer();
  void initializeVaryingBufferFromShader(const TypeDescriptor &typeDescriptor, int id);
  void setVertexCount(const u32 vertexCount);

  /* Inline */
  TypeDescriptor getVaryingDescriptor(int id) const { return context->varyingDescriptors[id]; }
  void allocateVaryings(int count) {
    context->varyings.resize(count * 2);
    context->varyingDescriptors.resize(count * 2);
  }
  inline void initializeVaryingBuffer(int id) { context->varyings[id].resize(vertexCount); }
  inline Ifrit::Math::SIMD::SVector4f *getVaryingBuffer(int id) { return context->varyings[id].data(); }
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics