
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
#include "MeshletCommon.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/base/VertexBuffer.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::MeshletBuilder {
class IFRIT_APIDECL TrivialMeshletBuilder {
private:
  const VertexBuffer *vbuffer = nullptr;
  const std::vector<int> *ibuffer = nullptr;

public:
  void bindVertexBuffer(const VertexBuffer &vbuffer);
  void bindIndexBuffer(const std::vector<int> &ibuffer);
  void buildMeshlet(int posAttrId,
                    std::vector<std::unique_ptr<Meshlet>> &outData);
  void mergeMeshlet(const std::vector<std::unique_ptr<Meshlet>> &meshlets,
                    Meshlet &outData, std::vector<int> &outVertexOffset,
                    std::vector<int> &outIndexOffset, bool autoIncre);
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::MeshletBuilder