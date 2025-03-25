
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

namespace Ifrit::GraphicsBackend::SoftGraphics::Utility::Loader {
class WavefrontLoader {
public:
  void loadObject(const char *path, std::vector<Vector3f> &vertices, std::vector<Vector3f> &normals,
                  std::vector<Vector2f> &uvs, std::vector<u32> &indices);
  std::vector<Vector3f> remapNormals(std::vector<Vector3f> normals, std::vector<u32> indices, int numVertices);
  std::vector<Vector2f> remapUVs(std::vector<Vector2f> uvs, std::vector<u32> indices, int numVertices);
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Utility::Loader