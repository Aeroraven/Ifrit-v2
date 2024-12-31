
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


#include "ifrit/softgraphics/utility/loader/WavefrontLoader.h"
#include "ifrit/common/math/VectorOps.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::Utility::Loader {
std::vector<ifloat2> WavefrontLoader::remapUVs(std::vector<ifloat2> uvs,
                                               std::vector<uint32_t> indices,
                                               int numVertices) {
  std::vector<ifloat2> retNormals;
  std::vector<int> counters;
  retNormals.clear();
  counters.clear();
  retNormals.resize(numVertices);
  counters.resize(numVertices);
  for (int i = 0; i < numVertices; i++) {
    retNormals[i] = {0, 0};
    counters[i] = 0;
  }
  for (int i = 0; i < indices.size(); i += 3) {
    auto faceNode = indices[i];
    auto normalNode = indices[i + 1];
    retNormals[faceNode].x = uvs[normalNode].x;
    retNormals[faceNode].y = uvs[normalNode].y;
    counters[faceNode]++;
  }
  return retNormals;
}
std::vector<ifloat3>
WavefrontLoader::remapNormals(std::vector<ifloat3> normals,
                              std::vector<uint32_t> indices, int numVertices) {
  using namespace Ifrit::Math;
  std::vector<ifloat3> retNormals;
  std::vector<int> counters;
  retNormals.clear();
  counters.clear();
  retNormals.resize(numVertices);
  counters.resize(numVertices);
  for (int i = 0; i < numVertices; i++) {
    retNormals[i] = {0, 0, 0};
    counters[i] = 0;
  }
  for (int i = 0; i < indices.size(); i += 3) {
    auto faceNode = indices[i];
    auto normalNode = indices[i + 2];
    retNormals[faceNode].x += normals[normalNode].x;
    retNormals[faceNode].y += normals[normalNode].y;
    retNormals[faceNode].z += normals[normalNode].z;
    counters[faceNode]++;
  }
  for (int i = 0; i < numVertices; i++) {
    retNormals[i] = normalize(retNormals[i]);
  }
  return retNormals;
}
void WavefrontLoader::loadObject(const char *path,
                                 std::vector<ifloat3> &vertices,
                                 std::vector<ifloat3> &normals,
                                 std::vector<ifloat2> &uvs,
                                 std::vector<uint32_t> &indices) {

  // This section is auto-generated from Copilot

  std::ifstream file(path);
  std::string line;

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string type;
    iss >> type;

    if (type == "v") {
      ifloat3 vertex;
      iss >> vertex.x >> vertex.y >> vertex.z;
      vertices.push_back(vertex);
    } else if (type == "vn") {
      ifloat3 normal;
      iss >> normal.x >> normal.y >> normal.z;
      normals.push_back(normal);
    } else if (type == "vt") {
      ifloat2 uv;
      iss >> uv.x >> uv.y;
      uvs.push_back(uv);
    } else if (type == "f") {
      std::string vertex;
      for (int i = 0; i < 3; i++) {
        iss >> vertex;
        std::istringstream vss(vertex);
        std::string index;
        for (int j = 0; j < 3; j++) {
          std::getline(vss, index, '/');
          if (index.size() != 0) {
            indices.push_back(std::stoi(index) - 1);
          } else {
            indices.push_back(0);
          }
        }
      }
    }
  }
}
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Utility::Loader