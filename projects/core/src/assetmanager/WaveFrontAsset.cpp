
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

#include "ifrit/core/assetmanager/WaveFrontAsset.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/util/TypingUtil.h"
#include <fstream>

using Ifrit::Common::Utility::size_cast;

namespace Ifrit::Core {

void loadWaveFrontObject(const char *path, Vec<ifloat3> &vertices, Vec<ifloat3> &normals, Vec<ifloat2> &uvs,
                         Vec<u32> &indices) {

  // This section is auto-generated from Copilot
  std::ifstream file(path);
  std::string line;
  Vec<u32> interIdx;
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
            interIdx.push_back(std::stoi(index) - 1);
          } else {
            interIdx.push_back(0);
          }
        }
      }
    }
  }
  indices.resize(interIdx.size());
  for (int i = 0; i < interIdx.size(); i++) {
    indices[i] = interIdx[i];
  }
}
Vec<ifloat3> remapNormals(Vec<ifloat3> normals, Vec<u32> indices, int numVertices) {
  using namespace Ifrit::Math;
  Vec<ifloat3> retNormals;
  Vec<int> counters;
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

Vec<ifloat2> remapUVs(Vec<ifloat2> uvs, Vec<u32> indices, int numVertices) {
  Vec<ifloat2> retNormals;
  Vec<int> counters;
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
// Mesh class

IFRIT_APIDECL std::shared_ptr<MeshData> WaveFrontAsset::loadMesh() {
  if (m_loaded) {
    return m_selfData;
  } else {
    m_loaded = true;
    m_selfData = std::make_shared<MeshData>();
    Vec<ifloat3> vertices;
    Vec<ifloat3> normals;
    Vec<ifloat3> remappedNormals;
    Vec<ifloat2> uvs;
    Vec<ifloat2> remappedUVs;
    Vec<u32> remappedIndices;
    Vec<u32> indices;
    auto rawPath = m_path.generic_string();
    loadWaveFrontObject(rawPath.c_str(), vertices, normals, uvs, indices);
    remappedNormals = remapNormals(normals, indices, size_cast<int>(vertices.size()));
    if (uvs.size() != 0) {
      remappedUVs = remapUVs(uvs, indices, size_cast<int>(vertices.size()));
    } else {
      remappedUVs.resize(vertices.size());
    }

    m_selfData->m_vertices = vertices;
    m_selfData->m_normals = remappedNormals;
    m_selfData->m_uvs = remappedUVs;

    // remap indices
    remappedIndices.resize(indices.size() / 3);
    for (int i = 0; i < indices.size(); i += 3) {
      remappedIndices[i / 3] = indices[i];
    }
    m_selfData->m_indices = remappedIndices;

    // align vertices
    m_selfData->m_verticesAligned.resize(vertices.size());
    m_selfData->m_normalsAligned.resize(vertices.size());
    for (int i = 0; i < vertices.size(); i++) {
      m_selfData->m_verticesAligned[i] = ifloat4(vertices[i].x, vertices[i].y, vertices[i].z, 1.0);
      m_selfData->m_normalsAligned[i] = ifloat4(remappedNormals[i].x, remappedNormals[i].y, remappedNormals[i].z, 1.0);
    }
    this->createMeshLodHierarchy(m_selfData, "");
  }
  return m_selfData;
}

IFRIT_APIDECL MeshData *WaveFrontAsset::loadMeshUnsafe() {
  if (m_selfDataRaw == nullptr) {
    if (m_selfData == nullptr) {
      m_selfDataRaw = loadMesh().get();
    } else {
      m_selfDataRaw = m_selfData.get();
    }
  }
  return m_selfDataRaw;
}

// Importer
IFRIT_APIDECL void WaveFrontAssetImporter::processMetadata(AssetMetadata &metadata) {
  metadata.m_importer = IMPORTER_NAME;
}

IFRIT_APIDECL Vec<std::string> WaveFrontAssetImporter::getSupportedExtensionNames() { return {".obj"}; }

IFRIT_APIDECL void WaveFrontAssetImporter::importAsset(const std::filesystem::path &path, AssetMetadata &metadata) {
  auto asset = std::make_shared<WaveFrontAsset>(metadata, path);
  m_assetManager->registerAsset(asset);
  // iInfo("Imported asset: [WaveFrontMesh] {}", metadata.m_uuid);
}

} // namespace Ifrit::Core