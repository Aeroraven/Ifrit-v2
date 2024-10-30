#include "ifrit/core/assetmanager/WaveFrontAsset.h"
#include <fstream>
namespace Ifrit::Core {

void loadWaveFrontObject(const char *path, std::vector<ifloat3> &vertices,
                         std::vector<ifloat3> &normals,
                         std::vector<ifloat2> &uvs,
                         std::vector<uint32_t> &indices) {

  // This section is auto-generated from Copilot
  std::ifstream file(path);
  std::string line;
  std::vector<uint32_t> interIdx;
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
  indices.resize(interIdx.size() / 3);
  for (int i = 0; i < interIdx.size(); i += 3) {
    indices[i / 3] = interIdx[i];
  }
}

// Mesh class

IFRIT_APIDECL std::shared_ptr<MeshData> WaveFrontAsset::loadMesh() {
  if (m_loaded) {
    return m_selfData;
  } else {
    m_loaded = true;
    m_selfData = std::make_shared<MeshData>();
    std::vector<ifloat3> vertices;
    std::vector<ifloat3> normals;
    std::vector<ifloat2> uvs;
    std::vector<uint32_t> indices;
    auto rawPath = m_path.generic_string();
    loadWaveFrontObject(rawPath.c_str(), vertices, normals, uvs, indices);
    m_selfData->m_vertices = vertices;
    m_selfData->m_normals = normals;
    m_selfData->m_uvs = uvs;
    m_selfData->m_indices = indices;
  }
  return m_selfData;
}

// Importer
IFRIT_APIDECL void
WaveFrontAssetImporter::processMetadata(AssetMetadata &metadata) {
  metadata.m_importer = IMPORTER_NAME;
}

IFRIT_APIDECL std::vector<std::string>
WaveFrontAssetImporter::getSupportedExtensionNames() {
  return {".obj"};
}

IFRIT_APIDECL void
WaveFrontAssetImporter::importAsset(const std::filesystem::path &path,
                                    AssetMetadata &metadata) {
  auto asset = std::make_shared<WaveFrontAsset>(metadata, path);
  m_assetManager->registerAsset(asset);
  printf("Imported asset: [WaveFrontMesh] %s\n", metadata.m_uuid.c_str());
}

} // namespace Ifrit::Core