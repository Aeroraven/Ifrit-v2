
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

#include "ifrit/common/logging/Logging.h"

#include "ifrit/core/assetmanager/ShaderAsset.h"
#include <fstream>
namespace Ifrit::Core {

// Shader class
IFRIT_APIDECL ShaderAsset::ShaderRef *ShaderAsset::loadShader() {
  if (m_loaded) {
    return m_selfData;
  } else {
    m_loaded = true;
    std::ifstream file(m_path, std::ios::binary);
    Vec<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    auto rhi = m_app->getRhiLayer();
    GraphicsBackend::Rhi::RhiShaderStage stage;
    auto fileName = m_path.filename().string();
    // endswith .vert.glsl
    auto endsWith = [](const std::string &str, const std::string &suffix) {
      return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (endsWith(fileName, ".vert.glsl")) {
      stage = GraphicsBackend::Rhi::RhiShaderStage::Vertex;
    } else if (endsWith(fileName, ".frag.glsl")) {
      stage = GraphicsBackend::Rhi::RhiShaderStage::Fragment;
    } else if (endsWith(fileName, ".comp.glsl")) {
      stage = GraphicsBackend::Rhi::RhiShaderStage::Compute;
    } else if (endsWith(fileName, ".mesh.glsl")) {
      stage = GraphicsBackend::Rhi::RhiShaderStage::Mesh;
    } else if (endsWith(fileName, ".task.glsl")) {
      stage = GraphicsBackend::Rhi::RhiShaderStage::Task;
    } else {
      throw std::runtime_error("Unknown shader stage");
    }

    auto p = rhi->createShader(fileName, data, "main", stage, GraphicsBackend::Rhi::RhiShaderSourceType::GLSLCode);
    // TODO: eliminate raw pointer
    m_selfData = p;
    return m_selfData;
  }
}

// Importer
IFRIT_APIDECL void ShaderAssetImporter::processMetadata(AssetMetadata &metadata) {
  metadata.m_importer = IMPORTER_NAME;
}

IFRIT_APIDECL Vec<std::string> ShaderAssetImporter::getSupportedExtensionNames() { return {".glsl"}; }

IFRIT_APIDECL void ShaderAssetImporter::importAsset(const std::filesystem::path &path, AssetMetadata &metadata) {
  auto asset = std::make_shared<ShaderAsset>(metadata, path, m_assetManager->getApplication());
  m_assetManager->registerAsset(asset);
  // iInfo("Imported asset: [Shader] {}", metadata.m_uuid);
}

} // namespace Ifrit::Core