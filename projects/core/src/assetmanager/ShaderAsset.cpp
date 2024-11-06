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
    std::vector<char> data((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

    auto rhi = m_app->getRhiLayer();
    GraphicsBackend::Rhi::RhiShaderStage stage;
    auto fileName = m_path.filename().string();
    // endswith .vert.glsl
    auto endsWith = [](const std::string &str, const std::string &suffix) {
      return str.size() >= suffix.size() &&
             str.compare(str.size() - suffix.size(), suffix.size(), suffix) ==
                 0;
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

    auto p =
        rhi->createShader(data, "main", stage,
                          GraphicsBackend::Rhi::RhiShaderSourceType::GLSLCode);
    // TODO: eliminate raw pointer
    m_selfData = p;
    return m_selfData;
  }
}

// Importer
IFRIT_APIDECL void
ShaderAssetImporter::processMetadata(AssetMetadata &metadata) {
  metadata.m_importer = IMPORTER_NAME;
}

IFRIT_APIDECL std::vector<std::string>
ShaderAssetImporter::getSupportedExtensionNames() {
  return {".glsl"};
}

IFRIT_APIDECL void
ShaderAssetImporter::importAsset(const std::filesystem::path &path,
                                 AssetMetadata &metadata) {
  auto asset = std::make_shared<ShaderAsset>(metadata, path,
                                             m_assetManager->getApplication());
  m_assetManager->registerAsset(asset);
  printf("Imported asset: [Shader] %s\n", metadata.m_uuid.c_str());
}

} // namespace Ifrit::Core