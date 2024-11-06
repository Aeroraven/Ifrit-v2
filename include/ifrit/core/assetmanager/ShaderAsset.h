#pragma once
#include "ifrit/core/assetmanager/Asset.h"
#include "ifrit/core/base/ApplicationInterface.h"

namespace Ifrit::Core {
class IFRIT_APIDECL ShaderAsset : public Asset {
private:
  using ShaderRef = GraphicsBackend::Rhi::RhiShader;
  ShaderAsset::ShaderRef *m_selfData;
  bool m_loaded = false;
  IApplication *m_app;

public:
  ShaderAsset(AssetMetadata metadata, std::filesystem::path path,
              IApplication *app)
      : Asset(metadata, path), m_app(app) {}

  ShaderRef *loadShader();
};
class IFRIT_APIDECL ShaderAssetImporter : public AssetImporter {
private:
  IApplication *m_app;

public:
  constexpr static const char *IMPORTER_NAME = "ShaderImporter";
  ShaderAssetImporter(AssetManager *manager)
      : AssetImporter(manager), m_app(manager->getApplication()) {}
  void processMetadata(AssetMetadata &metadata) override;
  void importAsset(const std::filesystem::path &path,
                   AssetMetadata &metadata) override;
  std::vector<std::string> getSupportedExtensionNames() override;
};
} // namespace Ifrit::Core