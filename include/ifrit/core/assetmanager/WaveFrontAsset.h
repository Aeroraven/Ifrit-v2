#pragma once
#include "ifrit/core/assetmanager/Asset.h"

namespace Ifrit::Core {
class WaveFrontAsset : public Asset {};
class WaveFrontAssetImporter : public AssetImporter {
public:
  constexpr static const char *IMPORTER_NAME = "WaveFrontImporter";
  WaveFrontAssetImporter(AssetManager *manager) : AssetImporter(manager) {}
  void processMetadata(AssetMetadata &metadata) override;
  std::vector<std::string> getSupportedExtensionNames() override;
};
} // namespace Ifrit::Core