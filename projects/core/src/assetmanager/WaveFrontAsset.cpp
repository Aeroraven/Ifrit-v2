#include "ifrit/core/assetmanager/WaveFrontAsset.h"

namespace Ifrit::Core {
void WaveFrontAssetImporter::processMetadata(AssetMetadata &metadata) {
  metadata.m_importer = IMPORTER_NAME;
}

std::vector<std::string> WaveFrontAssetImporter::getSupportedExtensionNames() {
  return {".obj"};
}

} // namespace Ifrit::Core