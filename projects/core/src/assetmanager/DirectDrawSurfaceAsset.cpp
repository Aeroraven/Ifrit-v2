
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

#include "ifrit/core/assetmanager/DirectDrawSurfaceAsset.h"
#include <fstream>
namespace Ifrit::Core {

void parseDDS(std ::filesystem::path path) {
  std::ifstream ifs;
  ifs.open(path, std::ios::binary);
  if (!ifs.is_open()) {
    iError("Failed to open file: {}", path.generic_string());
    return;
  }
  ifs.seekg(0, std::ios::end);
  auto size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<char> data(size);
  ifs.read(data.data(), size);
  ifs.close();
}

// Importer
IFRIT_APIDECL void
DirectDrawSurfaceAssetImporter::processMetadata(AssetMetadata &metadata) {
  metadata.m_importer = IMPORTER_NAME;
}

IFRIT_APIDECL std::vector<std::string>
DirectDrawSurfaceAssetImporter::getSupportedExtensionNames() {
  return {".dds"};
}

IFRIT_APIDECL void
DirectDrawSurfaceAssetImporter::importAsset(const std::filesystem::path &path,
                                            AssetMetadata &metadata) {
  auto asset = std::make_shared<DirectDrawSurfaceAsset>(
      metadata, path, m_assetManager->getApplication());
  // Test
  parseDDS(path);
  // End Test
  m_assetManager->registerAsset(asset);
  iInfo("Imported asset: [DDSTexture] {}", metadata.m_uuid);
}

} // namespace Ifrit::Core