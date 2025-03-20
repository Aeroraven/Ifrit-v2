
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
#include "ifrit/core/assetmanager/Asset.h"
#include "ifrit/core/assetmanager/TextureAsset.h"
#include "ifrit/core/base/ApplicationInterface.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Core {
class IFRIT_APIDECL TrivialImageAsset : public TextureAsset {
private:
  bool m_loaded = false;
  IApplication *m_app;
  std::shared_ptr<GraphicsBackend::Rhi::RhiTexture> m_texture = nullptr;

public:
  TrivialImageAsset(AssetMetadata metadata, std::filesystem::path path, IApplication *app);
  std::shared_ptr<GraphicsBackend::Rhi::RhiTexture> getTexture() override;
};
class IFRIT_APIDECL TrivialImageAssetImporter : public AssetImporter {
private:
  IApplication *m_app;

public:
  constexpr static const char *IMPORTER_NAME = "TrivialImageAssetImporter";
  TrivialImageAssetImporter(AssetManager *manager) : AssetImporter(manager), m_app(manager->getApplication()) {}
  void processMetadata(AssetMetadata &metadata) override;
  void importAsset(const std::filesystem::path &path, AssetMetadata &metadata) override;
  std::vector<std::string> getSupportedExtensionNames() override;
};
} // namespace Ifrit::Core