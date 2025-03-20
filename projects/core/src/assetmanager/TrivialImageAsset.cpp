
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

#include "ifrit/core/assetmanager/TrivialImageAsset.h"
#include "ifrit/common/logging/Logging.h"
#include <fstream>

#include <stb/stb_image.h>

namespace Ifrit::Core {

std::shared_ptr<GraphicsBackend::Rhi::RhiTexture> parseTex(std ::filesystem::path path, IApplication *app) {
  // read image use stb
  i32 width, height, channels;
  auto data = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
  if (!data) {
    iError("Failed to load image: {}", path.string());
    return nullptr;
  }
  auto rhi = app->getRhiLayer();
  auto tex = rhi->createTexture2D(width, height, GraphicsBackend::Rhi::RhiImageFormat::RHI_FORMAT_R8G8B8A8_UINT,
                                  GraphicsBackend::Rhi::RHI_IMAGE_USAGE_TRANSFER_DST_BIT);
  auto tq = rhi->getQueue(GraphicsBackend::Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
  auto totalSize = width * height * 4;
  auto buffer =
      rhi->createBuffer(totalSize, GraphicsBackend::Rhi::RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
  buffer->map();
  buffer->writeBuffer(data, totalSize, 0);
  buffer->flush();
  buffer->unmap();

  using namespace GraphicsBackend::Rhi;
  auto imageBarrier = [&](const RhiCommandBuffer *cmd, RhiTexture *tex, RhiResourceState2 src, RhiResourceState2 dst,
                          RhiImageSubResource sub) {
    RhiTransitionBarrier barrier;
    barrier.m_texture = tex;
    barrier.m_dstState = dst;
    barrier.m_subResource = sub;
    barrier.m_type = RhiResourceType::Texture;

    RhiResourceBarrier resBarrier;
    resBarrier.m_type = RhiBarrierType::Transition;
    resBarrier.m_transition = barrier;

    cmd->resourceBarrier({resBarrier});
  };
  // iInfo("Loading image: {}", path.string());
  tq->runSyncCommand([&](const RhiCommandBuffer *cmd) {
    imageBarrier(cmd, tex.get(), RhiResourceState2::Undefined, RhiResourceState2::CopyDst, {0, 0, 1, 1});
    cmd->copyBufferToImage(buffer.get(), tex.get(), {0, 0, 1, 1});
    imageBarrier(cmd, tex.get(), RhiResourceState2::CopyDst, RhiResourceState2::Common, {0, 0, 1, 1});
  });
  // iInfo("Image loaded: {}", path.string());
  stbi_image_free(data);
  return tex;
}

IFRIT_APIDECL TrivialImageAsset::TrivialImageAsset(AssetMetadata metadata, std::filesystem::path path,
                                                   IApplication *app)
    : TextureAsset(metadata, path), m_app(app) {
  // Pass
}

IFRIT_APIDECL std::shared_ptr<GraphicsBackend::Rhi::RhiTexture> TrivialImageAsset::getTexture() {
  if (!m_texture) {
    m_texture = parseTex(m_path, m_app);
  }
  return m_texture;
}

// Importer
IFRIT_APIDECL void TrivialImageAssetImporter::processMetadata(AssetMetadata &metadata) {
  metadata.m_importer = IMPORTER_NAME;
}

IFRIT_APIDECL std::vector<std::string> TrivialImageAssetImporter::getSupportedExtensionNames() { return {".png"}; }

IFRIT_APIDECL void TrivialImageAssetImporter::importAsset(const std::filesystem::path &path, AssetMetadata &metadata) {
  auto asset = std::make_shared<TrivialImageAsset>(metadata, path, m_assetManager->getApplication());
  m_assetManager->registerAsset(asset);
}

} // namespace Ifrit::Core