
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

namespace Ifrit::Core
{

    Graphics::Rhi::RhiTextureRef ParseTex(std ::filesystem::path path, IApplication* app)
    {
        // read image use stb
        i32  width, height, channels;
        auto data = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
        if (!data)
        {
            iError("Failed to load image: {}", path.string());
            return nullptr;
        }
        auto rhi = app->GetRhi();
        auto tex =
            rhi->CreateTexture2D("Asset_Img", width, height, Graphics::Rhi::RhiImageFormat::RhiImgFmt_R8G8B8A8_UINT,
                Graphics::Rhi::RHI_IMAGE_USAGE_TRANSFER_DST_BIT, false);
        auto tq        = rhi->GetQueue(Graphics::Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
        auto totalSize = width * height * 4;
        auto buffer    = rhi->CreateBuffer("Asset_Buf", totalSize, Graphics::Rhi::RhiBufferUsage::RhiBufferUsage_CopySrc,
               true, false);
        buffer->MapMemory();
        buffer->WriteBuffer(data, totalSize, 0);
        buffer->FlushBuffer();
        buffer->UnmapMemory();

        using namespace Graphics::Rhi;
        auto imageBarrier = [&](const RhiCommandList* cmd, RhiTexture* tex, RhiResourceState src, RhiResourceState dst,
                                RhiImageSubResource sub) {
            RhiTransitionBarrier barrier;
            barrier.m_texture     = tex;
            barrier.m_dstState    = dst;
            barrier.m_subResource = sub;
            barrier.m_type        = RhiResourceType::Texture;

            RhiResourceBarrier resBarrier;
            resBarrier.m_type       = RhiBarrierType::Transition;
            resBarrier.m_transition = barrier;

            cmd->AddResourceBarrier({ resBarrier });
        };
        // iInfo("Loading image: {}", path.string());
        tq->RunSyncCommand([&](const RhiCommandList* cmd) {
            imageBarrier(cmd, tex.get(), RhiResourceState::Undefined, RhiResourceState::CopyDst, { 0, 0, 1, 1 });
            cmd->CopyBufferToImage(buffer.get(), tex.get(), { 0, 0, 1, 1 });
            imageBarrier(cmd, tex.get(), RhiResourceState::CopyDst, RhiResourceState::Common, { 0, 0, 1, 1 });
        });
        // iInfo("Image loaded: {}", path.string());
        stbi_image_free(data);
        return tex;
    }

    IFRIT_APIDECL TrivialImageAsset::TrivialImageAsset(AssetMetadata metadata, std::filesystem::path path,
        IApplication* app)
        : TextureAsset(metadata, path), m_app(app)
    {
        // Pass
    }

    IFRIT_APIDECL Graphics::Rhi::RhiTextureRef TrivialImageAsset::GetTexture()
    {
        if (m_texture == nullptr)
        {
            m_texture = ParseTex(m_path, m_app);
        }
        return m_texture;
    }

    // Importer
    IFRIT_APIDECL void TrivialImageAssetImporter::ProcessMetadata(AssetMetadata& metadata)
    {
        metadata.m_importer = IMPORTER_NAME;
    }

    IFRIT_APIDECL Vec<std::string> TrivialImageAssetImporter::GetSupportedExtensionNames()
    {
        return { ".png" };
    }

    IFRIT_APIDECL void TrivialImageAssetImporter::ImportAsset(const std::filesystem::path& path, AssetMetadata& metadata)
    {
        auto asset = std::make_shared<TrivialImageAsset>(metadata, path, m_assetManager->GetApplication());
        m_assetManager->RegisterAsset(asset);
    }

} // namespace Ifrit::Core