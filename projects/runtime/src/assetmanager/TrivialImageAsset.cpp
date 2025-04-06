
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

#include "ifrit/runtime/assetmanager/TrivialImageAsset.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/imaging/compress/CompressedTextureUtil.h"
#include <fstream>

#include <stb/stb_image.h>

namespace Ifrit::Runtime
{

    Graphics::Rhi::RhiTextureRef ParseTex(std ::filesystem::path path, IApplication* app, const String& uuid)
    {
        // read image use stb
        i32          width, height, channels;
        u32          texSize;
        u8*          dataRaw = nullptr;
        RSizedBuffer data;
        bool         endsWithPng = path.extension() == ".png";
        bool         isNormalMap = path.string().ends_with("Normal.png");
        auto         defaultFmt  = Graphics::Rhi::RhiImageFormat::RhiImgFmt_R8G8B8A8_UNORM;
        if (endsWithPng)
        {
            using namespace Ifrit::Imaging::Compress;
            String cacheFile = app->GetCacheDir() + "/asset.bc7." + uuid + ".cache";
            if (isNormalMap)
            {
                cacheFile  = app->GetCacheDir() + "/asset.bc5." + uuid + ".cache";
                defaultFmt = Graphics::Rhi::RhiImageFormat::RhiImgFmt_BC5_UNORM_BLOCK;
            }
            bool fileExists          = std::filesystem::exists(cacheFile);
            bool shouldGenCompressed = false;
            if (!fileExists)
            {
                shouldGenCompressed = true;
            }

            if (shouldGenCompressed)
            {
                iInfo("Compressing image to BCn format: {}", uuid);
                // read png file
                dataRaw = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
                if (!dataRaw)
                {
                    iError("Failed to load image: {}", path.string());
                    return nullptr;
                }

                if (isNormalMap)
                {
                    RSizedBuffer inputData(dataRaw, width * height * 4);
                    RSizedBuffer normalRG;
                    DiscardBAChannel(inputData, normalRG, width, height, 1, 4, sizeof(u8));
                    WriteTex2DToBlockCompressedFile(
                        normalRG, cacheFile, TextureFormat::RG8_UNORM, width, height, 1, CompressionAlgo::BC5);
                    iInfo("Compressed image to BC5 format: {}", cacheFile);
                }
                else
                {
                    RSizedBuffer inputData(dataRaw, width * height * 4);
                    WriteTex2DToBlockCompressedFile(
                        inputData, cacheFile, TextureFormat::RGBA8_UNORM, width, height, 1, CompressionAlgo::BC7);
                    iInfo("Compressed image to BC7 format: {}", cacheFile);
                }
                stbi_image_free(dataRaw);
            }

            defaultFmt = Graphics::Rhi::RhiImageFormat::RhiImgFmt_BC7_UNORM_BLOCK;
            if (isNormalMap)
            {
                defaultFmt = Graphics::Rhi::RhiImageFormat::RhiImgFmt_BC5_UNORM_BLOCK;
            }
            TextureFormat fmt;
            u32           baseWidth, baseHeight, baseDepth;
            ReadBlockCompressedTex2DFromFile(data, cacheFile, baseWidth, baseHeight, baseDepth);
            texSize  = data.GetSize();
            height   = baseHeight;
            width    = baseWidth;
            channels = (isNormalMap) ? 2 : 4;
        }
        else
        {
            dataRaw = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
            if (!dataRaw)
            {
                iError("Failed to load image: {}", path.string());
                return nullptr;
            }
            texSize = width * height * 4;
            data.CopyFromRaw(dataRaw, texSize);
            stbi_image_free(dataRaw);
        }

        auto rhi = app->GetRhi();
        auto tex = rhi->CreateTexture2D(
            "Asset_Img", width, height, defaultFmt, Graphics::Rhi::RHI_IMAGE_USAGE_TRANSFER_DST_BIT, false);
        auto tq = rhi->GetQueue(Graphics::Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
        auto buffer =
            rhi->CreateBuffer("Asset_Buf", texSize, Graphics::Rhi::RhiBufferUsage::RhiBufferUsage_CopySrc, true, false);
        buffer->MapMemory();
        buffer->WriteBuffer(data.GetData(), texSize, 0);
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
        return tex;
    }

    IFRIT_APIDECL TrivialImageAsset::TrivialImageAsset(
        AssetMetadata metadata, std::filesystem::path path, IApplication* app)
        : TextureAsset(metadata, path), m_app(app)
    {
        // Pass
    }

    IFRIT_APIDECL Graphics::Rhi::RhiTextureRef TrivialImageAsset::GetTexture()
    {
        if (m_texture == nullptr)
        {
            auto uuid = m_metadata.m_uuid;
            m_texture = ParseTex(m_path, m_app, uuid);
        }
        return m_texture;
    }

    // Importer
    IFRIT_APIDECL void TrivialImageAssetImporter::ProcessMetadata(AssetMetadata& metadata)
    {
        metadata.m_importer = IMPORTER_NAME;
    }

    IFRIT_APIDECL Vec<std::string> TrivialImageAssetImporter::GetSupportedExtensionNames() { return { ".png" }; }

    IFRIT_APIDECL void             TrivialImageAssetImporter::ImportAsset(
        const std::filesystem::path& path, AssetMetadata& metadata)
    {
        auto asset = std::make_shared<TrivialImageAsset>(metadata, path, m_assetManager->GetApplication());
        m_assetManager->RegisterAsset(asset);
    }

} // namespace Ifrit::Runtime