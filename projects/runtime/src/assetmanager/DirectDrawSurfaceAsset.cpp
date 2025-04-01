
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

#include "ifrit/core/logging/Logging.h"

#include "ifrit/runtime/assetmanager/DirectDrawSurfaceAsset.h"
#include <fstream>

#ifndef MAKEFOURCC
    #define MAKEFOURCC(ch0, ch1, ch2, ch3) \
        ((u32)(char)(ch0) | ((u32)(char)(ch1) << 8) | ((u32)(char)(ch2) << 16) | ((u32)(char)(ch3) << 24))
#endif

namespace Ifrit::Runtime
{

    struct DDS_PIXELFORMAT
    {
        u32 dwSize;
        u32 dwFlags;
        u32 dwFourCC;
        u32 dwRGBBitCount;
        u32 dwRBitMask;
        u32 dwGBitMask;
        u32 dwBBitMask;
        u32 dwABitMask;
    };

    typedef struct
    {
        u32 dxgiFormat;
        u32 resourceDimension;
        u32 miscFlag;
        u32 arraySize;
        u32 miscFlags2;
    } DDS_HEADER_DXT10;

    typedef struct
    {
        u32             dwSize;
        u32             dwFlags;
        u32             dwHeight;
        u32             dwWidth;
        u32             dwPitchOrLinearSize;
        u32             dwDepth;
        u32             dwMipMapCount;
        u32             dwReserved1[11];
        DDS_PIXELFORMAT ddspf;
        u32             dwCaps;
        u32             dwCaps2;
        u32             dwCaps3;
        u32             dwCaps4;
        u32             dwReserved2;
    } DDS_HEADER;

    enum class DDSCompressedType
    {
        Unknown,
        DXT1,
        DXT2,
        DXT3,
        DXT4,
        DXT5,
        ATI1,
        ATI2,
    };

    Graphics::Rhi::RhiTextureRef parseDDS(std ::filesystem::path path, IApplication* app)
    {
        std::ifstream ifs;
        ifs.open(path, std::ios::binary);
        if (!ifs.is_open())
        {
            iError("Failed to open file: {}", path.generic_string());
            return nullptr;
        }
        ifs.seekg(0, std::ios::end);
        auto size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        Vec<char> data(size);
        ifs.read(data.data(), size);
        ifs.close();

        // Check magic
        using namespace Ifrit::Graphics::Rhi;
        u32 bodyOffset = 4;
        u32 magic      = *reinterpret_cast<u32*>(data.data());
        if (magic != 0x20534444)
        {
            iError("Invalid magic number: {}", magic);
            std::abort();
        }

        DDS_HEADER*       header         = reinterpret_cast<DDS_HEADER*>(data.data() + 4);
        DDS_HEADER_DXT10* header10       = nullptr;
        DDSCompressedType compressedType = DDSCompressedType::Unknown;
        RhiImageFormat    format         = RhiImageFormat::RhiImgFmt_UNDEFINED;
        bool              hasAlpha       = false;

        bodyOffset += sizeof(DDS_HEADER);
        hasAlpha = header->ddspf.dwFlags & 0x1;

        // Check DX10 header
        if (header->ddspf.dwFourCC == MAKEFOURCC('D', 'X', '1', '0'))
        {
            iInfo("DX10 header detected");
            header10 = reinterpret_cast<DDS_HEADER_DXT10*>(data.data() + 4 + sizeof(DDS_HEADER));
            iInfo("DX10 Header: dxgiFormat: {}, resourceDimension: {}, miscFlag: {}, "
                  "arraySize: {}, miscFlags2: {}",
                header10->dxgiFormat, header10->resourceDimension, header10->miscFlag, header10->arraySize,
                header10->miscFlags2);
            bodyOffset += sizeof(DDS_HEADER_DXT10);
        }
        else if (header->ddspf.dwFourCC == MAKEFOURCC('D', 'X', 'T', '1'))
        {

            compressedType = DDSCompressedType::DXT1;
            if (hasAlpha)
            {
                format = RhiImageFormat::RhiImgFmt_BC1_RGBA_UNORM_BLOCK;
            }
            else
            {
                format = RhiImageFormat::RhiImgFmt_BC1_RGB_UNORM_BLOCK;
            }
        }
        else if (header->ddspf.dwFourCC == MAKEFOURCC('D', 'X', 'T', '2'))
        {
            compressedType = DDSCompressedType::DXT2;
            format         = RhiImageFormat::RhiImgFmt_BC2_UNORM_BLOCK;
        }
        else if (header->ddspf.dwFourCC == MAKEFOURCC('D', 'X', 'T', '3'))
        {
            compressedType = DDSCompressedType::DXT3;
            format         = RhiImageFormat::RhiImgFmt_BC2_UNORM_BLOCK;
        }
        else if (header->ddspf.dwFourCC == MAKEFOURCC('D', 'X', 'T', '4'))
        {
            compressedType = DDSCompressedType::DXT4;
            format         = RhiImageFormat::RhiImgFmt_BC3_UNORM_BLOCK;
        }
        else if (header->ddspf.dwFourCC == MAKEFOURCC('D', 'X', 'T', '5'))
        {
            compressedType = DDSCompressedType::DXT5;
            format         = RhiImageFormat::RhiImgFmt_BC3_UNORM_BLOCK;
        }
        else if (header->ddspf.dwFourCC == MAKEFOURCC('A', 'T', 'I', '1'))
        {
            compressedType = DDSCompressedType::ATI1;
            format         = RhiImageFormat::RhiImgFmt_BC4_UNORM_BLOCK;
        }
        else if (header->ddspf.dwFourCC == MAKEFOURCC('A', 'T', 'I', '2'))
        {
            compressedType = DDSCompressedType::ATI2;
            format         = RhiImageFormat::RhiImgFmt_BC5_UNORM_BLOCK;
        }
        else
        {
            char fourcc[5];
            fourcc[0] = (header->ddspf.dwFourCC >> 0) & 0xFF;
            fourcc[1] = (header->ddspf.dwFourCC >> 8) & 0xFF;
            fourcc[2] = (header->ddspf.dwFourCC >> 16) & 0xFF;
            fourcc[3] = (header->ddspf.dwFourCC >> 24) & 0xFF;
            fourcc[4] = '\0';
            iError("Unsupported FourCC: {}", fourcc);
            std::abort();
        }

        auto rhi = app->GetRhi();
        auto tex = rhi->CreateTexture2D("Asset_Tex", header->dwWidth, header->dwHeight, format,
            RhiImageUsage::RHI_IMAGE_USAGE_TRANSFER_DST_BIT, false);

        auto tq               = rhi->GetQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
        auto totalBodySize    = data.size() - bodyOffset;
        auto requiredBodySize = header->dwPitchOrLinearSize * header->dwHeight;
        if (format == RhiImageFormat::RhiImgFmt_UNDEFINED)
        {
            iError("Invalid format");
            std::abort();
        }
        else
        {
            // This is compressed format
            requiredBodySize = header->dwPitchOrLinearSize;
        }
        if (totalBodySize < requiredBodySize)
        {
            iError("Invalid body size: {}, required: {}", totalBodySize, requiredBodySize);
            std::abort();
        }
        auto buffer =
            rhi->CreateBuffer("Asset_Buf", requiredBodySize, RhiBufferUsage::RhiBufferUsage_CopySrc, true, false);
        buffer->MapMemory();
        buffer->WriteBuffer(data.data() + bodyOffset, requiredBodySize, 0);
        buffer->FlushBuffer();
        buffer->UnmapMemory();

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

        tq->RunSyncCommand([&](const RhiCommandList* cmd) {
            imageBarrier(cmd, tex.get(), RhiResourceState::Undefined, RhiResourceState::CopyDst, { 0, 0, 1, 1 });
            cmd->CopyBufferToImage(buffer.get(), tex.get(), { 0, 0, 1, 1 });
            imageBarrier(cmd, tex.get(), RhiResourceState::CopyDst, RhiResourceState::Common, { 0, 0, 1, 1 });
        });
        return tex;
    }

    IFRIT_APIDECL DirectDrawSurfaceAsset::DirectDrawSurfaceAsset(
        AssetMetadata metadata, std::filesystem::path path, IApplication* app)
        : TextureAsset(metadata, path), m_app(app)
    {
        // Pass
    }

    IFRIT_APIDECL Graphics::Rhi::RhiTextureRef DirectDrawSurfaceAsset::GetTexture()
    {
        if (m_texture == nullptr)
        {
            m_texture = parseDDS(m_path, m_app);
        }
        return m_texture;
    }

    // Importer
    IFRIT_APIDECL void DirectDrawSurfaceAssetImporter::ProcessMetadata(AssetMetadata& metadata)
    {
        metadata.m_importer = IMPORTER_NAME;
    }

    IFRIT_APIDECL Vec<String> DirectDrawSurfaceAssetImporter::GetSupportedExtensionNames() { return { ".dds" }; }

    IFRIT_APIDECL void        DirectDrawSurfaceAssetImporter::ImportAsset(
        const std::filesystem::path& path, AssetMetadata& metadata)
    {
        auto asset = std::make_shared<DirectDrawSurfaceAsset>(metadata, path, m_assetManager->GetApplication());
        m_assetManager->RegisterAsset(asset);
        // iInfo("Imported asset: [DDSTexture] {}", metadata.m_uuid);
    }

} // namespace Ifrit::Runtime