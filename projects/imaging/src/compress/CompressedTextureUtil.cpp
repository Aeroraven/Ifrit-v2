#pragma once
#include "ifrit/imaging/compress/CompressedTextureUtil.h"
#include "ifrit/core/logging/Logging.h"
#include "ktx.h"
#include "VkFormatList.h"

namespace Ifrit::Imaging::Compress
{
    IFRIT_IMAGING_API void DiscardBAChannel(
        const RSizedBuffer& in, RSizedBuffer& out, u32 width, u32 height, u32 depth, u32 inChannels, u32 channelWidth)
    {
        iAssertion(in.GetSize() > 0, "Compress: size is 0");
        iAssertion(inChannels == 4, "Compress: only RGBA format is supported");

        out = RSizedBuffer(width * height * depth * 2);
        Vec<u8> outData(width * height * depth * 2);
        for (u32 i = 0; i < width * height * depth; ++i)
        {
            outData[i * 2]     = in[i * inChannels + 0];
            outData[i * 2 + 1] = in[i * inChannels + 1];
        }
        out.CopyFromRaw(outData.data(), out.GetSize());
    }

    IFRIT_IMAGING_API void WriteTex2DToBlockCompressedFile(const RSizedBuffer& in, const String& outFile,
        TextureFormat fmt, u32 baseWidth, u32 baseHeight, u32 baseDepth, CompressionAlgo algo)
    {
        iAssertion(in.GetSize() > 0, "Compress: size is 0");

        ktxTexture2*         texture;
        ktxTextureCreateInfo createInfo;
        KTX_error_code       result;
        ktx_uint32_t         level, layer, faceSlice;
        const u8*            src;
        ktx_size_t           srcSize;
        ktxBasisParams       params = { 0 };
        params.structSize           = sizeof(params);
        createInfo.glInternalformat = 0; // Ignored as we'll create a KTX2 texture.
        u32 channels                = 0;
        switch (fmt)
        {
            case TextureFormat::RGBA8_UNORM:
                createInfo.vkFormat = VK_FORMAT_R8G8B8A8_UNORM;
                channels            = 4;
                break;
            case TextureFormat::RG8_UNORM:
                createInfo.vkFormat = VK_FORMAT_R8G8_UNORM;
                channels            = 2;
                break;
            default:
                iAssertion(false, "Compress: Unsupported format");
                break;
        }
        createInfo.baseWidth       = baseWidth;
        createInfo.baseHeight      = baseHeight;
        createInfo.baseDepth       = baseDepth;
        createInfo.numDimensions   = 2 + (baseDepth > 1 ? 1 : 0);
        createInfo.numLevels       = 1;
        createInfo.numLayers       = 1;
        createInfo.numFaces        = 1;
        createInfo.isArray         = KTX_FALSE;
        createInfo.generateMipmaps = KTX_FALSE;

        result = ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture);

        src       = reinterpret_cast<const u8*>(in.GetData());
        srcSize   = in.GetSize();
        level     = 0;
        layer     = 0;
        faceSlice = 0;
        result    = ktxTexture_SetImageFromMemory(ktxTexture(texture), level, layer, faceSlice, src, srcSize);

        channels = ktxTexture2_GetNumComponents(texture);

        params.compressionLevel = KTX_ETC1S_DEFAULT_COMPRESSION_LEVEL;
        params.uastc            = KTX_TRUE;

        if (algo == CompressionAlgo::BC7)
        {
            result = ktxTexture2_CompressBasisEx(texture, &params);
            iAssertion(channels == 4, "Compress: BC7 requires RGBA format");
            result = ktxTexture2_TranscodeBasis(texture, KTX_TTF_BC7_RGBA, 0);
        }
        else if (algo == CompressionAlgo::BC5)
        {
            params.normalMap = KTX_TRUE;
            result           = ktxTexture2_CompressBasisEx(texture, &params);
            iAssertion(channels == 2, "Compress: BC5 requires RG format");
            result = ktxTexture2_TranscodeBasis(texture, KTX_TTF_BC5_RG, 0);
        }

        ktxTexture_WriteToNamedFile(ktxTexture(texture), outFile.c_str());
        ktxTexture_Destroy(ktxTexture(texture));
    }

    IFRIT_APIDECL void ReadBlockCompressedTex2DFromFile(
        RSizedBuffer& out, const String& inFile, u32& baseWidth, u32& baseHeight, u32& baseDepth)
    {
        ktxTexture*    texture;
        KTX_error_code result;
        ktx_size_t     offset;
        ktx_uint8_t*   image;
        ktx_uint32_t   level, layer, faceSlice, sliceSize;

        result = ktxTexture_CreateFromNamedFile(inFile.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);

        level     = 0;
        layer     = 0;
        faceSlice = 0;
        result    = ktxTexture_GetImageOffset(texture, level, layer, faceSlice, &offset);
        sliceSize = ktxTexture_GetImageSize(texture, level);
        image     = ktxTexture_GetData(texture) + offset;

        baseWidth  = texture->baseWidth;
        baseHeight = texture->baseHeight;
        baseDepth  = texture->baseDepth;
        out.CopyFromRaw(image, sliceSize);
        ktxTexture_Destroy(texture);
    }
} // namespace Ifrit::Imaging::Compress