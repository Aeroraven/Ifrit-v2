#pragma once
#include "ifrit/imaging/compress/CompressedTextureUtil.h"
#include "ifrit/core/logging/Logging.h"
#include "ktx.h"
#include "VkFormatList.h"

namespace Ifrit::Imaging::Compress
{
    IFRIT_APIDECL void WriteTex2DToBc7File(const void* data, u32 size, const String& outFile, TextureFormat fmt,
        u32 baseWidth, u32 baseHeight, u32 baseDepth, u32 quality)
    {
        iAssertion(data, "Compress: data is null");
        iAssertion(size > 0, "Compress: size is 0");

        ktxTexture2*         texture;
        ktxTextureCreateInfo createInfo;
        KTX_error_code       result;
        ktx_uint32_t         level, layer, faceSlice;
        const u8*            src;
        ktx_size_t           srcSize;
        ktxBasisParams       params = { 0 };
        params.structSize           = sizeof(params);

        createInfo.glInternalformat = 0; // Ignored as we'll create a KTX2 texture.
        switch (fmt)
        {
            case TextureFormat::RGBA8_UNORM:
                createInfo.vkFormat = VK_FORMAT_R8G8B8A8_UNORM;
                break;
            default:
                iAssertion(false, "Compress: Unsupported format");
                break;
        }
        createInfo.baseWidth     = baseWidth;
        createInfo.baseHeight    = baseHeight;
        createInfo.baseDepth     = baseDepth;
        createInfo.numDimensions = 2 + (baseDepth > 1 ? 1 : 0); // 2D or 3D texture
        // Note: it is not necessary to provide a full mipmap pyramid.
        createInfo.numLevels       = 1;
        createInfo.numLayers       = 1;
        createInfo.numFaces        = 1;
        createInfo.isArray         = KTX_FALSE;
        createInfo.generateMipmaps = KTX_FALSE;

        result = ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture);

        src       = static_cast<const u8*>(data);
        srcSize   = size;
        level     = 0;
        layer     = 0;
        faceSlice = 0;
        result    = ktxTexture_SetImageFromMemory(ktxTexture(texture), level, layer, faceSlice, src, srcSize);
        // Repeat for the other 15 slices of the base level and all other levels
        // up to createInfo.numLevels.

        // params.threadCount    = 1;
        // params.blockDimension = KTX_PACK_ASTC_BLOCK_DIMENSION_6x6;
        // params.mode           = KTX_PACK_ASTC_ENCODER_MODE_LDR;
        // params.qualityLevel   = quality;
        // result                = ktxTexture2_CompressAstcEx(texture, &params);

        params.compressionLevel = KTX_ETC1S_DEFAULT_COMPRESSION_LEVEL;
        params.uastc            = KTX_TRUE;
        result                  = ktxTexture2_CompressBasisEx(texture, &params);

        result = ktxTexture2_TranscodeBasis(texture, KTX_TTF_BC7_RGBA, 0);

        ktxTexture_WriteToNamedFile(ktxTexture(texture), outFile.c_str());
        ktxTexture_Destroy(ktxTexture(texture));
    }

    IFRIT_APIDECL void ReadBc7Tex2DFromFile(void** data, u32& size, const String& inFile, TextureFormat& fmt,
        u32& baseWidth, u32& baseHeight, u32& baseDepth)
    {
        ktxTexture*    texture;
        KTX_error_code result;
        ktx_size_t     offset;
        ktx_uint8_t*   image;
        ktx_uint32_t   level, layer, faceSlice, sliceSize;

        result = ktxTexture_CreateFromNamedFile(inFile.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);
        ktx_uint32_t numLevels = texture->numLevels;
        ktx_bool_t   isArray   = texture->isArray;

        // Retrieve a pointer to the image for a specific mip level, array layer
        // & face or depth slice.
        level     = 0;
        layer     = 0;
        faceSlice = 0;
        result    = ktxTexture_GetImageOffset(texture, level, layer, faceSlice, &offset);
        sliceSize = ktxTexture_GetImageSize(texture, level);
        image     = ktxTexture_GetData(texture) + offset;

        fmt        = TextureFormat::BC7_UNORM;
        baseWidth  = texture->baseWidth;
        baseHeight = texture->baseHeight;
        baseDepth  = texture->baseDepth;
        size       = sliceSize;
        *data      = new u8[sliceSize];
        memcpy(*data, image, sliceSize);
        ktxTexture_Destroy(texture);
    }
} // namespace Ifrit::Imaging::Compress