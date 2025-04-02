#pragma once

/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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
#include "ifrit/core/typing/CountRef.h"

namespace Ifrit::Graphics::Rhi
{

    class RhiBackend;
    class RhiContext;

    class RhiDeviceResource;
    class RhiBuffer;
    class RhiTexture;
    class RhiSampler;
    class RhiShader;
    class RhiPipeline;
    class RhiSwapchain;
    class RhiDevice;
    class RhiMultiBuffer;
    class RhiStagedSingleBuffer;

    class RhiCommandList;

    // Note here 'passes' are in fact 'pipeline references'
    // If two pass hold similar pipeline CI, they are the same
    class RhiComputePass;
    class RhiGraphicsPass;

    class RhiQueue;
    class RhiTaskSubmission;
    class RhiGraphicsQueue;
    class RhiComputeQueue;
    class RhiTransferQueue;

    class RhiBindlessDescriptorRef;
    struct RhiDescriptorHandle;

    class RhiRenderTargets;
    struct RhiRenderTargetsFormat;
    class RhiColorAttachment;
    class RhiDepthStencilAttachment;

    class RhiVertexBufferView;

    struct RhiImageSubResource;

    class RhiDeviceTimer;

    // Raytracing types
    struct RhiRTShaderGroup;
    struct RhiRTGeometryReference; // Geometry
    class RhiRTInstance;           // BLAS
    class RhiRTScene;              // TLAS
    class RhiRTShaderBindingTable; // SBT
    class RhiRTPipeline;           // Pipeline
    class RhiRTPass;

    // Enums
    enum RhiBufferUsage
    {
        RhiBufferUsage_CopySrc      = 0x00000001,
        RhiBufferUsage_CopyDst      = 0x00000002,
        RhiBufferUsage_UniformTexel = 0x00000004,
        RhiBufferUsage_StorageTexel = 0x00000008,
        RhiBufferUsage_Uniform      = 0x00000010,
        RhiBufferUsage_SSBO         = 0x00000020,
        RhiBufferUsage_Index        = 0x00000040,
        RhiBufferUsage_Vertex       = 0x00000080,
        RhiBufferUsage_Indirect     = 0x00000100,
        RhiBufferUsage_DeviceAddr   = 0x00020000,
    };

    enum RhiImageUsage
    {
        RHI_IMAGE_USAGE_TRANSFER_SRC_BIT             = 1,
        RHI_IMAGE_USAGE_TRANSFER_DST_BIT             = 2,
        RHI_IMAGE_USAGE_SAMPLED_BIT                  = 4,
        RHI_IMAGE_USAGE_STORAGE_BIT                  = 8,
        RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT         = 16,
        RHI_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT = 32,
        RHI_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT     = 64,
        RHI_IMAGE_USAGE_INPUT_ATTACHMENT_BIT         = 128,
    };

    enum RhiQueueCapability
    {
        RHI_QUEUE_GRAPHICS_BIT = 0x00000001,
        RHI_QUEUE_COMPUTE_BIT  = 0x00000002,
        RHI_QUEUE_TRANSFER_BIT = 0x00000004,
    };

    enum RhiBlendOp
    {
        RHI_BLEND_OP_ADD              = 0,
        RHI_BLEND_OP_SUBTRACT         = 1,
        RHI_BLEND_OP_REVERSE_SUBTRACT = 2,
        RHI_BLEND_OP_MIN              = 3,
        RHI_BLEND_OP_MAX              = 4,
    };

    enum RhiBlendFactor
    {
        RHI_BLEND_FACTOR_ZERO                     = 0,
        RHI_BLEND_FACTOR_ONE                      = 1,
        RHI_BLEND_FACTOR_SRC_COLOR                = 2,
        RHI_BLEND_FACTOR_ONE_MINUS_SRC_COLOR      = 3,
        RHI_BLEND_FACTOR_DST_COLOR                = 4,
        RHI_BLEND_FACTOR_ONE_MINUS_DST_COLOR      = 5,
        RHI_BLEND_FACTOR_SRC_ALPHA                = 6,
        RHI_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA      = 7,
        RHI_BLEND_FACTOR_DST_ALPHA                = 8,
        RHI_BLEND_FACTOR_ONE_MINUS_DST_ALPHA      = 9,
        RHI_BLEND_FACTOR_CONSTANT_COLOR           = 10,
        RHI_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR = 11,
        RHI_BLEND_FACTOR_CONSTANT_ALPHA           = 12,
        RHI_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA = 13,
        RHI_BLEND_FACTOR_SRC_ALPHA_SATURATE       = 14,
        RHI_BLEND_FACTOR_SRC1_COLOR               = 15,
        RHI_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR     = 16,
        RHI_BLEND_FACTOR_SRC1_ALPHA               = 17,
        RHI_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA     = 18,
        RHI_BLEND_FACTOR_MAX_ENUM                 = 0x7FFFFFFF
    };

    // These are just mapped from vulkan spec
    enum RhiImageFormat
    {
        RhiImgFmt_UNDEFINED                  = 0,
        RhiImgFmt_R4G4_UNORM_PACK8           = 1,
        RhiImgFmt_R4G4B4A4_UNORM_PACK16      = 2,
        RhiImgFmt_B4G4R4A4_UNORM_PACK16      = 3,
        RhiImgFmt_R5G6B5_UNORM_PACK16        = 4,
        RhiImgFmt_B5G6R5_UNORM_PACK16        = 5,
        RhiImgFmt_R5G5B5A1_UNORM_PACK16      = 6,
        RhiImgFmt_B5G5R5A1_UNORM_PACK16      = 7,
        RhiImgFmt_A1R5G5B5_UNORM_PACK16      = 8,
        RhiImgFmt_R8_UNORM                   = 9,
        RhiImgFmt_R8_SNORM                   = 10,
        RhiImgFmt_R8_USCALED                 = 11,
        RhiImgFmt_R8_SSCALED                 = 12,
        RhiImgFmt_R8_UINT                    = 13,
        RhiImgFmt_R8_SINT                    = 14,
        RhiImgFmt_R8_SRGB                    = 15,
        RhiImgFmt_R8G8_UNORM                 = 16,
        RhiImgFmt_R8G8_SNORM                 = 17,
        RhiImgFmt_R8G8_USCALED               = 18,
        RhiImgFmt_R8G8_SSCALED               = 19,
        RhiImgFmt_R8G8_UINT                  = 20,
        RhiImgFmt_R8G8_SINT                  = 21,
        RhiImgFmt_R8G8_SRGB                  = 22,
        RhiImgFmt_R8G8B8_UNORM               = 23,
        RhiImgFmt_R8G8B8_SNORM               = 24,
        RhiImgFmt_R8G8B8_USCALED             = 25,
        RhiImgFmt_R8G8B8_SSCALED             = 26,
        RhiImgFmt_R8G8B8_UINT                = 27,
        RhiImgFmt_R8G8B8_SINT                = 28,
        RhiImgFmt_R8G8B8_SRGB                = 29,
        RhiImgFmt_B8G8R8_UNORM               = 30,
        RhiImgFmt_B8G8R8_SNORM               = 31,
        RhiImgFmt_B8G8R8_USCALED             = 32,
        RhiImgFmt_B8G8R8_SSCALED             = 33,
        RhiImgFmt_B8G8R8_UINT                = 34,
        RhiImgFmt_B8G8R8_SINT                = 35,
        RhiImgFmt_B8G8R8_SRGB                = 36,
        RhiImgFmt_R8G8B8A8_UNORM             = 37,
        RhiImgFmt_R8G8B8A8_SNORM             = 38,
        RhiImgFmt_R8G8B8A8_USCALED           = 39,
        RhiImgFmt_R8G8B8A8_SSCALED           = 40,
        RhiImgFmt_R8G8B8A8_UINT              = 41,
        RhiImgFmt_R8G8B8A8_SINT              = 42,
        RhiImgFmt_R8G8B8A8_SRGB              = 43,
        RhiImgFmt_B8G8R8A8_UNORM             = 44,
        RhiImgFmt_B8G8R8A8_SNORM             = 45,
        RhiImgFmt_B8G8R8A8_USCALED           = 46,
        RhiImgFmt_B8G8R8A8_SSCALED           = 47,
        RhiImgFmt_B8G8R8A8_UINT              = 48,
        RhiImgFmt_B8G8R8A8_SINT              = 49,
        RhiImgFmt_B8G8R8A8_SRGB              = 50,
        RhiImgFmt_A8B8G8R8_UNORM_PACK32      = 51,
        RhiImgFmt_A8B8G8R8_SNORM_PACK32      = 52,
        RhiImgFmt_A8B8G8R8_USCALED_PACK32    = 53,
        RhiImgFmt_A8B8G8R8_SSCALED_PACK32    = 54,
        RhiImgFmt_A8B8G8R8_UINT_PACK32       = 55,
        RhiImgFmt_A8B8G8R8_SINT_PACK32       = 56,
        RhiImgFmt_A8B8G8R8_SRGB_PACK32       = 57,
        RhiImgFmt_A2R10G10B10_UNORM_PACK32   = 58,
        RhiImgFmt_A2R10G10B10_SNORM_PACK32   = 59,
        RhiImgFmt_A2R10G10B10_USCALED_PACK32 = 60,
        RhiImgFmt_A2R10G10B10_SSCALED_PACK32 = 61,
        RhiImgFmt_A2R10G10B10_UINT_PACK32    = 62,
        RhiImgFmt_A2R10G10B10_SINT_PACK32    = 63,
        RhiImgFmt_A2B10G10R10_UNORM_PACK32   = 64,
        RhiImgFmt_A2B10G10R10_SNORM_PACK32   = 65,
        RhiImgFmt_A2B10G10R10_USCALED_PACK32 = 66,
        RhiImgFmt_A2B10G10R10_SSCALED_PACK32 = 67,
        RhiImgFmt_A2B10G10R10_UINT_PACK32    = 68,
        RhiImgFmt_A2B10G10R10_SINT_PACK32    = 69,
        RhiImgFmt_R16_UNORM                  = 70,
        RhiImgFmt_R16_SNORM                  = 71,
        RhiImgFmt_R16_USCALED                = 72,
        RhiImgFmt_R16_SSCALED                = 73,
        RhiImgFmt_R16_UINT                   = 74,
        RhiImgFmt_R16_SINT                   = 75,
        RhiImgFmt_R16_SFLOAT                 = 76,
        RhiImgFmt_R16G16_UNORM               = 77,
        RhiImgFmt_R16G16_SNORM               = 78,
        RhiImgFmt_R16G16_USCALED             = 79,
        RhiImgFmt_R16G16_SSCALED             = 80,
        RhiImgFmt_R16G16_UINT                = 81,
        RhiImgFmt_R16G16_SINT                = 82,
        RhiImgFmt_R16G16_SFLOAT              = 83,
        RhiImgFmt_R16G16B16_UNORM            = 84,
        RhiImgFmt_R16G16B16_SNORM            = 85,
        RhiImgFmt_R16G16B16_USCALED          = 86,
        RhiImgFmt_R16G16B16_SSCALED          = 87,
        RhiImgFmt_R16G16B16_UINT             = 88,
        RhiImgFmt_R16G16B16_SINT             = 89,
        RhiImgFmt_R16G16B16_SFLOAT           = 90,
        RhiImgFmt_R16G16B16A16_UNORM         = 91,
        RhiImgFmt_R16G16B16A16_SNORM         = 92,
        RhiImgFmt_R16G16B16A16_USCALED       = 93,
        RhiImgFmt_R16G16B16A16_SSCALED       = 94,
        RhiImgFmt_R16G16B16A16_UINT          = 95,
        RhiImgFmt_R16G16B16A16_SINT          = 96,
        RhiImgFmt_R16G16B16A16_SFLOAT        = 97,
        RhiImgFmt_R32_UINT                   = 98,
        RhiImgFmt_R32_SINT                   = 99,
        RhiImgFmt_R32_SFLOAT                 = 100,
        RhiImgFmt_R32G32_UINT                = 101,
        RhiImgFmt_R32G32_SINT                = 102,
        RhiImgFmt_R32G32_SFLOAT              = 103,
        RhiImgFmt_R32G32B32_UINT             = 104,
        RhiImgFmt_R32G32B32_SINT             = 105,
        RhiImgFmt_R32G32B32_SFLOAT           = 106,
        RhiImgFmt_R32G32B32A32_UINT          = 107,
        RhiImgFmt_R32G32B32A32_SINT          = 108,
        RhiImgFmt_R32G32B32A32_SFLOAT        = 109,
        RhiImgFmt_R64_UINT                   = 110,
        RhiImgFmt_R64_SINT                   = 111,
        RhiImgFmt_R64_SFLOAT                 = 112,
        RhiImgFmt_R64G64_UINT                = 113,
        RhiImgFmt_R64G64_SINT                = 114,
        RhiImgFmt_R64G64_SFLOAT              = 115,
        RhiImgFmt_R64G64B64_UINT             = 116,
        RhiImgFmt_R64G64B64_SINT             = 117,
        RhiImgFmt_R64G64B64_SFLOAT           = 118,
        RhiImgFmt_R64G64B64A64_UINT          = 119,
        RhiImgFmt_R64G64B64A64_SINT          = 120,
        RhiImgFmt_R64G64B64A64_SFLOAT        = 121,
        RhiImgFmt_B10G11R11_UFLOAT_PACK32    = 122,
        RhiImgFmt_E5B9G9R9_UFLOAT_PACK32     = 123,
        RhiImgFmt_D16_UNORM                  = 124,
        RhiImgFmt_X8_D24_UNORM_PACK32        = 125,
        RhiImgFmt_D32_SFLOAT                 = 126,
        RhiImgFmt_S8_UINT                    = 127,
        RhiImgFmt_D16_UNORM_S8_UINT          = 128,
        RhiImgFmt_D24_UNORM_S8_UINT          = 129,
        RhiImgFmt_D32_SFLOAT_S8_UINT         = 130,
        RhiImgFmt_BC1_RGB_UNORM_BLOCK        = 131,
        RhiImgFmt_BC1_RGB_SRGB_BLOCK         = 132,
        RhiImgFmt_BC1_RGBA_UNORM_BLOCK       = 133,
        RhiImgFmt_BC1_RGBA_SRGB_BLOCK        = 134,
        RhiImgFmt_BC2_UNORM_BLOCK            = 135,
        RhiImgFmt_BC2_SRGB_BLOCK             = 136,
        RhiImgFmt_BC3_UNORM_BLOCK            = 137,
        RhiImgFmt_BC3_SRGB_BLOCK             = 138,
        RhiImgFmt_BC4_UNORM_BLOCK            = 139,
        RhiImgFmt_BC4_SNORM_BLOCK            = 140,
        RhiImgFmt_BC5_UNORM_BLOCK            = 141,
        RhiImgFmt_BC5_SNORM_BLOCK            = 142,
        RhiImgFmt_BC6H_UFLOAT_BLOCK          = 143,
        RhiImgFmt_BC6H_SFLOAT_BLOCK          = 144,
        RhiImgFmt_BC7_UNORM_BLOCK            = 145,
        RhiImgFmt_BC7_SRGB_BLOCK             = 146,
        RhiImgFmt_ETC2_R8G8B8_UNORM_BLOCK    = 147,
        RhiImgFmt_ETC2_R8G8B8_SRGB_BLOCK     = 148,
        RhiImgFmt_ETC2_R8G8B8A1_UNORM_BLOCK  = 149,
        RhiImgFmt_ETC2_R8G8B8A1_SRGB_BLOCK   = 150,
        RhiImgFmt_ETC2_R8G8B8A8_UNORM_BLOCK  = 151,
        RhiImgFmt_ETC2_R8G8B8A8_SRGB_BLOCK   = 152,
        RhiImgFmt_EAC_R11_UNORM_BLOCK        = 153,
        RhiImgFmt_EAC_R11_SNORM_BLOCK        = 154,
        RhiImgFmt_EAC_R11G11_UNORM_BLOCK     = 155,
        RhiImgFmt_EAC_R11G11_SNORM_BLOCK     = 156,
        RhiImgFmt_ASTC_4x4_UNORM_BLOCK       = 157,
        RhiImgFmt_ASTC_4x4_SRGB_BLOCK        = 158,
        RhiImgFmt_ASTC_5x4_UNORM_BLOCK       = 159,
        RhiImgFmt_ASTC_5x4_SRGB_BLOCK        = 160,
        RhiImgFmt_ASTC_5x5_UNORM_BLOCK       = 161,
        RhiImgFmt_ASTC_5x5_SRGB_BLOCK        = 162,
        RhiImgFmt_ASTC_6x5_UNORM_BLOCK       = 163,
        RhiImgFmt_ASTC_6x5_SRGB_BLOCK        = 164,
        RhiImgFmt_ASTC_6x6_UNORM_BLOCK       = 165,
        RhiImgFmt_ASTC_6x6_SRGB_BLOCK        = 166,
        RhiImgFmt_ASTC_8x5_UNORM_BLOCK       = 167,
        RhiImgFmt_ASTC_8x5_SRGB_BLOCK        = 168,
        RhiImgFmt_ASTC_8x6_UNORM_BLOCK       = 169,
        RhiImgFmt_ASTC_8x6_SRGB_BLOCK        = 170,
        RhiImgFmt_ASTC_8x8_UNORM_BLOCK       = 171,
        RhiImgFmt_ASTC_8x8_SRGB_BLOCK        = 172,
        RhiImgFmt_ASTC_10x5_UNORM_BLOCK      = 173,
        RhiImgFmt_ASTC_10x5_SRGB_BLOCK       = 174,
        RhiImgFmt_ASTC_10x6_UNORM_BLOCK      = 175,
        RhiImgFmt_ASTC_10x6_SRGB_BLOCK       = 176,
        RhiImgFmt_ASTC_10x8_UNORM_BLOCK      = 177,
        RhiImgFmt_ASTC_10x8_SRGB_BLOCK       = 178,
        RhiImgFmt_ASTC_10x10_UNORM_BLOCK     = 179,
        RhiImgFmt_ASTC_10x10_SRGB_BLOCK      = 180,
        RhiImgFmt_ASTC_12x10_UNORM_BLOCK     = 181,
        RhiImgFmt_ASTC_12x10_SRGB_BLOCK      = 182,
        RhiImgFmt_ASTC_12x12_UNORM_BLOCK     = 183,
        RhiImgFmt_ASTC_12x12_SRGB_BLOCK      = 184
    };

    enum class RhiShaderStage
    {
        Vertex,
        Fragment,
        Compute,
        Mesh,
        Task,
        RTRayGen,
        RTMiss,
        RTAnyHit,
        RTClosestHit,
        RTIntersection,
        RTCallable,
    };
    enum class RhiShaderSourceType
    {
        GLSLCode,
        Binary
    };
    enum class RhiVertexInputRate
    {
        Vertex,
        Instance
    };

    enum class RhiDescriptorBindPoint
    {
        Compute,
        Graphics
    };

    enum class RhiDescriptorType
    {
        UniformBuffer,
        StorageBuffer,
        CombinedImageSampler,
        StorageImage,
        MaxEnum
    };

    enum class RhiResourceAccessType
    {
        Read,
        Write,
        ReadOrWrite,
        ReadAndWrite,
    };

    enum class RhiRenderTargetLoadOp
    {
        Load,
        Clear,
        DontCare
    };
    enum class RhiCullMode
    {
        None,
        Front,
        Back
    };
    enum class RhiRasterizerTopology
    {
        TriangleList,
        Line,
        Point
    };
    enum class RhiGeometryGenerationType
    {
        Conventional,
        Mesh
    };

    enum class RhiResourceType
    {
        Buffer,
        Texture
    };
    enum class RhiCompareOp
    {
        Never,
        Less,
        Equal,
        LessOrEqual,
        Greater,
        NotEqual,
        GreaterOrEqual,
        Always
    };

    enum class RhiResourceState
    {
        Undefined,
        Common,
        ColorRT,
        DepthStencilRT,
        ShaderRead,
        UnorderedAccess,
        CopySrc,
        CopyDst,
        Present,

        AutoTraced,
    };
    enum class RhiBarrierType
    {
        UAVAccess,
        Transition
    };

    using RhiDeviceAddr = u64;

} // namespace Ifrit::Graphics::Rhi

namespace Ifrit::Graphics::Rhi
{
    using RhiTextureRef = RCountRef<RhiTexture>;
    using RhiSamplerRef = RCountRef<RhiSampler>;
    using RhiBufferRef  = RCountRef<RhiBuffer>;

} // namespace Ifrit::Graphics::Rhi