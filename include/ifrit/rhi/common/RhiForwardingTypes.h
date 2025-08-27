#pragma once
#include "ifrit/core/typing/CountRef.h"

namespace Ifrit::RHI
{

    class RhiBackend;
    class RhiContext;

    class RhiDeviceResource;
    class RhiBuffer;
    class RhiTexture;
    class RhiSampler;

    class RhiPipeline;
    class RhiSwapchain;
    class RhiDevice;
    class RhiMultiBuffer;
    class RhiStagedSingleBuffer;

    class RhiCommandListContext;

    class RhiCommandListBase;

    class RhiShader;
    class RhiShaderCollection;

    // Note here 'passes' are in fact 'pipeline references'
    // If two pass hold similar pipeline CI, they are the same
    class RhiComputePass;
    class RhiGraphicsPass;

    class RhiQueue;
    class RhiTaskSubmission;
    class RhiGraphicsQueue;
    class RhiComputeQueue;
    class RhiTransferQueue;

    class RhiResourceView;
    class RhiShaderReadView;
    class RhiUnorderedAccessView;
    class RhiConstantBufferView;

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

    enum class ERhiResourceType : u32
    {
        Unknown      = 0,
        Buffer       = 1,
        Texture      = 2,
        SamplerState = 3,
        View         = 4,
    };

    namespace ERhiBufferUsageFlag
    {
        enum Enum : u32
        {
            None             = 0,
            CopySrc          = 1 << 0,
            CopyDst          = 1 << 1,
            UniformBuffer    = 1 << 4,
            UnorderedAccess  = 1 << 5,
            IndexBuffer      = 1 << 6,
            VertexBuffer     = 1 << 7,
            IndirectBuffer   = 1 << 8,
            DeviceAddr       = 1 << 9,
            StructuredBuffer = 1 << 10,

            CPUAccess = 1 << 16,

            // Dynamic buffers are shortlived buffer, w/ reference to UE5 and NVRHI
            Dynamic = 1 << 17
        };
    } // namespace ERhiBufferUsageFlag

    using ERhiBufferUsage = u32;

    enum class ERhiImageDimension
    {
        Unknown          = 0,
        Texture2D        = 1,
        Texture2DArray   = 2,
        Texture3D        = 3,
        TextureCube      = 4,
        TextureCubeArray = 5
    };

    namespace ERhiImageUsageFlag
    {
        enum Enum : u64
        {
            None = 0,

            // Update 250824: these three are always on
            // CopySrc         = 1 << 1,
            // CopyDst         = 1 << 2,
            // ShaderRead      = 1 << 3,

            UnorderedAccess = 1 << 4,
            RenderTarget    = 1 << 5,
            Depth           = 1 << 6,
            Transient       = 1 << 7,
            InputAttachment = 1 << 8,
            Presentable     = 1 << 9,

            CPUWritable = 1 << 16
        };
    } // namespace ERhiImageUsageFlag
    using ERhiImageUsage = u64;

    namespace ERhiQueueCapabilityFlags
    {
        enum Enum
        {
            Graphics   = 0x00000001,
            Compute    = 0x00000002,
            Transfer   = 0x00000004,
            Raytracing = 0x00000008,
        };
    } // namespace ERhiQueueCapabilityFlags
    using ERhiQueueCapability = ERhiQueueCapabilityFlags::Enum;

    enum class ERhiPipelineType
    {
        Graphics = 0,
        Compute  = 1,
        Transfer = 2,
    };

    enum class ERhiVendor
    {
        Any = 0x7fffffff,

        Amd      = 0x1002,
        ImgTec   = 0x1010,
        Nvidia   = 0x10de,
        Arm      = 0x13b5,
        Qualcomm = 0x5143,
        Intel    = 0x8086,
    };

    enum class ERhiBlendOp
    {
        Add             = 0,
        Subtract        = 1,
        ReverseSubtract = 2,
        Min             = 3,
        Max             = 4,
    };

    enum class ERhiBlendFactor : u32
    {
        Zero                  = 0,
        One                   = 1,
        SrcColor              = 2,
        OneMinusSrcColor      = 3,
        DstColor              = 4,
        OneMinusDstColor      = 5,
        SrcAlpha              = 6,
        OneMinusSrcAlpha      = 7,
        DstAlpha              = 8,
        OneMinusDstAlpha      = 9,
        ConstantColor         = 10,
        OneMinusConstantColor = 11,
        ConstantAlpha         = 12,
        OneMinusConstantAlpha = 13,
        SrcAlphaSaturate      = 14,
        Src1Color             = 15,
        OneMinusSrc1Color     = 16,
        Src1Alpha             = 17,
        OneMinusSrc1Alpha     = 18,
    };

    // These are just mapped from vulkan spec
    enum class ERhiImageFormat : u32
    {
        Undefined                  = 0,
        R4G4_UNORM_PACK8           = 1,
        R4G4B4A4_UNORM_PACK16      = 2,
        B4G4R4A4_UNORM_PACK16      = 3,
        R5G6B5_UNORM_PACK16        = 4,
        B5G6R5_UNORM_PACK16        = 5,
        R5G5B5A1_UNORM_PACK16      = 6,
        B5G5R5A1_UNORM_PACK16      = 7,
        A1R5G5B5_UNORM_PACK16      = 8,
        R8_UNORM                   = 9,
        R8_SNORM                   = 10,
        R8_USCALED                 = 11,
        R8_SSCALED                 = 12,
        R8_UINT                    = 13,
        R8_SINT                    = 14,
        R8_SRGB                    = 15,
        R8G8_UNORM                 = 16,
        R8G8_SNORM                 = 17,
        R8G8_USCALED               = 18,
        R8G8_SSCALED               = 19,
        R8G8_UINT                  = 20,
        R8G8_SINT                  = 21,
        R8G8_SRGB                  = 22,
        R8G8B8_UNORM               = 23,
        R8G8B8_SNORM               = 24,
        R8G8B8_USCALED             = 25,
        R8G8B8_SSCALED             = 26,
        R8G8B8_UINT                = 27,
        R8G8B8_SINT                = 28,
        R8G8B8_SRGB                = 29,
        B8G8R8_UNORM               = 30,
        B8G8R8_SNORM               = 31,
        B8G8R8_USCALED             = 32,
        B8G8R8_SSCALED             = 33,
        B8G8R8_UINT                = 34,
        B8G8R8_SINT                = 35,
        B8G8R8_SRGB                = 36,
        R8G8B8A8_UNORM             = 37,
        R8G8B8A8_SNORM             = 38,
        R8G8B8A8_USCALED           = 39,
        R8G8B8A8_SSCALED           = 40,
        R8G8B8A8_UINT              = 41,
        R8G8B8A8_SINT              = 42,
        R8G8B8A8_SRGB              = 43,
        B8G8R8A8_UNORM             = 44,
        B8G8R8A8_SNORM             = 45,
        B8G8R8A8_USCALED           = 46,
        B8G8R8A8_SSCALED           = 47,
        B8G8R8A8_UINT              = 48,
        B8G8R8A8_SINT              = 49,
        B8G8R8A8_SRGB              = 50,
        A8B8G8R8_UNORM_PACK32      = 51,
        A8B8G8R8_SNORM_PACK32      = 52,
        A8B8G8R8_USCALED_PACK32    = 53,
        A8B8G8R8_SSCALED_PACK32    = 54,
        A8B8G8R8_UINT_PACK32       = 55,
        A8B8G8R8_SINT_PACK32       = 56,
        A8B8G8R8_SRGB_PACK32       = 57,
        A2R10G10B10_UNORM_PACK32   = 58,
        A2R10G10B10_SNORM_PACK32   = 59,
        A2R10G10B10_USCALED_PACK32 = 60,
        A2R10G10B10_SSCALED_PACK32 = 61,
        A2R10G10B10_UINT_PACK32    = 62,
        A2R10G10B10_SINT_PACK32    = 63,
        A2B10G10R10_UNORM_PACK32   = 64,
        A2B10G10R10_SNORM_PACK32   = 65,
        A2B10G10R10_USCALED_PACK32 = 66,
        A2B10G10R10_SSCALED_PACK32 = 67,
        A2B10G10R10_UINT_PACK32    = 68,
        A2B10G10R10_SINT_PACK32    = 69,
        R16_UNORM                  = 70,
        R16_SNORM                  = 71,
        R16_USCALED                = 72,
        R16_SSCALED                = 73,
        R16_UINT                   = 74,
        R16_SINT                   = 75,
        R16_SFLOAT                 = 76,
        R16G16_UNORM               = 77,
        R16G16_SNORM               = 78,
        R16G16_USCALED             = 79,
        R16G16_SSCALED             = 80,
        R16G16_UINT                = 81,
        R16G16_SINT                = 82,
        R16G16_SFLOAT              = 83,
        R16G16B16_UNORM            = 84,
        R16G16B16_SNORM            = 85,
        R16G16B16_USCALED          = 86,
        R16G16B16_SSCALED          = 87,
        R16G16B16_UINT             = 88,
        R16G16B16_SINT             = 89,
        R16G16B16_SFLOAT           = 90,
        R16G16B16A16_UNORM         = 91,
        R16G16B16A16_SNORM         = 92,
        R16G16B16A16_USCALED       = 93,
        R16G16B16A16_SSCALED       = 94,
        R16G16B16A16_UINT          = 95,
        R16G16B16A16_SINT          = 96,
        R16G16B16A16_SFLOAT        = 97,
        R32_UINT                   = 98,
        R32_SINT                   = 99,
        R32_SFLOAT                 = 100,
        R32G32_UINT                = 101,
        R32G32_SINT                = 102,
        R32G32_SFLOAT              = 103,
        R32G32B32_UINT             = 104,
        R32G32B32_SINT             = 105,
        R32G32B32_SFLOAT           = 106,
        R32G32B32A32_UINT          = 107,
        R32G32B32A32_SINT          = 108,
        R32G32B32A32_SFLOAT        = 109,
        R64_UINT                   = 110,
        R64_SINT                   = 111,
        R64_SFLOAT                 = 112,
        R64G64_UINT                = 113,
        R64G64_SINT                = 114,
        R64G64_SFLOAT              = 115,
        R64G64B64_UINT             = 116,
        R64G64B64_SINT             = 117,
        R64G64B64_SFLOAT           = 118,
        R64G64B64A64_UINT          = 119,
        R64G64B64A64_SINT          = 120,
        R64G64B64A64_SFLOAT        = 121,
        B10G11R11_UFLOAT_PACK32    = 122,
        E5B9G9R9_UFLOAT_PACK32     = 123,
        D16_UNORM                  = 124,
        X8_D24_UNORM_PACK32        = 125,
        D32_SFLOAT                 = 126,
        S8_UINT                    = 127,
        D16_UNORM_S8_UINT          = 128,
        D24_UNORM_S8_UINT          = 129,
        D32_SFLOAT_S8_UINT         = 130,
        BC1_RGB_UNORM_BLOCK        = 131,
        BC1_RGB_SRGB_BLOCK         = 132,
        BC1_RGBA_UNORM_BLOCK       = 133,
        BC1_RGBA_SRGB_BLOCK        = 134,
        BC2_UNORM_BLOCK            = 135,
        BC2_SRGB_BLOCK             = 136,
        BC3_UNORM_BLOCK            = 137,
        BC3_SRGB_BLOCK             = 138,
        BC4_UNORM_BLOCK            = 139,
        BC4_SNORM_BLOCK            = 140,
        BC5_UNORM_BLOCK            = 141,
        BC5_SNORM_BLOCK            = 142,
        BC6H_UFLOAT_BLOCK          = 143,
        BC6H_SFLOAT_BLOCK          = 144,
        BC7_UNORM_BLOCK            = 145,
        BC7_SRGB_BLOCK             = 146,
        ETC2_R8G8B8_UNORM_BLOCK    = 147,
        ETC2_R8G8B8_SRGB_BLOCK     = 148,
        ETC2_R8G8B8A1_UNORM_BLOCK  = 149,
        ETC2_R8G8B8A1_SRGB_BLOCK   = 150,
        ETC2_R8G8B8A8_UNORM_BLOCK  = 151,
        ETC2_R8G8B8A8_SRGB_BLOCK   = 152,
        EAC_R11_UNORM_BLOCK        = 153,
        EAC_R11_SNORM_BLOCK        = 154,
        EAC_R11G11_UNORM_BLOCK     = 155,
        EAC_R11G11_SNORM_BLOCK     = 156,
        ASTC_4x4_UNORM_BLOCK       = 157,
        ASTC_4x4_SRGB_BLOCK        = 158,
        ASTC_5x4_UNORM_BLOCK       = 159,
        ASTC_5x4_SRGB_BLOCK        = 160,
        ASTC_5x5_UNORM_BLOCK       = 161,
        ASTC_5x5_SRGB_BLOCK        = 162,
        ASTC_6x5_UNORM_BLOCK       = 163,
        ASTC_6x5_SRGB_BLOCK        = 164,
        ASTC_6x6_UNORM_BLOCK       = 165,
        ASTC_6x6_SRGB_BLOCK        = 166,
        ASTC_8x5_UNORM_BLOCK       = 167,
        ASTC_8x5_SRGB_BLOCK        = 168,
        ASTC_8x6_UNORM_BLOCK       = 169,
        ASTC_8x6_SRGB_BLOCK        = 170,
        ASTC_8x8_UNORM_BLOCK       = 171,
        ASTC_8x8_SRGB_BLOCK        = 172,
        ASTC_10x5_UNORM_BLOCK      = 173,
        ASTC_10x5_SRGB_BLOCK       = 174,
        ASTC_10x6_UNORM_BLOCK      = 175,
        ASTC_10x6_SRGB_BLOCK       = 176,
        ASTC_10x8_UNORM_BLOCK      = 177,
        ASTC_10x8_SRGB_BLOCK       = 178,
        ASTC_10x10_UNORM_BLOCK     = 179,
        ASTC_10x10_SRGB_BLOCK      = 180,
        ASTC_12x10_UNORM_BLOCK     = 181,
        ASTC_12x10_SRGB_BLOCK      = 182,
        ASTC_12x12_UNORM_BLOCK     = 183,
        ASTC_12x12_SRGB_BLOCK      = 184
    };

    enum class ERhiShaderStage
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

    enum class ERhiShaderSourceType
    {
        GLSLCode,
        HLSLCode,
        SlangCode,
        Binary
    };

    enum class ERhiVertexInputRate
    {
        Vertex,
        Instance
    };

    enum class ERhiDescriptorBindPoint
    {
        Compute,
        Graphics
    };

    enum class ERhiDescriptorType
    {
        UniformBuffer,        // CBV
        StorageBuffer,        // SRV
        RWStorageBuffer,      // UAV
        CombinedImageSampler, // Considering compatibility problems
        StorageImage,         // UAV
        SampledImage,         // SRV
        Sampler,
        MaxEnum
    };

    enum class ERhiResourceAccessType
    {
        Read,
        Write,
        ReadOrWrite,
        ReadAndWrite,
    };

    enum class ERhiRenderTargetLoadOp
    {
        Load,
        Clear,
        LoadNoStore,
        ClearNoStore,
        DontCare
    };
    enum class ERhiCullMode
    {
        None,
        Front,
        Back
    };
    enum class ERhiRasterizerTopology
    {
        TriangleList,
        Line,
        Point
    };
    enum class ERhiGeometryGenerationType
    {
        Conventional,
        Mesh
    };

    enum class ERhiCompareOp
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

    enum class ERhiResourceState
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
    };
    enum class ERhiBarrierType
    {
        UAVAccess,
        Transition
    };

    enum class ERhiSamplerFilter
    {
        Nearest,
        Linear
    };

    enum class ERhiSamplerWrapMode
    {
        Clamp,
        Repeat,
    };

    enum class ERhiDepthFunc
    {
        Never,
        Less,
        Equal,
        Greater
    };

    using RhiDeviceAddr = u64;

    enum class ERhiCommandListPipelineType
    {
        Invalid,
        Graphics,
        Compute,
        Transfer,
    };

    enum class ERhiCommandListState
    {
        Invalid,
        ReadyToRecord,
        Recording,
        ReadyToSubmit,
        Submitted,
    };

    enum class ERhiCommandListType
    {
        Invalid,
        Active,
        Upload,
    };

} // namespace Ifrit::RHI

namespace Ifrit::RHI
{
    using RhiTextureRef = TCountRef<RhiTexture>;
    using RhiSamplerRef = TCountRef<RhiSampler>;
    using RhiBufferRef  = TCountRef<RhiBuffer>;

    using RhiUAVRef = TCountRef<RhiUnorderedAccessView>;
    using RhiSRVRef = TCountRef<RhiShaderReadView>;
    using RhiCBVRef = TCountRef<RhiConstantBufferView>;

    using RhiRawHandle = void*;
} // namespace Ifrit::RHI