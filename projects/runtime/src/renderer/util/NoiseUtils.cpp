#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/logging/Logging.h"

#include "ifrit/runtime/renderer/util/NoiseUtils.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#undef STB_IMAGE_IMPLEMENTATION

using namespace Ifrit::Graphics::Rhi;

namespace Ifrit::Runtime::RenderingUtil
{
    IFRIT_APIDECL RhiTextureRef loadBlueNoise(Graphics::Rhi::RhiBackend* rhi)
    {
        auto path = IFRIT_RUNTIME_SHARED_ASSET_PATH "/NoiseTexture/BlueNoiseRGBA.png";
        i32  width, height, channels;
        auto data = stbi_load(path, &width, &height, &channels, 4);
        if (data == nullptr)
        {
            iError("Failed to load blue noise texture");
            return nullptr;
        }
        auto tex = rhi->CreateTexture2D("Noise_Tex2D", width, height,
            Graphics::Rhi::RhiImageFormat::RhiImgFmt_R8G8B8A8_UNORM,
            Graphics::Rhi::RhiImageUsage::RhiImgUsage_ShaderRead | Graphics::Rhi::RhiImageUsage::RhiImgUsage_CopyDst,
            false);
        auto buf = rhi->CreateBuffer(
            "Noise_Buffer", width * height * 4, Graphics::Rhi::RhiBufferUsage::RhiBufferUsage_CopySrc, true, false);
        buf->MapMemory();
        buf->WriteBuffer(data, width * height * 4, 0);
        buf->FlushBuffer();
        buf->UnmapMemory();
        auto tq = rhi->GetQueue(Graphics::Rhi::RhiQueueCapability::RhiQueue_Transfer);
        tq->RunSyncCommand([&](const Graphics::Rhi::RhiCommandList* cmd) {
            Graphics::Rhi::RhiTransitionBarrier barrier;
            barrier.m_texture     = tex.get();
            barrier.m_type        = Graphics::Rhi::RhiResourceType::Texture;
            barrier.m_dstState    = Graphics::Rhi::RhiResourceState::CopyDst;
            barrier.m_srcState    = Graphics::Rhi::RhiResourceState::AutoTraced;
            barrier.m_subResource = { 0, 0, 1, 1 };

            Graphics::Rhi::RhiResourceBarrier barrier2;
            barrier2.m_type       = Graphics::Rhi::RhiBarrierType::Transition;
            barrier2.m_transition = barrier;

            cmd->AddResourceBarrier({ barrier2 });
            cmd->CopyBufferToImage(buf.get(), tex.get(), { 0, 0, 1, 1 });

            barrier2.m_transition.m_dstState = Graphics::Rhi::RhiResourceState::ShaderRead;
            cmd->AddResourceBarrier({ barrier2 });
        });
        return tex;
    }
} // namespace Ifrit::Runtime::RenderingUtil