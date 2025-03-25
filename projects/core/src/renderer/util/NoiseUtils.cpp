#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/logging/Logging.h"

#include "ifrit/core/renderer/util/NoiseUtils.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#undef STB_IMAGE_IMPLEMENTATION

using namespace Ifrit::GraphicsBackend::Rhi;

namespace Ifrit::Core::RenderingUtil {
IFRIT_APIDECL RhiTextureRef loadBlueNoise(GraphicsBackend::Rhi::RhiBackend *rhi) {
  auto path = IFRIT_CORELIB_SHARED_ASSET_PATH "/NoiseTexture/BlueNoiseRGBA.png";
  i32 width, height, channels;
  auto data = stbi_load(path, &width, &height, &channels, 4);
  if (data == nullptr) {
    iError("Failed to load blue noise texture");
    return nullptr;
  }
  auto tex = rhi->createTexture2D("Noise_Tex2D", width, height,
                                  GraphicsBackend::Rhi::RhiImageFormat::RHI_FORMAT_R8G8B8A8_UNORM,
                                  GraphicsBackend::Rhi::RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT |
                                      GraphicsBackend::Rhi::RhiImageUsage::RHI_IMAGE_USAGE_TRANSFER_DST_BIT);
  auto buf = rhi->createBuffer("Noise_Buffer", width * height * 4,
                               GraphicsBackend::Rhi::RhiBufferUsage::RhiBufferUsage_CopySrc, true);
  buf->map();
  buf->writeBuffer(data, width * height * 4, 0);
  buf->flush();
  buf->unmap();
  auto tq = rhi->getQueue(GraphicsBackend::Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
  tq->runSyncCommand([&](const GraphicsBackend::Rhi::RhiCommandList *cmd) {
    GraphicsBackend::Rhi::RhiTransitionBarrier barrier;
    barrier.m_texture = tex.get();
    barrier.m_type = GraphicsBackend::Rhi::RhiResourceType::Texture;
    barrier.m_dstState = GraphicsBackend::Rhi::RhiResourceState::CopyDst;
    barrier.m_srcState = GraphicsBackend::Rhi::RhiResourceState::AutoTraced;
    barrier.m_subResource = {0, 0, 1, 1};

    GraphicsBackend::Rhi::RhiResourceBarrier barrier2;
    barrier2.m_type = GraphicsBackend::Rhi::RhiBarrierType::Transition;
    barrier2.m_transition = barrier;

    cmd->resourceBarrier({barrier2});
    cmd->copyBufferToImage(buf.get(), tex.get(), {0, 0, 1, 1});

    barrier2.m_transition.m_dstState = GraphicsBackend::Rhi::RhiResourceState::ShaderRead;
    cmd->resourceBarrier({barrier2});
  });
  return tex;
}
} // namespace Ifrit::Core::RenderingUtil