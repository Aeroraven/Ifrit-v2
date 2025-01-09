#include "ifrit/vkgraphics/engine/fsr2extension/FSR2Processor.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Command.h"
#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"
#include "ifrit/vkgraphics/utility/Logger.h"
#include <vector>

#define FFX_IMPORTS
#include "ifrit.external/fsr2/ffx_fsr2.h"
#include "ifrit.external/fsr2/vk/ffx_fsr2_vk.h"
#undef FFX_IMPORTS

namespace Ifrit::GraphicsBackend::VulkanGraphics::FSR2 {

using Ifrit::Common::Utility::checked_cast;

struct FSR2Context {
  size_t scratchSize;
  std::vector<char> scratchBuffer;
  FfxFsr2Interface fsr2Interface;
  FfxFsr2ContextDescription initContext{};
  FfxFsr2Context fsr2ctx;
  bool fsr2Initialized = false;
};

static void onFSR2Msg(FfxFsr2MsgType type, const wchar_t *message) {
  std::string msg(message, message + wcslen(message));
  if (type == FFX_FSR2_MESSAGE_TYPE_ERROR) {
    iError("FSR2_API_DEBUG_ERROR: {}", msg);
    std::abort();
  } else if (type == FFX_FSR2_MESSAGE_TYPE_WARNING) {
    iWarn("FSR2_API_DEBUG_WARNING: {}", msg);
  } else {
    iInfo("FSR2_API_DEBUG_INFO: {}", msg);
  }
}

IFRIT_APIDECL FSR2Processor::FSR2Processor(EngineContext *ctx) {
  m_engineContext = ctx;
  m_context = new FSR2Context();
  m_context->scratchSize =
      ffxFsr2GetScratchMemorySizeVK(m_engineContext->getPhysicalDevice());
  m_context->scratchBuffer.resize(m_context->scratchSize);
  auto errorCode = ffxFsr2GetInterfaceVK(
      &m_context->initContext.callbacks, m_context->scratchBuffer.data(),
      m_context->scratchSize, m_engineContext->getPhysicalDevice(),
      vkGetDeviceProcAddr);
  if (errorCode != FFX_OK) {
    iError("Failed to get FSR2 interface, error code: {}", int(errorCode));
    std::abort();
  }
}

IFRIT_APIDECL void
FSR2Processor::init(const Rhi::FSR2::RhiFSR2InitialzeArgs &args) {
  m_context->initContext.device = ffxGetDeviceVK(m_engineContext->getDevice());
  m_context->initContext.maxRenderSize.width = args.maxRenderWidth;
  m_context->initContext.maxRenderSize.height = args.maxRenderHeight;
  m_context->initContext.displaySize.width = args.displayWidth;
  m_context->initContext.displaySize.height = args.displayHeight;
  m_context->initContext.flags =
      FFX_FSR2_ENABLE_AUTO_EXPOSURE | FFX_FSR2_ENABLE_DEBUG_CHECKING;
  // hdr
  m_context->initContext.flags |= FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE;
  m_context->initContext.fpMessage = onFSR2Msg;

  auto errorCode =
      ffxFsr2ContextCreate(&m_context->fsr2ctx, &m_context->initContext);
  if (errorCode != FFX_OK) {
    iError("Failed to create FSR2 context, error code: {}", int(errorCode));
    std::abort();
  }
  m_context->fsr2Initialized = true;
}

IFRIT_APIDECL void
FSR2Processor::dispatch(const Rhi::RhiCommandBuffer *cmd,
                        const Rhi::FSR2::RhiFSR2DispatchArgs &args) {
  FfxFsr2DispatchDescription dispatchParams = {};
  auto cmdVk = checked_cast<CommandBuffer>(cmd)->getCommandBuffer();
  auto colorImgVk = checked_cast<SingleDeviceImage>(args.color);
  auto depthImgVk = checked_cast<SingleDeviceImage>(args.depth);
  auto motionImgVk = checked_cast<SingleDeviceImage>(args.motion);
  auto outputImgVk = checked_cast<SingleDeviceImage>(args.output);

  SingleDeviceImage *exposureImgVk = nullptr;
  SingleDeviceImage *reactiveMaskImgVk = nullptr;
  SingleDeviceImage *transparencyMaskImgVk = nullptr;
  if (args.exposure) {
    exposureImgVk = checked_cast<SingleDeviceImage>(args.exposure);
  }
  if (args.reactiveMask) {
    reactiveMaskImgVk = checked_cast<SingleDeviceImage>(args.reactiveMask);
  }
  if (args.transparencyMask) {
    transparencyMaskImgVk =
        checked_cast<SingleDeviceImage>(args.transparencyMask);
  }

  dispatchParams.commandList = ffxGetCommandListVK(cmdVk);
  dispatchParams.color = ffxGetTextureResourceVK(
      &m_context->fsr2ctx, colorImgVk->getImage(), colorImgVk->getImageView(),
      colorImgVk->getWidth(), colorImgVk->getHeight(), colorImgVk->getFormat(),
      L"FSR2_InputColor");
  dispatchParams.depth = ffxGetTextureResourceVK(
      &m_context->fsr2ctx, depthImgVk->getImage(), depthImgVk->getImageView(),
      depthImgVk->getWidth(), depthImgVk->getHeight(), depthImgVk->getFormat(),
      L"FSR2_InputDepth");
  dispatchParams.motionVectors = ffxGetTextureResourceVK(
      &m_context->fsr2ctx, motionImgVk->getImage(), motionImgVk->getImageView(),
      motionImgVk->getWidth(), motionImgVk->getHeight(),
      motionImgVk->getFormat(), L"FSR2_InputMotionVectors");
  dispatchParams.output = ffxGetTextureResourceVK(
      &m_context->fsr2ctx, outputImgVk->getImage(), outputImgVk->getImageView(),
      outputImgVk->getWidth(), outputImgVk->getHeight(),
      outputImgVk->getFormat(), L"FSR2_OutputColor");
  if (exposureImgVk) {
    dispatchParams.exposure = ffxGetTextureResourceVK(
        &m_context->fsr2ctx, exposureImgVk->getImage(),
        exposureImgVk->getImageView(), exposureImgVk->getWidth(),
        exposureImgVk->getHeight(), exposureImgVk->getFormat(),
        L"FSR2_InputExposure");
  } else {
    dispatchParams.exposure = ffxGetTextureResourceVK(
        &m_context->fsr2ctx, nullptr, nullptr, 1, 1, VK_FORMAT_UNDEFINED,
        L"FSR2_EmptyInputExposure");
  }
  if (reactiveMaskImgVk) {
    dispatchParams.reactive = ffxGetTextureResourceVK(
        &m_context->fsr2ctx, reactiveMaskImgVk->getImage(),
        reactiveMaskImgVk->getImageView(), reactiveMaskImgVk->getWidth(),
        reactiveMaskImgVk->getHeight(), reactiveMaskImgVk->getFormat(),
        L"FSR2_InputReactiveMap");
  } else {
    dispatchParams.reactive = ffxGetTextureResourceVK(
        &m_context->fsr2ctx, nullptr, nullptr, 1, 1, VK_FORMAT_UNDEFINED,
        L"FSR2_EmptyInputReactiveMap");
  }
  if (transparencyMaskImgVk) {
    dispatchParams.transparencyAndComposition = ffxGetTextureResourceVK(
        &m_context->fsr2ctx, transparencyMaskImgVk->getImage(),
        transparencyMaskImgVk->getImageView(),
        transparencyMaskImgVk->getWidth(), transparencyMaskImgVk->getHeight(),
        transparencyMaskImgVk->getFormat(),
        L"FSR2_TransparencyAndCompositionMap");
  } else {
    dispatchParams.transparencyAndComposition = ffxGetTextureResourceVK(
        &m_context->fsr2ctx, nullptr, nullptr, 1, 1, VK_FORMAT_UNDEFINED,
        L"FSR2_EmptyTransparencyAndCompositionMap");
  }
  dispatchParams.renderSize.width = m_context->initContext.maxRenderSize.width;
  dispatchParams.renderSize.height =
      m_context->initContext.maxRenderSize.height;
  dispatchParams.jitterOffset.x = args.jitterX;
  dispatchParams.jitterOffset.y = args.jitterY;
  dispatchParams.frameTimeDelta = args.deltaTime * 5.0f;
  dispatchParams.motionVectorScale.x =
      m_context->initContext.maxRenderSize.width;
  dispatchParams.motionVectorScale.y =
      m_context->initContext.maxRenderSize.height;

  dispatchParams.cameraNear = args.camNear;
  dispatchParams.cameraFar = args.camFar;
  dispatchParams.cameraFovAngleVertical = args.camFovY;
  dispatchParams.viewSpaceToMetersFactor = 1.0f;

  dispatchParams.reset = false;
  dispatchParams.enableSharpening = false;
  dispatchParams.sharpness = 0.0f;
  dispatchParams.preExposure = 1.0f;

  auto errorCode = ffxFsr2ContextDispatch(&m_context->fsr2ctx, &dispatchParams);
  if (errorCode != FFX_OK) {
    iError("Failed to dispatch FSR2, error code: {}", int(errorCode));
    std::abort();
  }
}

IFRIT_APIDECL void FSR2Processor::getJitters(float *jitterX, float *jitterY,
                                             uint32_t frameIdx,
                                             uint32_t rtWidth,
                                             uint32_t dispWidth) {
  auto phase = ffxFsr2GetJitterPhaseCount(rtWidth, dispWidth);
  ffxFsr2GetJitterOffset(jitterX, jitterY, frameIdx, phase);
}

IFRIT_APIDECL FSR2Processor::~FSR2Processor() {
  if (m_context->fsr2Initialized) {
    auto errorCode = ffxFsr2ContextDestroy(&m_context->fsr2ctx);
    if (errorCode != FFX_OK) {
      iError("Failed to destroy FSR2 context, error code: {}", int(errorCode));
    }
  }
  delete m_context;
}

} // namespace Ifrit::GraphicsBackend::VulkanGraphics::FSR2