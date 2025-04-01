#include "ifrit/vkgraphics/engine/fsr2extension/FSR2Processor.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Command.h"
#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"
#include "ifrit/vkgraphics/utility/Logger.h"
#include <vector>

#define FFX_IMPORTS
#include "ifrit.external/fsr2/ffx_fsr2.h"
#include "ifrit.external/fsr2/vk/ffx_fsr2_vk.h"
#undef FFX_IMPORTS

namespace Ifrit::Graphics::VulkanGraphics::FSR2
{

    using Ifrit::CheckedCast;

    struct FSR2Context
    {
        size_t                    scratchSize;
        Vec<char>                 scratchBuffer;
        FfxFsr2Interface          fsr2Interface;
        FfxFsr2ContextDescription initContext{};
        FfxFsr2Context            fsr2ctx;
        bool                      fsr2Initialized = false;
    };

    static void onFSR2Msg(FfxFsr2MsgType type, const wchar_t* message)
    {
        std::string msg(message, message + wcslen(message));
        if (type == FFX_FSR2_MESSAGE_TYPE_ERROR)
        {
            iError("FSR2_API_DEBUG_ERROR: {}", msg);
            std::abort();
        }
        else if (type == FFX_FSR2_MESSAGE_TYPE_WARNING)
        {
            iWarn("FSR2_API_DEBUG_WARNING: {}", msg);
        }
        else
        {
            iInfo("FSR2_API_DEBUG_INFO: {}", msg);
        }
    }

    IFRIT_APIDECL FSR2Processor::FSR2Processor(EngineContext* ctx)
    {
        m_engineContext        = ctx;
        m_context              = new FSR2Context();
        m_context->scratchSize = ffxFsr2GetScratchMemorySizeVK(m_engineContext->GetPhysicalDevice());
        m_context->scratchBuffer.resize(m_context->scratchSize);
        auto errorCode = ffxFsr2GetInterfaceVK(&m_context->initContext.callbacks, m_context->scratchBuffer.data(),
            m_context->scratchSize, m_engineContext->GetPhysicalDevice(), vkGetDeviceProcAddr);
        if (errorCode != FFX_OK)
        {
            iError("Failed to get FSR2 interface, error code: {}", int(errorCode));
            std::abort();
        }
    }

    IFRIT_APIDECL void FSR2Processor::Init(const Rhi::FSR2::RhiFSR2InitialzeArgs& args)
    {
        m_context->initContext.device               = ffxGetDeviceVK(m_engineContext->GetDevice());
        m_context->initContext.maxRenderSize.width  = args.maxRenderWidth;
        m_context->initContext.maxRenderSize.height = args.maxRenderHeight;
        m_context->initContext.displaySize.width    = args.displayWidth;
        m_context->initContext.displaySize.height   = args.displayHeight;
        m_context->initContext.flags                = FFX_FSR2_ENABLE_AUTO_EXPOSURE | FFX_FSR2_ENABLE_DEBUG_CHECKING;
        // hdr
        m_context->initContext.flags |= FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE;
        m_context->initContext.fpMessage = onFSR2Msg;

        auto errorCode = ffxFsr2ContextCreate(&m_context->fsr2ctx, &m_context->initContext);
        if (errorCode != FFX_OK)
        {
            iError("Failed to create FSR2 context, error code: {}", int(errorCode));
            std::abort();
        }
        m_context->fsr2Initialized = true;
    }

    IFRIT_APIDECL void FSR2Processor::Dispatch(
        const Rhi::RhiCommandList* cmd, const Rhi::FSR2::RhiFSR2DispatchArgs& args)
    {
        FfxFsr2DispatchDescription dispatchParams = {};
        auto                       cmdVk          = CheckedCast<CommandBuffer>(cmd)->GetCommandBuffer();
        auto                       colorImgVk     = CheckedCast<SingleDeviceImage>(args.color);
        auto                       depthImgVk     = CheckedCast<SingleDeviceImage>(args.depth);
        auto                       motionImgVk    = CheckedCast<SingleDeviceImage>(args.motion);
        auto                       outputImgVk    = CheckedCast<SingleDeviceImage>(args.output);

        SingleDeviceImage*         exposureImgVk         = nullptr;
        SingleDeviceImage*         reactiveMaskImgVk     = nullptr;
        SingleDeviceImage*         transparencyMaskImgVk = nullptr;
        if (args.exposure)
        {
            exposureImgVk = CheckedCast<SingleDeviceImage>(args.exposure);
        }
        if (args.reactiveMask)
        {
            reactiveMaskImgVk = CheckedCast<SingleDeviceImage>(args.reactiveMask);
        }
        if (args.transparencyMask)
        {
            transparencyMaskImgVk = CheckedCast<SingleDeviceImage>(args.transparencyMask);
        }

        dispatchParams.commandList = ffxGetCommandListVK(cmdVk);
        dispatchParams.color =
            ffxGetTextureResourceVK(&m_context->fsr2ctx, colorImgVk->GetImage(), colorImgVk->GetImageView(),
                colorImgVk->GetWidth(), colorImgVk->GetHeight(), colorImgVk->GetFormat(), L"FSR2_InputColor");
        dispatchParams.depth =
            ffxGetTextureResourceVK(&m_context->fsr2ctx, depthImgVk->GetImage(), depthImgVk->GetImageView(),
                depthImgVk->GetWidth(), depthImgVk->GetHeight(), depthImgVk->GetFormat(), L"FSR2_InputDepth");
        dispatchParams.motionVectors = ffxGetTextureResourceVK(&m_context->fsr2ctx, motionImgVk->GetImage(),
            motionImgVk->GetImageView(), motionImgVk->GetWidth(), motionImgVk->GetHeight(), motionImgVk->GetFormat(),
            L"FSR2_InputMotionVectors");
        dispatchParams.output =
            ffxGetTextureResourceVK(&m_context->fsr2ctx, outputImgVk->GetImage(), outputImgVk->GetImageView(),
                outputImgVk->GetWidth(), outputImgVk->GetHeight(), outputImgVk->GetFormat(), L"FSR2_OutputColor");
        if (exposureImgVk)
        {
            dispatchParams.exposure = ffxGetTextureResourceVK(&m_context->fsr2ctx, exposureImgVk->GetImage(),
                exposureImgVk->GetImageView(), exposureImgVk->GetWidth(), exposureImgVk->GetHeight(),
                exposureImgVk->GetFormat(), L"FSR2_InputExposure");
        }
        else
        {
            dispatchParams.exposure = ffxGetTextureResourceVK(
                &m_context->fsr2ctx, nullptr, nullptr, 1, 1, VK_FORMAT_UNDEFINED, L"FSR2_EmptyInputExposure");
        }
        if (reactiveMaskImgVk)
        {
            dispatchParams.reactive = ffxGetTextureResourceVK(&m_context->fsr2ctx, reactiveMaskImgVk->GetImage(),
                reactiveMaskImgVk->GetImageView(), reactiveMaskImgVk->GetWidth(), reactiveMaskImgVk->GetHeight(),
                reactiveMaskImgVk->GetFormat(), L"FSR2_InputReactiveMap");
        }
        else
        {
            dispatchParams.reactive = ffxGetTextureResourceVK(
                &m_context->fsr2ctx, nullptr, nullptr, 1, 1, VK_FORMAT_UNDEFINED, L"FSR2_EmptyInputReactiveMap");
        }
        if (transparencyMaskImgVk)
        {
            dispatchParams.transparencyAndComposition = ffxGetTextureResourceVK(&m_context->fsr2ctx,
                transparencyMaskImgVk->GetImage(), transparencyMaskImgVk->GetImageView(),
                transparencyMaskImgVk->GetWidth(), transparencyMaskImgVk->GetHeight(),
                transparencyMaskImgVk->GetFormat(), L"FSR2_TransparencyAndCompositionMap");
        }
        else
        {
            dispatchParams.transparencyAndComposition = ffxGetTextureResourceVK(&m_context->fsr2ctx, nullptr, nullptr,
                1, 1, VK_FORMAT_UNDEFINED, L"FSR2_EmptyTransparencyAndCompositionMap");
        }
        dispatchParams.renderSize.width    = m_context->initContext.maxRenderSize.width;
        dispatchParams.renderSize.height   = m_context->initContext.maxRenderSize.height;
        dispatchParams.jitterOffset.x      = args.jitterX;
        dispatchParams.jitterOffset.y      = args.jitterY;
        dispatchParams.frameTimeDelta      = args.deltaTime * 5.0f;
        dispatchParams.motionVectorScale.x = m_context->initContext.maxRenderSize.width;
        dispatchParams.motionVectorScale.y = m_context->initContext.maxRenderSize.height;

        dispatchParams.cameraNear              = args.camNear;
        dispatchParams.cameraFar               = args.camFar;
        dispatchParams.cameraFovAngleVertical  = args.camFovY;
        dispatchParams.viewSpaceToMetersFactor = 1.0f;

        dispatchParams.reset            = args.reset;
        dispatchParams.enableSharpening = true;
        dispatchParams.sharpness        = 0.7f;
        dispatchParams.preExposure      = 1.0f;

        // First transition output to shader read. This should be untraced
        VkImageMemoryBarrier barrier{};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image                           = outputImgVk->GetImage();
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.layerCount     = 1;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.srcAccessMask                   = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout                       = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout                       = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cmdVk, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
            nullptr, 0, nullptr, 1, &barrier);

        auto errorCode = ffxFsr2ContextDispatch(&m_context->fsr2ctx, &dispatchParams);
        if (errorCode != FFX_OK)
        {
            iError("Failed to dispatch FSR2, error code: {}", int(errorCode));
            std::abort();
        }
    }

    IFRIT_APIDECL void FSR2Processor::GetJitters(
        float* jitterX, float* jitterY, uint32_t frameIdx, uint32_t rtWidth, uint32_t dispWidth)
    {
        auto phase = ffxFsr2GetJitterPhaseCount(rtWidth, dispWidth);
        ffxFsr2GetJitterOffset(jitterX, jitterY, frameIdx, phase);
    }

    IFRIT_APIDECL FSR2Processor::~FSR2Processor()
    {
        if (m_context->fsr2Initialized)
        {
            auto errorCode = ffxFsr2ContextDestroy(&m_context->fsr2ctx);
            if (errorCode != FFX_OK)
            {
                iError("Failed to destroy FSR2 context, error code: {}", int(errorCode));
            }
        }
        delete m_context;
    }

} // namespace Ifrit::Graphics::VulkanGraphics::FSR2