#pragma once
#include "ifrit/runtime/renderer/postprocessing/PostFxStockhamDFT2.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/math/constfunc/ConstFunc.h"
#include "ifrit/core/math/fastutil/FastUtil.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"

namespace Ifrit::Runtime::PostprocessPassCollection
{

    IFRIT_APIDECL PostFxStockhamDFT2::PostFxStockhamDFT2(IApplication* app)
        : PostprocessPass(app, { Internal::kIntShaderTable.Postprocess.StockhamDFT2CS, 12, 1, true })
    {
    }

    IFRIT_APIDECL void PostFxStockhamDFT2::RunCommand(const GPUCmdBuffer* cmd, u32 wgX, u32 wgY, const void* pc)
    {
        m_computePipeline->SetRecordFunction([&](const Graphics::Rhi::RhiRenderPassContext* ctx) {
            cmd->SetPushConst(pc, 0, 12 * sizeof(u32));
            ctx->m_cmd->Dispatch(wgX, wgY, 1);
        });
        m_computePipeline->Run(cmd, 0);
        cmd->GlobalMemoryBarrier();
    }

    IFRIT_APIDECL void PostFxStockhamDFT2::RenderPostFx(
        const GPUCmdBuffer* cmd, GPUBindId* srcSampId, GPUBindId* dstUAVImg, u32 width, u32 height, u32 downscale)
    {
        // Round to pow of 2
        struct PushConst
        {
            u32 logW;
            u32 logH;
            u32 rtW;
            u32 rtH;
            u32 downscaleFactor;
            u32 rawSampId;
            u32 srcImgId;
            u32 dstImgId;
            u32 orientation;
            u32 dftMode;
        } pc;

        struct PushConstBlur
        {
            u32 alignedRtW;
            u32 alignedRtH;
            u32 srcImgId;
            u32 kernelImgId;
            u32 dstImgId;
        } pcb;

        using Ifrit::Math::CountLeadingZero;
        using Ifrit::Math::IntegerLog2;
        auto p2Width  = 1 << (32 - CountLeadingZero(width / downscale - 1));
        auto p2Height = 1 << (32 - CountLeadingZero(height / downscale - 1));

        if (p2Width > 512 || p2Height > 512)
        {
            iError("Stockham DFT2: Image size too large, max 512x512");
            return;
        }

        if (m_tex1.find({ p2Width, p2Height }) == m_tex1.end())
        {
            auto rhi                      = m_app->GetRhi();
            auto tex1                     = rhi->CreateTexture2D("PostFx_DFT2_Tex", p2Width, p2Height,
                                    Graphics::Rhi::RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                                    Graphics::Rhi::RhiImageUsage::RhiImgUsage_UnorderedAccess, true);
            m_tex1[{ p2Width, p2Height }] = tex1;

            pc.logW            = IntegerLog2(p2Width);
            pc.logH            = IntegerLog2(p2Height);
            pc.rtW             = width;
            pc.rtH             = height;
            pc.downscaleFactor = downscale;
            pc.rawSampId       = srcSampId->GetActiveId();
            pc.srcImgId        = tex1->GetDescId();
            pc.dstImgId        = tex1->GetDescId();

            auto reqNumItersX = pc.logW;
            auto reqNumItersY = pc.logH;

            // DFT
            int  wgX, wgY;

            pc.dftMode = 0;
            cmd->BeginScope("Postprocess: Stockham DFT2, DFT-X");
            pc.orientation = 1;
            wgX            = Ifrit::Math::DivRoundUp(p2Width / 2, 256);
            wgY            = p2Height;
            RunCommand(cmd, wgX, wgY, &pc);
            cmd->EndScope();
            cmd->BeginScope("Postprocess: Stockham DFT2, DFT-Y");
            pc.orientation = 0;
            wgX            = Ifrit::Math::DivRoundUp(p2Height / 2, 256);
            wgY            = p2Width;
            RunCommand(cmd, wgX, wgY, &pc);
            cmd->EndScope();

            // IDFT
            pc.dftMode = 1;
            cmd->BeginScope("Postprocess: Stockham DFT2, IDFT-Y");
            pc.orientation = 0;
            wgX            = Ifrit::Math::DivRoundUp(p2Height / 2, 256);
            wgY            = p2Width;
            RunCommand(cmd, wgX, wgY, &pc);
            cmd->EndScope();
            cmd->BeginScope("Postprocess: Stockham DFT2, IDFT-X");
            pc.orientation = 1;
            wgX            = Ifrit::Math::DivRoundUp(p2Width / 2, 256);
            wgY            = p2Height;
            RunCommand(cmd, wgX, wgY, &pc);
            cmd->EndScope();
        }
    }

} // namespace Ifrit::Runtime::PostprocessPassCollection