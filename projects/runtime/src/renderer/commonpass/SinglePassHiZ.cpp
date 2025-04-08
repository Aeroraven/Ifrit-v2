#include "ifrit/core/base/IfritBase.h"

#include "ifrit/core/math/constfunc/ConstFunc.h"
#include "ifrit/runtime/renderer/commonpass/SinglePassHiZ.h"
#include "ifrit/runtime/renderer/util/RenderingUtils.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"

namespace Ifrit::Runtime
{
    using namespace RenderingUtil;
    using namespace Graphics::Rhi;
    using namespace Math;

    IF_CONSTEXPR auto kbImUsage_UAV_SRV = RhiImgUsage_UnorderedAccess | RhiImgUsage_ShaderRead;
    IF_CONSTEXPR auto kbImUsage_UAV_SRV_CopyDest =
        RhiImgUsage_UnorderedAccess | RhiImgUsage_ShaderRead | RhiImgUsage_CopyDst;
    IF_CONSTEXPR auto kbImUsage_UAV_SRV_RT_CopySrc =
        RhiImgUsage_UnorderedAccess | RhiImgUsage_ShaderRead | RhiImgUsage_RenderTarget | RhiImgUsage_CopySrc;
    IF_CONSTEXPR auto kbImUsage_UAV_SRV_RT =
        RhiImgUsage_UnorderedAccess | RhiImgUsage_ShaderRead | RhiImgUsage_RenderTarget;
    IF_CONSTEXPR auto kbImUsage_SRV_DEPTH = RhiImgUsage_ShaderRead | RhiImgUsage_Depth;
    IF_CONSTEXPR auto kbImUsage_UAV       = RhiImgUsage_UnorderedAccess;

    IF_CONSTEXPR auto kbBufUsage_SSBO_CopyDest = RhiBufferUsage_SSBO | RhiBufferUsage_CopyDst;

    IFRIT_APIDECL     SinglePassHiZPass::SinglePassHiZPass(IApplication* app)
    {
        auto rhi            = app->GetRhi();
        m_app               = app;
        m_singlePassHiZPass = CreateComputePassInternal(app, Internal::kIntShaderTable.Common.SinglePassHzbCS, 1, 6);
    }

    IFRIT_APIDECL bool SinglePassHiZPass::CheckResourceToRebuild(
        PerFrameData::SinglePassHiZData& data, u32 rtWidth, u32 rtHeight)
    {
        bool cond = (data.m_hizTexture == nullptr);

        // Make width and heights power of 2
        auto rWidth  = 1 << (int)std::ceil(std::log2(rtWidth));
        auto rHeight = 1 << (int)std::ceil(std::log2(rtHeight));
        if (!cond && (data.m_hizTexture->GetWidth() != rWidth || data.m_hizTexture->GetHeight() != rHeight))
        {
            cond = true;
        }
        if (!cond)
            return false;
        return true;
    }

    IFRIT_APIDECL void SinglePassHiZPass::PrepareHiZResources(
        PerFrameData::SinglePassHiZData& data, GPUTexture* depthTexture, GPUSampler* sampler, u32 rtWidth, u32 rtHeight)
    {
        auto rhi          = m_app->GetRhi();
        auto maxMip       = int(std::floor(std::log2(std::max(rtWidth, rtHeight))) + 1);
        data.m_hizTexture = rhi->CreateMipMapTexture(
            "SHiZ_Tex", rtWidth, rtHeight, maxMip, RhiImageFormat::RhiImgFmt_R32_SFLOAT, kbImUsage_UAV, true);

        data.m_hizRefs.resize(0);
        data.m_hizRefs.push_back(0);
        for (int i = 0; i < maxMip; i++)
        {
            auto bindlessId = rhi->RegisterUAVImage2(data.m_hizTexture.get(), { static_cast<u32>(i), 0, 1, 1 });
            data.m_hizRefs.push_back(bindlessId->GetActiveId());
        }
        data.m_hizRefBuffer =
            rhi->CreateBufferDevice("SHiZ_Ref", u32Size * data.m_hizRefs.size(), kbBufUsage_SSBO_CopyDest, true);
        data.m_hizAtomics =
            rhi->CreateBufferDevice("SHiZ_Atmoics", u32Size * data.m_hizRefs.size(), kbBufUsage_SSBO_CopyDest, true);
        auto staged = rhi->CreateStagedSingleBuffer(data.m_hizRefBuffer.get());
        auto tq     = rhi->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
        tq->RunSyncCommand([&](const GPUCmdBuffer* cmd) {
            staged->CmdCopyToDevice(cmd, data.m_hizRefs.data(), u32Size * data.m_hizRefs.size(), 0);
        });

        data.m_hizDesc = rhi->createBindlessDescriptorRef();
        data.m_hizDesc->AddStorageBuffer(data.m_hizRefBuffer.get(), 1);
        data.m_hizDesc->AddCombinedImageSampler(depthTexture, sampler, 0);
        data.m_hizDesc->AddStorageBuffer(data.m_hizAtomics.get(), 2);

        data.m_hizWidth  = rtWidth;
        data.m_hizHeight = rtHeight;
        data.m_hizIters  = maxMip;
    }

    IFRIT_APIDECL void SinglePassHiZPass::RunHiZPass(
        const PerFrameData::SinglePassHiZData& data, const GPUCmdBuffer* cmd, u32 rtWidth, u32 rtHeight, bool minMode)
    {
        struct SpHizPushConst
        {
            u32 m_hizWidth;
            u32 m_hizHeight;
            u32 m_rtWidth;
            u32 m_rtHeight;
            u32 m_hizIters;
            u32 m_minMode;
        } pc;
        pc.m_hizHeight = data.m_hizHeight;
        pc.m_hizWidth  = data.m_hizWidth;
        pc.m_rtWidth   = rtWidth;
        pc.m_rtHeight  = rtHeight;
        pc.m_hizIters  = data.m_hizIters;
        pc.m_minMode   = minMode ? 1 : 0;

        IF_CONSTEXPR static u32 cSPHiZTileSize = 64;

        m_singlePassHiZPass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            ctx->m_cmd->AttachBindlessRefCompute(m_singlePassHiZPass, 1, data.m_hizDesc);
            ctx->m_cmd->SetPushConst(m_singlePassHiZPass, 0, u32Size * 6, &pc);
            auto tgX = DivRoundUp(data.m_hizWidth, cSPHiZTileSize);
            auto tgY = DivRoundUp(data.m_hizHeight, cSPHiZTileSize);
            ctx->m_cmd->Dispatch(tgX, tgY, 1);
        });
        cmd->BeginScope("Single Pass Hierarchical Z-Buffer");
        m_singlePassHiZPass->Run(cmd, 0);
        cmd->EndScope();
    }
} // namespace Ifrit::Runtime