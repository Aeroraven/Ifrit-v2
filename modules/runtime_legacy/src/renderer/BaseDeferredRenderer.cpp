#include "ifrit/runtime/renderer/BaseDeferredRenderer.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"
#include "ifrit/runtime/rendercore/framegraph/FrameGraphUtils.h"
using namespace Ifrit::RHI;

namespace Ifrit::Runtime
{
    struct BaseDeferredRendererResources
    {
        Owner<FrameGraphCompiler>   m_FgCompiler;
        Owner<FrameGraphExecutor>   m_FgExecutor;
        Ref<FrameGraphResourcePool> m_ResourcePool = nullptr;
    };

    IFRIT_APIDECL BaseDeferredRenderer::BaseDeferredRenderer(IApplication* app) : RendererBase(app)
    {
        m_Resources = new BaseDeferredRendererResources();
        InitRenderer();
    }

    IFRIT_APIDECL BaseDeferredRenderer::~BaseDeferredRenderer()
    {
        if (m_Resources)
        {
            delete m_Resources;
            m_Resources = nullptr;
        }
    }

    IFRIT_APIDECL void BaseDeferredRenderer::InitRenderer()
    {
        PrepareImmutableResources();
        m_Resources->m_FgExecutor   = MakeOwner<FrameGraphExecutor>(m_app->GetRhi());
        m_Resources->m_FgCompiler   = MakeOwner<FrameGraphCompiler>();
        m_Resources->m_ResourcePool = MakeRef<FrameGraphResourcePool>(m_app->GetRhi());
    }

    IFRIT_APIDECL void BaseDeferredRenderer::SetupAndRunFrameGraph(
        Scene* scene, PerFrameData& perframe, RenderTargets* renderTargets, const GPUCmdBuffer* cmd)
    {
        cmd->BeginScope("BaseDeferred: Execute Render Graph");
        bool bHasValidRenderData = !perframe.mSkipRendering;
        if (bHasValidRenderData)
        {

            FrameGraphBuilder builder(m_app->GetShaderRegistry(), m_app->GetRhi(), m_Resources->m_ResourcePool.get());

            auto&             rdgMainView =
                builder.ImportBuffer("RDG.Imported.MainView", perframe.m_views[0].m_viewBuffer->GetActiveBuffer());
            auto& rdgDepthBuffer = builder.ImportTexture(
                "RDG.Imported.DepthBuffer", renderTargets->GetDepthStencilAttachment()->GetTexture());
            auto& rdgColorBuffer = builder.ImportTexture(
                "RDG.Imported.ColorBuffer", renderTargets->GetColorAttachment(0)->GetRenderTarget());

            auto rtWidth  = renderTargets->GetRenderArea().width;
            auto rtHeight = renderTargets->GetRenderArea().height;

            auto rdgGbufferAlbedo    = &builder.DeclareTexture("RDG.GBuffer.Albedo",
                   FrameGraphTextureDesc(rtWidth, rtHeight, 1, RHI::RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                       RHI::RhiImageUsage::RhiImgUsage_ShaderRead | RHI::RhiImageUsage::RhiImgUsage_RenderTarget));
            auto rdgGbufferNormal    = &builder.DeclareTexture("RDG.GBuffer.Normal",
                   FrameGraphTextureDesc(rtWidth, rtHeight, 1, RHI::RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                       RHI::RhiImageUsage::RhiImgUsage_ShaderRead | RHI::RhiImageUsage::RhiImgUsage_RenderTarget));
            auto rdgGbufferWorldPos  = &builder.DeclareTexture("RDG.GBuffer.WorldPos",
                 FrameGraphTextureDesc(rtWidth, rtHeight, 1, RHI::RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                     RHI::RhiImageUsage::RhiImgUsage_ShaderRead | RHI::RhiImageUsage::RhiImgUsage_RenderTarget));
            auto rdgGbufferGelMask   = &builder.DeclareTexture("RDG.GBuffer.GelMask",
                  FrameGraphTextureDesc(rtWidth, rtHeight, 1, RHI::RhiImageFormat::RhiImgFmt_R32_SFLOAT,
                      RHI::RhiImageUsage::RhiImgUsage_ShaderRead | RHI::RhiImageUsage::RhiImgUsage_RenderTarget));
            auto rdgBackwardDepth    = &builder.DeclareTexture("RDG.Backward.Depth",
                   FrameGraphTextureDesc(rtWidth, rtHeight, 1, RHI::RhiImageFormat::RhiImgFmt_D32_SFLOAT,
                       RHI::RhiImageUsage::RhiImgUsage_ShaderRead | RHI::RhiImageUsage::RhiImgUsage_Depth));
            auto rdgDepthTemp        = &builder.DeclareTexture("RDG.Temp.Depth",
                       FrameGraphTextureDesc(rtWidth, rtHeight, 1, RHI::RhiImageFormat::RhiImgFmt_D32_SFLOAT,
                           RHI::RhiImageUsage::RhiImgUsage_ShaderRead | RHI::RhiImageUsage::RhiImgUsage_Depth));
            auto rdgColorBeforeMerge = &builder.DeclareTexture("RDG.Backward.ColorBeforeMerge",
                FrameGraphTextureDesc(rtWidth, rtHeight, 1, RHI::RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT,
                    RHI::RhiImageUsage::RhiImgUsage_ShaderRead | RHI::RhiImageUsage::RhiImgUsage_RenderTarget));

            auto passExecFunc = [this, &perframe, rdgMainView, rtWidth, rtHeight](const FrameGraphPassContext& ctx,
                                    bool forwardFacing, bool renderGelatin, bool renderStatic) {
                auto        cmd = ctx.m_CmdList;

                RhiViewport viewport;
                viewport.x        = 0.0f * rtWidth;
                viewport.y        = 0.0f * rtHeight;
                viewport.width    = 1.0f * rtWidth;
                viewport.height   = 1.0f * rtHeight;
                viewport.minDepth = 0.0f;
                viewport.maxDepth = 1.0f;
                cmd->SetViewports({ viewport });

                RhiScissor scissor;
                scissor.x      = 0;
                scissor.y      = 0;
                scissor.width  = rtWidth;
                scissor.height = rtHeight;
                cmd->SetScissors({ scissor });

                for (auto& shaderEffects : perframe.m_shaderEffectData)
                {
                    for (u32 i = 0; i < SizeCast<u32>(shaderEffects.m_transforms.size()); i++)
                    {
                        struct PushConst
                        {
                            u32 m_ObjectBufferId;
                            u32 m_PerFrameId;
                            u32 m_InBatchOffset;
                            u32 m_MaskValue;
                        } pc;

                        pc.m_ObjectBufferId = shaderEffects.m_batchedObjectData->GetActiveBuffer()->GetDescId();
                        pc.m_PerFrameId     = ctx.m_FgDesc->GetSRV(rdgMainView);
                        pc.m_InBatchOffset  = i;
                        cmd->AttachIndexBuffer(shaderEffects.m_meshes[i]->m_resource.indexBuffer.get());
                        cmd->SetCullMode(RhiCullMode::None);
                        if (forwardFacing)
                        {
                            cmd->SetDepthFunc(RhiDepthFunc::Less);
                        }
                        else
                        {
                            cmd->SetDepthFunc(RhiDepthFunc::Greater);
                        }

                        auto meshData = shaderEffects.m_meshes[i]->LoadMeshUnsafe();

                        if (meshData->m_GenerationType == MeshGeneratorType::Static)
                        {
                            auto indexCount =
                                SizeCast<u32>(shaderEffects.m_meshes[i]->LoadMeshUnsafe()->m_indices.size());

                            if (indexCount != 0 && renderStatic)
                            {
                                pc.m_MaskValue = 0;
                                cmd->SetPushConst(&pc, 0, sizeof(PushConst));
                                cmd->DrawIndexed(indexCount, 1, 0, 0, 0);
                            }
                        }
                        else if (meshData->m_GenerationType == MeshGeneratorType::Procedual)
                        {
                            if (renderGelatin)
                            {
                                pc.m_MaskValue = 1;
                                cmd->SetPushConst(&pc, 0, sizeof(PushConst));
                                cmd->DrawIndexedIndirect(
                                    shaderEffects.m_meshes[i]->m_resource.procIndirectDrawBuffer.get(), 0);
                            }
                        }
                    }
                }
            };

            // Forward pass
            auto& pass = builder.AddGraphicsPass("GBuffer Pass",
                ShaderVariantDesc(Internal::kIntShaderTable.BaseDeferred.DeferredDefaultVS, {}),
                ShaderVariantDesc(Internal::kIntShaderTable.BaseDeferred.DeferredDefaultFS, {}), 4);
            pass.SetExecutionFunction(
                [passExecFunc](const FrameGraphPassContext& ctx) { passExecFunc(ctx, true, false, true); });
            pass.AddDepthTarget(rdgDepthBuffer, RhiRenderTargetLoadOp::Clear, 1.0f)
                .AddRenderTarget(*rdgGbufferAlbedo, RhiRenderTargetLoadOp::Clear, (Vector4f(0.0f, 0.0f, 0.0f, 1.0f)))
                .AddRenderTarget(*rdgGbufferNormal, RhiRenderTargetLoadOp::Clear, (Vector4f(0.0f, 0.0f, 0.0f, 1.0f)))
                .AddRenderTarget(*rdgGbufferWorldPos, RhiRenderTargetLoadOp::Clear, (Vector4f(0.0f, 0.0f, 0.0f, 1.0f)))
                .AddRenderTarget(*rdgGbufferGelMask, RhiRenderTargetLoadOp::Clear, (Vector4f(0.0f, 0.0f, 0.0f, 0.0f)));
            // End of forward pass

            // Backward pass
            auto& forwardPass = builder.AddGraphicsPass("Forward Pass",
                ShaderVariantDesc(Internal::kIntShaderTable.BaseDeferred.DeferredDefaultVS, {}),
                ShaderVariantDesc("", {}), 4);
            forwardPass.SetExecutionFunction(
                [passExecFunc](const FrameGraphPassContext& ctx) { passExecFunc(ctx, true, true, false); });
            forwardPass.AddDepthTarget(rdgDepthBuffer, RhiRenderTargetLoadOp::Clear, 1.0f);

            auto& backwardPass = builder.AddGraphicsPass("Backward Pass",
                ShaderVariantDesc(Internal::kIntShaderTable.BaseDeferred.DeferredDefaultVS, {}),
                ShaderVariantDesc("", {}), 4);
            backwardPass.SetExecutionFunction(
                [passExecFunc](const FrameGraphPassContext& ctx) { passExecFunc(ctx, false, true, false); });
            backwardPass.AddDepthTarget(*rdgBackwardDepth, RhiRenderTargetLoadOp::Clear, 0.0f);
            // End of backward pass

            // Shading pass
            struct PushConst_Shading
            {
                RHI::RhiSRVDesc m_Color;
                RHI::RhiSRVDesc m_Normal;
                RHI::RhiSRVDesc m_WorldPos;
                RHI::RhiSRVDesc mMask;
            } pc{};

            FrameGraphUtils::AddPostProcessPass<PushConst_Shading>(builder, "Shading Pass",
                ShaderVariantDesc(Internal::kIntShaderTable.BaseDeferred.DeferredShadingFS, {}), pc,
                [this, rdgGbufferAlbedo, rdgGbufferNormal, rdgGbufferWorldPos, rdgGbufferGelMask](
                    PushConst_Shading pc, const FrameGraphPassContext& ctx) {
                    pc.m_Color    = ctx.m_FgDesc->GetSRV(*rdgGbufferAlbedo);
                    pc.m_Normal   = ctx.m_FgDesc->GetSRV(*rdgGbufferNormal);
                    pc.m_WorldPos = ctx.m_FgDesc->GetSRV(*rdgGbufferWorldPos);
                    pc.mMask      = ctx.m_FgDesc->GetSRV(*rdgGbufferGelMask);
                    FrameGraphUtils::SetRootConstant(pc, ctx);
                })
                .AddRenderTarget(*rdgColorBeforeMerge, RhiRenderTargetLoadOp::Clear, Vector4f(0.0f, 0.0f, 0.0f, 1.0f))
                .AddReadResource(*rdgGbufferAlbedo)
                .AddReadResource(*rdgGbufferNormal)
                .AddReadResource(*rdgGbufferGelMask)
                .AddReadResource(*rdgGbufferWorldPos);
            // End of shading pass

            // Forward pass
            auto& gPass = builder.AddGraphicsPass("Gelatin GBuffer Pass",
                ShaderVariantDesc(Internal::kIntShaderTable.BaseDeferred.DeferredDefaultVS, {}),
                ShaderVariantDesc(Internal::kIntShaderTable.BaseDeferred.DeferredDefaultFS, {}), 4);
            gPass.SetExecutionFunction(
                [passExecFunc](const FrameGraphPassContext& ctx) { passExecFunc(ctx, true, true, false); });
            gPass.AddDepthTarget(*rdgDepthTemp, RhiRenderTargetLoadOp::Clear, 1.0f)
                .AddRenderTarget(*rdgGbufferAlbedo, RhiRenderTargetLoadOp::Clear, (Vector4f(0.0f, 0.0f, 0.0f, 1.0f)))
                .AddRenderTarget(*rdgGbufferNormal, RhiRenderTargetLoadOp::Clear, (Vector4f(0.0f, 0.0f, 0.0f, 1.0f)))
                .AddRenderTarget(*rdgGbufferWorldPos, RhiRenderTargetLoadOp::Clear, (Vector4f(0.0f, 0.0f, 0.0f, 1.0f)))
                .AddRenderTarget(*rdgGbufferGelMask, RhiRenderTargetLoadOp::Clear, (Vector4f(0.0f, 0.0f, 0.0f, 0.0f)));
            // End of forward pass

            // Test Gelatin pass
            struct PushConst_Gelatin
            {
                Vector4f        mCameraPos;
                Vector4f        mLightDir;
                f32             mCameraNear;
                f32             mCameraFar;
                RHI::RhiSRVDesc mDepthFront; // clip-space depth [0,1]
                RHI::RhiSRVDesc mDepthBack;  // clip-space depth [0,1]
                RHI::RhiSRVDesc mMask;
                RHI::RhiSRVDesc mCurColor;
                RHI::RhiSRVDesc mCurNormal;
                RHI::RhiSRVDesc mCurWorldPos;
            } pcGelatin{};
            pcGelatin.mCameraPos  = perframe.m_views[0].m_viewData.m_cameraPosition;
            pcGelatin.mLightDir   = Vector4f(-1.0f, -1.0f, 0.0f, 0.0f);
            pcGelatin.mCameraNear = perframe.m_views[0].m_viewData.m_cameraNear;
            pcGelatin.mCameraFar  = perframe.m_views[0].m_viewData.m_cameraFar;

            FrameGraphUtils::AddPostProcessPass<PushConst_Gelatin>(builder, "Gelatin Test Pass",
                ShaderVariantDesc(Internal::kIntShaderTable.Experimental.GelatinTestFS, {}), pcGelatin,
                [this, rdgDepthBuffer, rdgBackwardDepth, rdgGbufferGelMask, rdgColorBeforeMerge, rdgGbufferNormal,
                    rdgGbufferWorldPos](PushConst_Gelatin pc, const FrameGraphPassContext& ctx) {
                    pc.mDepthFront  = ctx.m_FgDesc->GetSRV(rdgDepthBuffer);
                    pc.mDepthBack   = ctx.m_FgDesc->GetSRV(*rdgBackwardDepth);
                    pc.mMask        = ctx.m_FgDesc->GetSRV(*rdgGbufferGelMask);
                    pc.mCurColor    = ctx.m_FgDesc->GetSRV(*rdgColorBeforeMerge);
                    pc.mCurNormal   = ctx.m_FgDesc->GetSRV(*rdgGbufferNormal);
                    pc.mCurWorldPos = ctx.m_FgDesc->GetSRV(*rdgGbufferWorldPos);
                    FrameGraphUtils::SetRootConstant(pc, ctx);
                })
                .AddRenderTarget(rdgColorBuffer, RhiRenderTargetLoadOp::Clear, Vector4f(0.0f, 0.0f, 0.0f, 1.0f))
                .AddReadResource(*rdgBackwardDepth)
                .AddReadResource(rdgDepthBuffer)
                .AddReadResource(*rdgColorBeforeMerge)
                .AddReadResource(*rdgGbufferAlbedo)
                .AddReadResource(*rdgGbufferNormal)
                .AddReadResource(*rdgGbufferWorldPos)
                .AddReadResource(*rdgGbufferGelMask);

            // End of Gelatin

            auto compiledFg = m_Resources->m_FgCompiler->Compile(builder);
            m_Resources->m_FgExecutor->ExecuteInSingleCmd(cmd, compiledFg);
        }
        cmd->EndScope();
    }

    IFRIT_APIDECL Owner<BaseDeferredRenderer::GPUCommandSubmission> BaseDeferredRenderer::Render(Scene* scene,
        Camera* camera, RenderTargets* renderTargets, const RendererConfig& config,
        const Vec<GPUCommandSubmission*>& cmdToWait)
    {
        SetRendererConfig(&config);

        SceneCollectConfig sceneConfig;
        sceneConfig.projectionTranslateX = 0.0f;
        sceneConfig.projectionTranslateY = 0.0f;

        auto& perframeData = *scene->GetPerFrameData();
        CollectPerframeData(perframeData, scene, camera, GraphicsShaderPassType::Opaque, renderTargets, sceneConfig);
        PrepareDeviceResources(perframeData, renderTargets);

        auto rhi = m_app->GetRhi();
        auto dq  = rhi->GetQueue(RhiQueueCapability::RhiQueue_Graphics);

        auto task = dq->RunAsyncCommand(
            [&](const RhiCommandList* cmd) { SetupAndRunFrameGraph(scene, perframeData, renderTargets, cmd); },
            cmdToWait, {});
        return task;
    }

} // namespace Ifrit::Runtime
