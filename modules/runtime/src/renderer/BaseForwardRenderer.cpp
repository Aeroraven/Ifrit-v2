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

#include "ifrit/runtime/renderer/BaseForwardRenderer.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"

using namespace Ifrit::RHI;

namespace Ifrit::Runtime
{
    struct BaseForwardRendererResources
    {
        Owner<FrameGraphCompiler>   m_FgCompiler;
        Owner<FrameGraphExecutor>   m_FgExecutor;
        Ref<FrameGraphResourcePool> m_ResourcePool = nullptr;
    };

    IFRIT_APIDECL BaseForwardRenderer::BaseForwardRenderer(IApplication* app) : RendererBase(app)
    {
        m_Resources = new BaseForwardRendererResources();
        InitRenderer();
    }

    IFRIT_APIDECL BaseForwardRenderer::~BaseForwardRenderer()
    {
        if (m_Resources)
        {
            delete m_Resources;
            m_Resources = nullptr;
        }
    }

    IFRIT_APIDECL void BaseForwardRenderer::InitRenderer()
    {
        PrepareImmutableResources();
        m_Resources->m_FgExecutor   = MakeOwner<FrameGraphExecutor>(m_app->GetRhi());
        m_Resources->m_FgCompiler   = MakeOwner<FrameGraphCompiler>();
        m_Resources->m_ResourcePool = MakeRef<FrameGraphResourcePool>(m_app->GetRhi());
    }

    IFRIT_APIDECL void BaseForwardRenderer::SetupAndRunFrameGraph(
        Scene* scene, PerFrameData& perframe, RenderTargets* renderTargets, const GPUCmdBuffer* cmd)
    {
        cmd->BeginScope("BaseForward: Execute Render Graph");
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

            // Forward pass
            auto& pass = builder.AddGraphicsPass("Forward Pass",
                ShaderVariantDesc(Internal::kIntShaderTable.BaseForward.ForwardVS, {}),
                ShaderVariantDesc(Internal::kIntShaderTable.BaseForward.ForwardPS, {}), 3);

            auto  rtWidth  = renderTargets->GetRenderArea().width;
            auto  rtHeight = renderTargets->GetRenderArea().height;

            pass.SetExecutionFunction(
                [this, &perframe, &rdgMainView, rtWidth, rtHeight](const FrameGraphPassContext& ctx) {
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
                            } pc;

                            pc.m_ObjectBufferId = shaderEffects.m_batchedObjectData->GetActiveBuffer()->GetDescId();
                            pc.m_PerFrameId     = ctx.m_FgDesc->GetSRV(rdgMainView);
                            pc.m_InBatchOffset  = i;
                            cmd->AttachIndexBuffer(shaderEffects.m_meshes[i]->m_resource.indexBuffer.get());
                            cmd->SetCullMode(RhiCullMode::None);

                            cmd->SetPushConst(&pc, 0, sizeof(PushConst));
                            auto meshData = shaderEffects.m_meshes[i]->LoadMeshUnsafe();

                            if (meshData->m_GenerationType == MeshGeneratorType::Static)
                            {
                                auto indexCount =
                                    SizeCast<u32>(shaderEffects.m_meshes[i]->LoadMeshUnsafe()->m_indices.size());

                                if (indexCount != 0)
                                {
                                    cmd->DrawIndexed(indexCount, 1, 0, 0, 0);
                                }
                            }
                            else if (meshData->m_GenerationType == MeshGeneratorType::Procedual)
                            {
                                cmd->DrawIndexedIndirect(
                                    shaderEffects.m_meshes[i]->m_resource.procIndirectDrawBuffer.get(), 0);
                            }
                        }
                    }
                });

            pass.AddDepthTarget(rdgDepthBuffer);
            pass.AddRenderTarget(rdgColorBuffer, RhiRenderTargetLoadOp::Clear, (Vector4f(0.0f, 0.0f, 0.0f, 1.0f)));

            // End of forward pass

            auto compiledFg = m_Resources->m_FgCompiler->Compile(builder);
            m_Resources->m_FgExecutor->ExecuteInSingleCmd(cmd, compiledFg);
        }
        cmd->EndScope();
    }

    IFRIT_APIDECL Owner<BaseForwardRenderer::GPUCommandSubmission> BaseForwardRenderer::Render(Scene* scene,
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