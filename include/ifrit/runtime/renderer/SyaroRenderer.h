
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

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

#pragma once
#include "AmbientOcclusionPass.h"
#include "PbrAtmosphereRenderer.h"
#include "RendererBase.h"
#include "framegraph/FrameGraph.h"
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/algo/Hash.h"
#include "postprocessing/PostFxAcesTonemapping.h"
#include "postprocessing/PostFxFFTConv2d.h"
#include "postprocessing/PostFxGaussianHori.h"
#include "postprocessing/PostFxGaussianVert.h"
#include "postprocessing/PostFxGlobalFog.h"
#include "postprocessing/PostFxJointBilaterialFilter.h"
#include "postprocessing/PostFxStockhamDFT2.h"

#include "commonpass/SinglePassHiZ.h"

namespace Ifrit::Runtime
{

    enum SyaroRenderRole
    {
        Shading     = 0x1,
        GBuffer     = 0x2,
        Shadowing   = 0x4,
        FullProcess = Shading | GBuffer | Shadowing,
    };

    class IFRIT_APIDECL SyaroRenderer : public RendererBase
    {
        using RenderTargets        = Graphics::Rhi::RhiRenderTargets;
        using GPUCommandSubmission = Graphics::Rhi::RhiTaskSubmission;
        using GPUBuffer            = Graphics::Rhi::RhiBufferRef;
        using GPUBindId            = Graphics::Rhi::RhiDescHandleLegacy;
        using GPUDescRef           = Graphics::Rhi::RhiBindlessDescriptorRef;
        using ComputePass          = Graphics::Rhi::RhiComputePass;
        using DrawPass             = Graphics::Rhi::RhiGraphicsPass;
        using GPUShader            = Graphics::Rhi::RhiShader;
        using GPUTexture           = Graphics::Rhi::RhiTextureRef;
        using GPUColorRT           = Graphics::Rhi::RhiColorAttachment;
        using GPURTs               = Graphics::Rhi::RhiRenderTargets;
        using GPUCmdBuffer         = Graphics::Rhi::RhiCommandList;
        using GPUSampler           = Graphics::Rhi::RhiSamplerRef;

        enum class CullingPass
        {
            First,
            Second
        };

    private:
        using PipeHash    = PipelineAttachmentConfigsHash;
        using PipeConf    = PipelineAttachmentConfigs;
        using PerViewData = PerFrameData::PerViewData;

        // Renderer Role
        u32                                               m_renderRole = SyaroRenderRole::FullProcess;

        // Base
        ComputePass*                                      m_persistentCullingPass = nullptr;
        GPUBuffer                                         m_indirectDrawBuffer    = nullptr;
        GPUDescRef*                                       m_persistCullDesc       = nullptr;

        DrawPass*                                         m_visibilityPassHW          = nullptr;
        DrawPass*                                         m_depthOnlyVisibilityPassHW = nullptr;
        ComputePass*                                      m_visibilityPassSW          = nullptr;
        ComputePass*                                      m_visibilityCombinePass     = nullptr;

        // Instance culling
        ComputePass*                                      m_instanceCullingPass = nullptr;

        // Single pass HiZ
        Ref<SinglePassHiZPass>                            m_singlePassHiZProc = nullptr;

        IF_CONSTEXPR static u32                           cSPHiZGroupSizeX = 256;
        IF_CONSTEXPR static u32                           cSPHiZTileSize   = 64;

        // Emit depth targets
        ComputePass*                                      m_emitDepthTargetsPass = nullptr;
        IF_CONSTEXPR static u32                           cEmitDepthGroupSizeX   = 16;
        IF_CONSTEXPR static u32                           cEmitDepthGroupSizeY   = 16;

        // Material classify
        ComputePass*                                      m_matclassCountPass             = nullptr;
        ComputePass*                                      m_matclassReservePass           = nullptr;
        ComputePass*                                      m_matclassScatterPass           = nullptr;
        IF_CONSTEXPR static u32                           cMatClassQuadSize               = 2;
        IF_CONSTEXPR static u32                           cMatClassGroupSizeCountScatterX = 8;
        IF_CONSTEXPR static u32                           cMatClassGroupSizeCountScatterY = 8;
        IF_CONSTEXPR static u32                           cMatClassGroupSizeReserveX      = 128;
        IF_CONSTEXPR static u32                           cMatClassCounterBufferSizeBase  = 2 * sizeof(u32);
        IF_CONSTEXPR static u32                           cMatClassCounterBufferSizeMult  = 2 * sizeof(u32);

        // Emit GBuffer, pass here is for default / debugging
        ComputePass*                                      m_defaultEmitGBufferPass = nullptr;

        // TAA
        ComputePass*                                      m_taaHistoryPass = nullptr;
        IF_CONSTEXPR static Graphics::Rhi::RhiImageFormat cTAAFormat =
            Graphics::Rhi::RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT;

        // Finally, deferred pass
        CustomHashMap<PipeConf, DrawPass*, PipeHash> m_deferredShadingPass;
        DrawPass*                                    m_deferredShadowPass = nullptr;
        CustomHashMap<PipeConf, DrawPass*, PipeHash> m_taaPass;

        // FSR2
        Uref<Graphics::Rhi::FSR2::RhiFsr2Processor>  m_fsr2proc;

        // Atmosphere
        ComputePass*                                 m_atmospherePass = nullptr;
        Ref<PbrAtmosphereRenderer>                   m_atmosphereRenderer;

        // Timer
        Ref<Graphics::Rhi::RhiDeviceTimer>           m_timer;
        Ref<Graphics::Rhi::RhiDeviceTimer>           m_timerDefer;

        // AO
        Ref<AmbientOcclusionPass>                    m_aoPass;

        // Postprocess, just 2 textures and 1 sampler is required.
        using PairHash = PairwiseHash<u32, u32>;
        CustomHashMap<Pair<u32, u32>, Array<GPUTexture, 2>, PairHash>      m_postprocTex;
        CustomHashMap<Pair<u32, u32>, Array<Ref<GPUBindId>, 2>, PairHash>  m_postprocTexSRV;
        CustomHashMap<Pair<u32, u32>, Array<Ref<GPUColorRT>, 2>, PairHash> m_postprocColorRT;
        CustomHashMap<Pair<u32, u32>, Array<Ref<GPURTs>, 2>, PairHash>     m_postprocRTs;
        GPUSampler                                                         m_postprocTexSampler;
        Ref<GPUBindId>                                                     m_postprocTexSamplerId;

        // All postprocess passes required
        Uref<PostprocessPassCollection::PostFxAcesToneMapping>             m_acesToneMapping;
        Uref<PostprocessPassCollection::PostFxGlobalFog>                   m_globalFogPass;
        Uref<PostprocessPassCollection::PostFxGaussianHori>                m_gaussianHori;
        Uref<PostprocessPassCollection::PostFxGaussianVert>                m_gaussianVert;
        Uref<PostprocessPassCollection::PostFxFFTConv2d>                   m_fftConv2d;
        Uref<PostprocessPassCollection::PostFxJointBilaterialFilter>       m_jointBilateralFilter;

        // Intermediate views
        CustomHashMap<PipeConf, DrawPass*, PipeHash>                       m_triangleViewPass;

        // Render config
        RendererConfig                                                     m_renderConfig;

    private:
        // Util functions
        GPUShader*   GetInternalShader(const char* shaderName);

        // Setup functions
        void         RecreateInstanceCullingBuffers(PerFrameData& perframe, u32 newMaxInstances);
        void         SetupInstanceCullingPass();
        void         SetupPersistentCullingPass();
        void         SetupVisibilityPass();
        void         SetupEmitDepthTargetsPass();
        void         SetupMaterialClassifyPass();
        void         SetupDefaultEmitGBufferPass();
        void         SetupPbrAtmosphereRenderer();
        void         SetupSinglePassHiZPass();
        void         SetupFSR2Data();
        void         SetupPostprocessPassAndTextures();
        void         CreateTimer();

        void         SetupDeferredShadingPass(RenderTargets* renderTargets);
        void         SetupTAAPass(RenderTargets* renderTargets);

        void         SphizBufferSetup(PerFrameData& perframeData, RenderTargets* renderTargets);
        void         VisibilityBufferSetup(PerFrameData& perframeData, RenderTargets* renderTargets);
        void         DepthTargetsSetup(PerFrameData& perframeData, RenderTargets* renderTargets);
        void         MaterialClassifyBufferSetup(PerFrameData& perframeData, RenderTargets* renderTargets);
        void         TaaHistorySetup(PerFrameData& perframeData, RenderTargets* renderTargets);
        void         Fsr2Setup(PerFrameData& perframeData, RenderTargets* renderTargets);
        void         CreatePostprocessTextures(u32 width, u32 height);
        void         PrepareAggregatedShadowData(PerFrameData& perframeData);

        void         SetupDebugPasses(PerFrameData& perframeData, RenderTargets* renderTargets);

        // Many passes are not material-dependent, so a unified instance buffer
        // might reduce calls
        void         GatherAllInstances(PerFrameData& perframeData);

        PerViewData& GetPrimaryView(PerFrameData& perframeData);

    private:
        // Decompose the rendering procedure into many parts
        void RenderTwoPassOcclCulling(CullingPass cullPass, PerFrameData& perframeData, RenderTargets* renderTargets,
            const GPUCmdBuffer* cmd, PerFrameData::ViewType filteredViewType, u32 idx);
        void RenderTriangleView(PerFrameData& perframeData, RenderTargets* renderTargets, const GPUCmdBuffer* cmd);
        void RenderEmitDepthTargets(PerFrameData& perframeData, RenderTargets* renderTargets, const GPUCmdBuffer* cmd);
        void RenderMaterialClassify(PerFrameData& perframeData, RenderTargets* renderTargets, const GPUCmdBuffer* cmd);
        void RenderDefaultEmitGBuffer(
            PerFrameData& perframeData, RenderTargets* renderTargets, const GPUCmdBuffer* cmd);
        void RenderAmbientOccl(PerFrameData& perframeData, RenderTargets* renderTargets, const GPUCmdBuffer* cmd);
        void SetupAndRunFrameGraph(PerFrameData& perframeData, RenderTargets* renderTargets, const GPUCmdBuffer* cmd);

    private:
        virtual Uref<GPUCommandSubmission> Render(
            PerFrameData& perframeData, RenderTargets* renderTargets, const Vec<GPUCommandSubmission*>& cmdToWait);

    public:
        SyaroRenderer(IApplication* app) : RendererBase(app)
        {
            SetupPersistentCullingPass();
            SetupVisibilityPass();
            SetupInstanceCullingPass();
            SetupEmitDepthTargetsPass();
            SetupMaterialClassifyPass();
            SetupDefaultEmitGBufferPass();
            SetupSinglePassHiZPass();
            SetupPbrAtmosphereRenderer();
            SetupPostprocessPassAndTextures();
            CreateTimer();

            m_aoPass = std::make_shared<AmbientOcclusionPass>(app);
        }
        inline void                        SetRenderRole(u32 role) { m_renderRole = role; }
        virtual Uref<GPUCommandSubmission> Render(Scene* scene, Camera* camera, RenderTargets* renderTargets,
            const RendererConfig& config, const Vec<GPUCommandSubmission*>& cmdToWait) override;
    };
} // namespace Ifrit::Runtime