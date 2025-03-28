
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

#include "ifrit/common/base/IfritBase.h"

#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/util/FileOps.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/renderer/RendererUtil.h"
#include "ifrit/core/renderer/SyaroRenderer.h"
#include "ifrit/core/renderer/util/RenderingUtils.h"
#include <algorithm>
#include <bit>

#include "ayanami/AyanamiGlobalDF.h"

using Ifrit::Common::Utility::SizeCast;
using Ifrit::Math::DivRoundUp;

namespace Ifrit::Core
{

    struct AyanamiRendererResources;
    class IFRIT_APIDECL AyanamiRenderer : public RendererBase
    {

        using RenderTargets        = Graphics::Rhi::RhiRenderTargets;
        using GPUCommandSubmission = Graphics::Rhi::RhiTaskSubmission;
        using GPUBuffer            = Graphics::Rhi::RhiBuffer;
        using GPUBindId            = Graphics::Rhi::RhiDescHandleLegacy;
        using GPUDescRef           = Graphics::Rhi::RhiBindlessDescriptorRef;
        using ComputePass          = Graphics::Rhi::RhiComputePass;
        using DrawPass             = Graphics::Rhi::RhiGraphicsPass;
        using GPUShader            = Graphics::Rhi::RhiShader;
        using GPUTexture           = Graphics::Rhi::RhiTexture;
        using GPUColorRT           = Graphics::Rhi::RhiColorAttachment;
        using GPURTs               = Graphics::Rhi::RhiRenderTargets;
        using GPUCmdBuffer         = Graphics::Rhi::RhiCommandList;
        using GPUSampler           = Graphics::Rhi::RhiSampler;

        // Perframe data maintained by the renderer, this is unsafe
        // This will be dropped in the future
        HashMap<Scene*, PerFrameData> m_perScenePerframe;

    private:
        Uref<SyaroRenderer>            m_gbufferRenderer;
        AyanamiRendererResources*      m_resources = nullptr;

        Ayanami::AyanamiRenderConfig   m_selfRenderConfig;
        Uref<Ayanami::AyanamiGlobalDF> m_globalDF = nullptr;

    private:
        void InitRenderer();
        void PrepareResources(RenderTargets* renderTargets, const RendererConfig& config);
        void SetupAndRunFrameGraph(PerFrameData& perframe, RenderTargets* renderTargets, const GPUCmdBuffer* cmd);

    public:
        AyanamiRenderer(IApplication* app, Ayanami::AyanamiRenderConfig config)
            : RendererBase(app), m_gbufferRenderer(std::make_unique<SyaroRenderer>(app)), m_selfRenderConfig(config)
        {
            m_gbufferRenderer->SetRenderRole(SyaroRenderRole::SYARO_DEFERRED_GBUFFER);
            InitRenderer();
            m_globalDF = std::make_unique<Ayanami::AyanamiGlobalDF>(config, app->GetRhi());
        }
        virtual ~AyanamiRenderer();

        virtual Uref<GPUCommandSubmission> Render(Scene* scene, Camera* camera, RenderTargets* renderTargets,
            const RendererConfig&             config,
            const Vec<GPUCommandSubmission*>& cmdToWait) override;
    };

} // namespace Ifrit::Core