#pragma once
#include "ifrit/runtime/common/Pch.h"

#include "ifrit/core/file/FileOps.h"
#include "ifrit/runtime/renderer/RendererUtil.h"
#include "ifrit/runtime/renderer/SyaroRenderer.h"
#include "ifrit/runtime/renderer/util/RenderingUtils.h"

namespace Ifrit::Runtime
{

    struct BaseDeferredRendererResources;

    class IFRIT_APIDECL BaseDeferredRenderer : public RendererBase
    {

        using RenderTargets        = RHI::RhiRenderTargets;
        using GPUCommandSubmission = RHI::RhiTaskSubmission;
        using GPUCmdBuffer         = RHI::RhiCommandList;

    private:
        BaseDeferredRendererResources* m_Resources = nullptr;

    private:
        void InitRenderer();
        void SetupAndRunFrameGraph(
            Scene* scene, PerFrameData& perframe, RenderTargets* renderTargets, const GPUCmdBuffer* cmd);

    public:
        BaseDeferredRenderer(IApplication* app);
        virtual ~BaseDeferredRenderer();

        virtual Owner<GPUCommandSubmission> Render(Scene* scene, Camera* camera, RenderTargets* renderTargets,
            const RendererConfig& config, const Vec<GPUCommandSubmission*>& cmdToWait) override;
    };

} // namespace Ifrit::Runtime
