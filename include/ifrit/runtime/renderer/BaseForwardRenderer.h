#pragma once
#include "ifrit/runtime/common/Pch.h"

#include "ifrit/core/file/FileOps.h"
#include "ifrit/runtime/renderer/RendererUtil.h"
#include "ifrit/runtime/renderer/SyaroRenderer.h"
#include "ifrit/runtime/renderer/util/RenderingUtils.h"

namespace Ifrit::Runtime
{

    struct BaseForwardRendererResources;

    // BaseForwardRenderer is a renderer that implements the forward rendering technique.
    // It's designed to add the minimal set of features required for rendering a scene without
    // advanced device support. For advanced devices, please use SyaroV1.
    class IFRIT_APIDECL BaseForwardRenderer : public RendererBase
    {

        using RenderTargets        = RHI::RhiRenderTargets;
        using GPUCommandSubmission = RHI::RhiTaskSubmission;
        using GPUCmdBuffer         = RHI::RhiCommandList;

    private:
        BaseForwardRendererResources* m_Resources = nullptr;

    private:
        void InitRenderer();
        void SetupAndRunFrameGraph(
            Scene* scene, PerFrameData& perframe, RenderTargets* renderTargets, const GPUCmdBuffer* cmd);

    public:
        BaseForwardRenderer(IApplication* app);
        virtual ~BaseForwardRenderer();

        virtual Owner<GPUCommandSubmission> Render(Scene* scene, Camera* camera, RenderTargets* renderTargets,
            const RendererConfig& config, const Vec<GPUCommandSubmission*>& cmdToWait) override;
    };

} // namespace Ifrit::Runtime
