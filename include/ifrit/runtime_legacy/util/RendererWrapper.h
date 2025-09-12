#pragma once
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/renderer/RendererBase.h"
#include "ifrit/runtime/forwarding/FwdShaderRegistry.h"
#include "ifrit/runtime/rendercore/framegraph/FrameGraph.h"
#include "ifrit/runtime/application/ProjectProperty.h"

namespace Ifrit::Runtime
{
    struct RendererWrapperData;

    class IFRIT_APIDECL RendererWrapper
    {
    private:
        RendererWrapperData* m_Data;

    public:
        RendererWrapper(RHI::RhiBackend* rhi, ShaderRegistry* shaderRegistry, const ProjectProperty& property);
        ~RendererWrapper();

        void SetRenderer(RendererBase* renderer);
        void BeginFrame();
        void EndFrame();
        void DrawToScreen();
        void EnqueueGeneralTask(Fn<Owner<RHI::RhiTaskSubmission>(RHI::RhiTaskSubmission*)> taskFn);
        void EnqueueRDGTask(Fn<void(FrameGraphBuilder*)> taskFn, FrameGraphResourcePool* pool);
        void EnqueueRendererTask(
            Scene* scene, Camera* camera, RHI::RhiRenderTargets* renderTargets, const RendererConfig& config);
        void                            SetRendererConfig(const RendererConfig& config);
        RendererConfig                  GetRendererConfig() const;

        RHI::RhiRenderTargets*          GetDefaultRenderTargets() const;
        RHI::RhiColorAttachment*        GetDefaultRenderTargetsColor() const;
        RHI::RhiDepthStencilAttachment* GetDefaultRenderTargetsDepth() const;
        RHI::RhiTextureRef              GetDefaultColorImage() const;
    };
} // namespace Ifrit::Runtime
