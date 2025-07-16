#pragma once
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/renderer/RendererBase.h"
#include "ifrit/runtime/forwarding/FwdShaderRegistry.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraph.h"

namespace Ifrit::Runtime
{
    struct RendererWrapperData;

    class IFRIT_APIDECL RendererWrapper
    {
    private:
        RendererWrapperData* m_Data;

    public:
        RendererWrapper(RHI::RhiBackend* rhi, ShaderRegistry* shaderRegistry);
        ~RendererWrapper();

        void SetRenderer(RendererBase* renderer);
        void BeginFrame();
        void EndFrame();
        void EnqueueGeneralTask(Fn<Owner<RHI::RhiTaskSubmission>(RHI::RhiTaskSubmission*)> taskFn);
        void EnqueueRDGTask(Fn<void(FrameGraphBuilder*)> taskFn, FrameGraphResourcePool* pool);
        void EnqueueRendererTask(
            Scene* scene, Camera* camera, RHI::RhiRenderTargets* renderTargets, const RendererConfig& config);
    };
} // namespace Ifrit::Runtime