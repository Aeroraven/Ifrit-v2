#include "ifrit/runtime/util/RendererWrapper.h"

namespace Ifrit::Runtime
{
    struct RendererWrapperData
    {
        RHI::RhiBackend*              m_RhiBackend;
        RendererBase*                 m_Renderer;
        Owner<RHI::RhiTaskSubmission> m_LastEnqueuedTasks;
        ShaderRegistry*               m_ShaderRegistry;
        Ref<FrameGraphCompiler>       m_FrameGraphCompiler;
        Ref<FrameGraphExecutor>       m_FrameGraphExecutor;
    };

    IFRIT_APIDECL RendererWrapper::RendererWrapper(RHI::RhiBackend* rhi, ShaderRegistry* shaderRegistry)
        : m_Data(new RendererWrapperData())
    {
        m_Data->m_RhiBackend         = rhi;
        m_Data->m_Renderer           = nullptr;
        m_Data->m_LastEnqueuedTasks  = nullptr;
        m_Data->m_ShaderRegistry     = shaderRegistry;
        m_Data->m_FrameGraphCompiler = MakeRef<FrameGraphCompiler>();
        m_Data->m_FrameGraphExecutor = MakeRef<FrameGraphExecutor>(rhi);
    }
    IFRIT_APIDECL      RendererWrapper::~RendererWrapper() { delete m_Data; }

    IFRIT_APIDECL void RendererWrapper::SetRenderer(RendererBase* renderer) { m_Data->m_Renderer = renderer; }
    IFRIT_APIDECL void RendererWrapper::BeginFrame()
    {
        iAssertion(m_Data->m_Renderer != nullptr, "RendererWrapper: Renderer must be set before beginning frame.");
        m_Data->m_LastEnqueuedTasks = m_Data->m_Renderer->BeginFrame();
    }
    IFRIT_APIDECL void RendererWrapper::EndFrame()
    {
        iAssertion(m_Data->m_Renderer != nullptr, "RendererWrapper: Renderer must be set before ending frame.");
        if (m_Data->m_LastEnqueuedTasks)
        {
            m_Data->m_Renderer->EndFrame({ m_Data->m_LastEnqueuedTasks.get() });
            m_Data->m_LastEnqueuedTasks = nullptr; // Clear after use
        }
        else
        {
            m_Data->m_Renderer->EndFrame({});
        }
    }
    IFRIT_APIDECL void RendererWrapper::EnqueueGeneralTask(
        Fn<Owner<RHI::RhiTaskSubmission>(RHI::RhiTaskSubmission*)> taskFn)
    {
        iAssertion(m_Data->m_Renderer != nullptr, "RendererWrapper: Renderer must be set before enqueuing tasks.");
        iAssertion(taskFn != nullptr, "RendererWrapper: Task function must not be null.");
        if (m_Data->m_LastEnqueuedTasks)
        {
            // If there are already tasks enqueued, we can chain the new task
            auto toWait = taskFn(m_Data->m_LastEnqueuedTasks.get());
            if (toWait)
            {
                m_Data->m_LastEnqueuedTasks = std::move(toWait);
            }
        }
        else
        {
            // If no tasks are enqueued, we create a new one
            m_Data->m_LastEnqueuedTasks = taskFn(nullptr);
        }
    }

    IFRIT_APIDECL void RendererWrapper::EnqueueRendererTask(
        Scene* scene, Camera* camera, RHI::RhiRenderTargets* renderTargets, const RendererConfig& config)
    {
        iAssertion(m_Data->m_Renderer != nullptr, "RendererWrapper: Renderer must be set before enqueuing tasks.");
        if (m_Data->m_LastEnqueuedTasks)
        {
            m_Data->m_LastEnqueuedTasks =
                m_Data->m_Renderer->Render(scene, camera, renderTargets, config, { m_Data->m_LastEnqueuedTasks.get() });
        }
        else
        {
            m_Data->m_LastEnqueuedTasks = m_Data->m_Renderer->Render(scene, camera, renderTargets, config, {});
        }
    }

    IFRIT_APIDECL void RendererWrapper::EnqueueRDGTask(
        Fn<void(FrameGraphBuilder*)> taskFn, FrameGraphResourcePool* pool)
    {
        auto rhi                    = m_Data->m_RhiBackend;
        auto dq                     = rhi->GetQueue(RHI::RhiQueueCapability::RhiQueue_Graphics);
        m_Data->m_LastEnqueuedTasks = dq->RunAsyncCommand(
            [&](const RHI::RhiCommandList* cmd) {
                FrameGraphBuilder builder(m_Data->m_ShaderRegistry, rhi, pool);
                taskFn(&builder);
                auto fg = m_Data->m_FrameGraphCompiler->Compile(builder);
                m_Data->m_FrameGraphExecutor->ExecuteInSingleCmd(cmd, fg);
            },
            { m_Data->m_LastEnqueuedTasks.get() }, {});
    }
} // namespace Ifrit::Runtime