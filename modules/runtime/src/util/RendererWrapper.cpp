#include "ifrit/runtime/util/RendererWrapper.h"
#include "ifrit/rhi/common/RhiStructHelper.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"

namespace Ifrit::Runtime
{
    struct RendererWrapperData
    {
        RHI::RhiBackend*                    m_RhiBackend;
        RendererBase*                       m_Renderer;
        Owner<RHI::RhiTaskSubmission>       m_LastEnqueuedTasks;
        ShaderRegistry*                     m_ShaderRegistry;
        Ref<FrameGraphCompiler>             m_FrameGraphCompiler;
        Ref<FrameGraphExecutor>             m_FrameGraphExecutor;

        Ref<RHI::RhiRenderTargets>          m_DefaultRenderTargets;
        Ref<RHI::RhiColorAttachment>        m_DefaultColorAttachment;
        RHI::RhiTextureRef                  m_DefaultDepthImage;
        RHI::RhiTextureRef                  m_DefaultColorImage;
        Ref<RHI::RhiDepthStencilAttachment> m_DefaultDepthAttachment;
        RendererConfig                      m_RendererConfig;
        Ref<FrameGraphResourcePool>         m_FrameGraphResourcePoolPrivate;
    };

    IFRIT_APIDECL RendererWrapper::RendererWrapper(
        RHI::RhiBackend* rhi, ShaderRegistry* shaderRegistry, const ProjectProperty& property)
        : m_Data(new RendererWrapperData())
    {
        m_Data->m_RhiBackend         = rhi;
        m_Data->m_Renderer           = nullptr;
        m_Data->m_LastEnqueuedTasks  = nullptr;
        m_Data->m_ShaderRegistry     = shaderRegistry;
        m_Data->m_FrameGraphCompiler = MakeRef<FrameGraphCompiler>();
        m_Data->m_FrameGraphExecutor = MakeRef<FrameGraphExecutor>(rhi);

        auto rtWidth         = property.m_DefaultRTWidth > 0 ? property.m_DefaultRTWidth : property.m_width;
        auto rtHeight        = property.m_DefaultRTHeight > 0 ? property.m_DefaultRTHeight : property.m_height;
        auto colorImageUsage = RHI::RhiImageUsage::RhiImgUsage_RenderTarget | RHI::RhiImageUsage::RhiImgUsage_ShaderRead
            | RHI::RhiImageUsage::RhiImgUsage_UnorderedAccess | RHI::RhiImageUsage::RhiImgUsage_CopyDst;

        m_Data->m_DefaultDepthImage = rhi->CreateDepthTexture("Default_Depth", rtWidth, rtHeight, false);
        m_Data->m_DefaultColorImage = rhi->CreateTexture2D(
            "Default_Color", rtWidth, rtHeight, RHI::RhiImageFormat::RhiImgFmt_R8G8B8A8_UNORM, colorImageUsage, false);
        m_Data->m_DefaultColorAttachment = rhi->CreateRenderTarget(m_Data->m_DefaultColorImage.get(),
            RHI::CreateRhiClearColorValue(Vector4f(0.0f, 0.0f, 0.0f, 1.0f)), RHI::RhiRenderTargetLoadOp::Clear, 0, 0);
        m_Data->m_DefaultDepthAttachment = rhi->CreateRenderTargetDepthStencil(m_Data->m_DefaultDepthImage.get(),
            RHI::CreateRhiClearDepthStencilValue(1.0f, 0), RHI::RhiRenderTargetLoadOp::Clear);

        m_Data->m_DefaultRenderTargets = rhi->CreateRenderTargets();
        m_Data->m_DefaultRenderTargets->SetColorAttachments({ m_Data->m_DefaultColorAttachment.get() });
        m_Data->m_DefaultRenderTargets->SetDepthStencilAttachment(m_Data->m_DefaultDepthAttachment.get());

        RHI::RhiScissor scissor;
        scissor.x      = 0;
        scissor.y      = 0;
        scissor.width  = rtWidth;
        scissor.height = rtHeight;
        m_Data->m_DefaultRenderTargets->SetRenderArea(scissor);

        m_Data->m_FrameGraphResourcePoolPrivate = MakeRef<FrameGraphResourcePool>(rhi);
    }
    IFRIT_APIDECL      RendererWrapper::~RendererWrapper() { delete m_Data; }

    IFRIT_APIDECL void RendererWrapper::SetRenderer(RendererBase* renderer) { m_Data->m_Renderer = renderer; }
    IFRIT_APIDECL void RendererWrapper::BeginFrame()
    {
        IF_LOG_ASSERTION("RendererWrapper", m_Data->m_Renderer != nullptr,
            "RendererWrapper: Renderer must be set before beginning frame.");
        m_Data->m_LastEnqueuedTasks = m_Data->m_Renderer->BeginFrame();
    }

    IFRIT_APIDECL void RendererWrapper::DrawToScreen()
    {
        auto rhi                    = m_Data->m_RhiBackend;
        auto dq                     = rhi->GetQueue(RHI::RhiQueueCapability::RhiQueue_Graphics);
        m_Data->m_LastEnqueuedTasks = dq->RunAsyncCommand(
            [&](const RHI::RhiCommandList* cmd) {
                cmd->BeginScope("Ifrit.Runtime: DrawToScreen");
                FrameGraphBuilder builder(m_Data->m_ShaderRegistry, rhi, m_Data->m_FrameGraphResourcePoolPrivate.get());
                // Import the default render targets
                auto              swapchainImg = rhi->GetSwapchainImage();
                auto              rdgRT        = &builder.ImportTexture("Default_Swapchain", swapchainImg);
                auto              srcTex       = m_Data->m_DefaultColorImage.get();
                auto              rdgSrcTex    = &builder.ImportTexture("Default_Color", srcTex);

                struct PushConst
                {
                    RHI::RhiSRVDesc m_SrcTex;
                } pc{};

                FrameGraphUtils::AddPostProcessPass<PushConst>(builder, "DrawToScreen",
                    ShaderVariantDesc(Internal::kIntShaderTable.Common.ResolveToSwapchainPS, {}), pc,
                    [rdgSrcTex](PushConst pc, const FrameGraphPassContext& ctx) {
                        pc.m_SrcTex = ctx.m_FgDesc->GetSRV(*rdgSrcTex);
                        FrameGraphUtils::SetRootConstant(pc, ctx);
                    })
                    .AddRenderTarget(*rdgRT, RHI::RhiRenderTargetLoadOp::Clear, Vector4f(0.0f, 0.0f, 0.0f, 1.0f))
                    .AddReadResource(*rdgSrcTex);

                auto fg = m_Data->m_FrameGraphCompiler->Compile(builder);
                m_Data->m_FrameGraphExecutor->ExecuteInSingleCmd(cmd, fg);
                cmd->EndScope();
            },
            { m_Data->m_LastEnqueuedTasks.get() }, {});
    }

    IFRIT_APIDECL void RendererWrapper::EndFrame()
    {
        IF_LOG_ASSERTION("RendererWrapper", m_Data->m_Renderer != nullptr,
            "RendererWrapper: Renderer must be set before ending frame.");
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
        IF_LOG_ASSERTION("RendererWrapper", m_Data->m_Renderer != nullptr,
            "RendererWrapper: Renderer must be set before enqueuing tasks.");
        IF_LOG_ASSERTION("RendererWrapper", taskFn != nullptr, "RendererWrapper: Task function must not be null.");
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
        IF_LOG_ASSERTION("RendererWrapper", m_Data->m_Renderer != nullptr,
            "RendererWrapper: Renderer must be set before enqueuing tasks.");
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

    IFRIT_APIDECL void RendererWrapper::SetRendererConfig(const RendererConfig& config)
    {
        m_Data->m_RendererConfig = config;
    }
    IFRIT_APIDECL RendererConfig RendererWrapper::GetRendererConfig() const { return m_Data->m_RendererConfig; }

    IFRIT_APIDECL RHI::RhiRenderTargets* RendererWrapper::GetDefaultRenderTargets() const
    {
        return m_Data->m_DefaultRenderTargets.get();
    }
    IFRIT_APIDECL RHI::RhiColorAttachment* RendererWrapper::GetDefaultRenderTargetsColor() const
    {
        return m_Data->m_DefaultColorAttachment.get();
    }
    IFRIT_APIDECL RHI::RhiDepthStencilAttachment* RendererWrapper::GetDefaultRenderTargetsDepth() const
    {
        return m_Data->m_DefaultDepthAttachment.get();
    }
    IFRIT_APIDECL RHI::RhiTextureRef RendererWrapper::GetDefaultColorImage() const
    {
        return m_Data->m_DefaultColorImage;
    }
} // namespace Ifrit::Runtime
