#include "ifrit/runtime/renderer/PostprocessPass.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/file/FileOps.h"

using namespace Ifrit::Graphics::Rhi;
namespace Ifrit::Runtime
{
    IFRIT_APIDECL PostprocessPass::GPUShader* PostprocessPass::CreateShaderFromFile(
        const String& shaderPath, const String& entry, Graphics::Rhi::RhiShaderStage stage)
    {
        auto              rhi            = m_app->GetRhi();
        String            shaderBasePath = IFRIT_RUNTIME_SHARED_SHADER_PATH;
        auto              path           = shaderBasePath + "/Postprocess/" + shaderPath;
        auto              shaderCode     = ReadTextFile(path);
        std::vector<char> shaderCodeVec(shaderCode.begin(), shaderCode.end());
        return rhi->CreateShader(shaderPath, shaderCodeVec, entry, stage, RhiShaderSourceType::GLSLCode);
    }

    IFRIT_APIDECL PostprocessPass::DrawPass* PostprocessPass::SetupRenderPipeline(RenderTargets* renderTargets)
    {
        auto                      rhi = m_app->GetRhi();
        PipelineAttachmentConfigs paCfg;
        auto                      rtCfg = renderTargets->GetFormat();
        paCfg.m_colorFormats            = { rtCfg.m_colorFormats[0] };
        paCfg.m_depthFormat             = rtCfg.m_depthFormat;
        rtCfg.m_colorFormats            = paCfg.m_colorFormats;

        DrawPass* pass = nullptr;
        if (m_renderPipelines.find(paCfg) != m_renderPipelines.end())
        {
            pass = m_renderPipelines[paCfg];
        }
        else
        {
            pass          = rhi->CreateGraphicsPass();
            auto vsShader = CreateShaderFromFile("Postproc.Common.vert.glsl", "main", RhiShaderStage::Vertex);
            auto fsShader = CreateShaderFromFile(m_cfg.fragPath, "main", RhiShaderStage::Fragment);
            pass->SetPixelShader(fsShader);
            pass->SetVertexShader(vsShader);
            pass->SetNumBindlessDescriptorSets(m_cfg.numDescriptorSets);
            pass->SetPushConstSize(sizeof(u32) * m_cfg.numPushConstants);
            pass->SetRenderTargetFormat(rtCfg);
            m_renderPipelines[paCfg] = pass;
        }
        return pass;
    }

    IFRIT_APIDECL PostprocessPass::ComputePass* PostprocessPass::SetupComputePipeline()
    {
        auto rhi = m_app->GetRhi();
        if (m_computePipeline == nullptr)
        {
            m_computePipeline = rhi->CreateComputePass();
            auto csShader     = CreateShaderFromFile(m_cfg.fragPath, "main", RhiShaderStage::Compute);
            m_computePipeline->SetNumBindlessDescriptorSets(m_cfg.numDescriptorSets);
            m_computePipeline->SetComputeShader(csShader);
            m_computePipeline->SetPushConstSize(sizeof(u32) * m_cfg.numPushConstants);
        }
        return m_computePipeline;
    }

    IFRIT_APIDECL void PostprocessPass::RenderInternal(PerFrameData* perframeData, RenderTargets* renderTargets,
        const GPUCmdBuffer* cmd, const void* pushConstants, const std::vector<GPUBindlessRef*>& bindDescs,
        const String& scopeName)
    {
        auto pass = SetupRenderPipeline(renderTargets);
        auto rhi  = m_app->GetRhi();
        pass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            for (auto i = 0; i < bindDescs.size(); i++)
            {
                ctx->m_cmd->AttachBindlessRefGraphics(pass, i + 1, bindDescs[i]);
            }
            ctx->m_cmd->SetPushConst(pass, 0, m_cfg.numPushConstants * sizeof(u32), pushConstants);
            ctx->m_cmd->AttachVertexBufferView(*rhi->GetFullScreenQuadVertexBufferView());
            ctx->m_cmd->AttachVertexBuffers(0, { rhi->GetFullScreenQuadVertexBuffer().get() });
            ctx->m_cmd->DrawInstanced(3, 1, 0, 0);
        });
        if (scopeName.size() > 0)
            cmd->BeginScope(scopeName);
        pass->Run(cmd, renderTargets, 0);
        if (scopeName.size() > 0)
            cmd->EndScope();
    }

    IFRIT_APIDECL PostprocessPass::PostprocessPass(IApplication* app, const PostprocessPassConfig& cfg)
        : m_app(app), m_cfg(cfg)
    {

        if (cfg.isComputeShader)
        {
            // iInfo("Creating compute shader pipeline");
            SetupComputePipeline();
        }
    }

} // namespace Ifrit::Runtime