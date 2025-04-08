#include "ifrit/runtime/material/SyaroDefaultGBufEmitter.h"
#include "ifrit/core/file/FileOps.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
namespace Ifrit::Runtime
{
    IFRIT_APIDECL
    SyaroDefaultGBufEmitter::SyaroDefaultGBufEmitter(IApplication* app) : Material()
    {
        m_materialData.m_albedoId = ~0u;
        if (m_shader == nullptr)
        {
            auto              rhi            = app->GetRhi();
            String            shaderBasePath = IFRIT_RUNTIME_SHARED_SHADER_PATH;
            auto              path           = shaderBasePath + "/Syaro/Syaro.EmitGBuffer.Default.comp.glsl";
            auto              shaderCode     = ReadTextFile(path);
            std::vector<char> shaderCodeVec(shaderCode.begin(), shaderCode.end());
            m_shader = rhi->CreateShader(path, shaderCodeVec, "main", Graphics::Rhi::RhiShaderStage::Compute,
                Graphics::Rhi::RhiShaderSourceType::GLSLCode);
            m_shaderEffect.m_shaders.push_back(m_shader);
            m_shaderEffect.m_type = ShaderEffectType::Compute;
        }
        BuildMaterial();
    }

    IFRIT_APIDECL void SyaroDefaultGBufEmitter::BuildMaterial()
    {
        if (m_data.size() < 1)
        {
            m_data.resize(1);
        }
        m_data[0].resize(sizeof(SyaroDefaultGBufEmitterData));
        auto& data = *reinterpret_cast<SyaroDefaultGBufEmitterData*>(m_data[0].data());
        data       = m_materialData;
        this->m_effectTemplates[GraphicsShaderPassType::Opaque] = m_shaderEffect;
    }

    Graphics::Rhi::RhiShader* SyaroDefaultGBufEmitter::m_shader       = nullptr;
    ShaderEffect              SyaroDefaultGBufEmitter::m_shaderEffect = {};

} // namespace Ifrit::Runtime