#pragma once
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/core/application/Application.h"
#include "ifrit/core/base/Material.h"

namespace Ifrit::Core
{

    struct SyaroDefaultGBufEmitterData
    {
        u32 m_albedoId;
        u32 m_normalMapId;
    };

    class IFRIT_APIDECL SyaroDefaultGBufEmitter : public Material
    {
    private:
        SyaroDefaultGBufEmitterData      m_materialData;
        static Graphics::Rhi::RhiShader* m_shader;
        static ShaderEffect              m_shaderEffect;

    public:
        SyaroDefaultGBufEmitter(IApplication* app);
        ~SyaroDefaultGBufEmitter() = default;

        void        BuildMaterial();

        inline void SetAlbedoId(u32 id) { m_materialData.m_albedoId = id; }
        inline void SetNormalMapId(u32 id) { m_materialData.m_normalMapId = id; }
        inline u32  GetAlbedoId() const { return m_materialData.m_albedoId; }
        inline u32  GetNormalMapId() const { return m_materialData.m_normalMapId; }
    };
} // namespace Ifrit::Core