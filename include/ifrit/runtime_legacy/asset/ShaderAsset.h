#pragma once
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/runtime/base/ApplicationInterface.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL IF_CLASS() ShaderAsset : public Asset
    {
    private:
        using ShaderRef        = RHI::RhiShader;
        using ShaderCollection = RHI::RhiShaderCollection;

        Ref<ShaderCollection> m_selfData;
        bool                  m_loaded = false;

    public:
        using Asset::Asset;
        ShaderRef*                LoadShader(const Vec<String>& permutations);
        inline virtual EAssetType GetAsseType() const final { return EAssetType::Shader; }
    };
} // namespace Ifrit::Runtime
