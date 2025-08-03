#pragma once
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/core/reflection/ReflAttrs.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL IF_CLASS() TextureAsset : public Asset
    {
    public:
        using Asset::Asset;
        virtual RHI::RhiTextureRef GetTexture() = 0;
        inline virtual EAssetType  GetAsseType() const final { return EAssetType::Texture; }
    };

} // namespace Ifrit::Runtime
