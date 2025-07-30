#pragma once
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/core/reflection/ReflAttrs.h"
#include "ifrit/runtime/base/Mesh.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL IF_CLASS() MeshAsset : public Asset
    {
    public:
        using Asset::Asset;

        virtual Mesh* GetMesh() = 0;
    };

    class IFRIT_APIDECL IF_CLASS() ImportedMeshAsset : public MeshAsset
    {
    public:
        using MeshAsset::MeshAsset;
    };

} // namespace Ifrit::Runtime