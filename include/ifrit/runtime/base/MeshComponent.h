#pragma once
#include "Mesh.h"
#include "Component.h"
#include "Material.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/geomproc/mesh/MeshClusterBase.h"
#include "ifrit/core/reflection/ReflAttrs.h"
#include "ifrit/runtime/asset/MeshAsset.h"
#include "ifrit/runtime/asset/MaterialAsset.h"
namespace Ifrit::Runtime
{
    class IFRIT_APIDECL IF_CLASS() MeshFilter : public Component
    {
    public:
        IF_PROPERTY(Editable, AssetCategory = "Mesh")
        AssetReferenceId mMesh;

    private:
        Ref<MeshInstance> m_instance = nullptr;

    public:
        MeshFilter() { m_instance = MakeRef<MeshInstance>(); }
        MeshFilter(GameObject* owner) : Component(owner) { m_instance = MakeRef<MeshInstance>(); }
        virtual ~MeshFilter() = default;

        void                     SetMeshSource(MeshAsset* p);
        Mesh*                    GetMesh();

        inline Ref<MeshInstance> GetMeshInstance() { return m_instance; }
    };

    class IFRIT_APIDECL IF_CLASS() MeshRenderer : public Component
    {
    public:
        IF_PROPERTY(Editable, AssetCategory = "Material")
        AssetReferenceId mMaterial;

    public:
        using Component::Component;
        virtual ~MeshRenderer() = default;

        Material* GetMaterial();
        void      SetMaterialSource(MaterialAsset* p);
    };

} // namespace Ifrit::Runtime
