#pragma once
#include "Mesh.h"
#include "Component.h"
#include "Material.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/geomproc/mesh/MeshClusterBase.h"
#include "ifrit/core/reflection/ReflAttrs.h"
#include "ifrit/runtime/asset/MeshAsset.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL IF_CLASS() MeshFilter : public Component
    {
    public:
        IF_PROPERTY()
        AssetReferenceId mMesh;

    private:
        AssetReferenceId  m_meshReference;
        Ref<MeshInstance> m_instance = nullptr;

    public:
        MeshFilter() { m_instance = MakeRef<MeshInstance>(); }
        MeshFilter(GameObject* owner) : Component(owner) { m_instance = MakeRef<MeshInstance>(); }
        virtual ~MeshFilter() = default;

        void                     SetMeshSource(MeshAsset* p);
        Mesh*                    GetMesh();

        inline Ref<MeshInstance> GetMeshInstance() { return m_instance; }
    };

    class IF_CLASS() MeshRenderer : public Component
    {
    private:
        Ref<Material>    m_material = nullptr;
        AssetReferenceId m_materialReference;

    public:
        using Component::Component;
        virtual ~MeshRenderer() = default;

        inline Ref<Material> GetMaterial() { return m_material; }
        inline void          SetMaterial(Ref<Material> p) { m_material = p; }
    };

} // namespace Ifrit::Runtime
