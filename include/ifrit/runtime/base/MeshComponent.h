#pragma once
#include "Mesh.h"
#include "Component.h"
#include "Material.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/geomproc/mesh/MeshClusterBase.h"
#include "ifrit/core/reflection/ReflAttrs.h"

namespace Ifrit::Runtime
{
    class IF_CLASS() MeshFilter : public Component
    {
    private:
        bool              m_meshLoaded = false;
        Ref<Mesh>         m_rawData    = nullptr;
        AssetReference    m_meshReference;
        // this points to the actual object used for primitive gathering
        Ref<Mesh>         m_attribute = nullptr;
        Ref<MeshInstance> m_instance  = nullptr;

    public:
        MeshFilter() { m_instance = MakeRef<MeshInstance>(); }
        MeshFilter(GameObject* owner) : Component(owner) { m_instance = MakeRef<MeshInstance>(); }
        virtual ~MeshFilter() = default;

        void        LoadMesh();
        inline void SetMesh(Ref<Mesh> p)
        {
            m_meshReference = p->m_assetReference;
            if (!p->m_usingAsset)
            {
                m_rawData = p;
            }
            m_attribute = p;
        }
        inline virtual Vec<AssetReference*> GetAssetRefs() override
        {
            if (m_meshReference.m_usingAsset == false)
                return {};
            return { &m_meshReference };
        }
        inline virtual void SetAssetReferencedAttributes(const Vec<Ref<IAssetCompatible>>& out) override
        {
            if (m_meshReference.m_usingAsset)
            {
                auto mesh   = CheckedPointerCast<Mesh>(out[0]);
                m_attribute = mesh;
            }
        }
        inline Ref<Mesh>         GetMesh() { return m_attribute; }
        inline Ref<MeshInstance> GetMeshInstance() { return m_instance; }
        IFRIT_COMPONENT_SERIALIZE_EMPTY();
    };

    class IF_CLASS() MeshRenderer : public Component
    {
    private:
        Ref<Material>  m_material = nullptr;
        AssetReference m_materialReference;

    public:
        using Component::Component;
        virtual ~MeshRenderer() = default;

        inline Ref<Material> GetMaterial() { return m_material; }
        inline void          SetMaterial(Ref<Material> p) { m_material = p; }

        IFRIT_COMPONENT_SERIALIZE_EMPTY();
    };

} // namespace Ifrit::Runtime
