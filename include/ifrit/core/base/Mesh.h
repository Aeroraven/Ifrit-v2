
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "AssetReference.h"
#include "Component.h"
#include "Material.h"
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/meshproc/engine/mesh/MeshClusterBase.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Core
{
    struct MeshData
    {
        struct GPUCPCounter
        {
            u32 totalBvhNodes;
            u32 totalNumClusters;
            u32 totalLods;
            u32 pad1;
        };

        struct MeshletData
        {
            u32      vertexOffset;
            u32      triangleOffset;
            u32      vertexCount;
            u32      triangleCount;
            Vector4f normalConeAxisCutoff;
            Vector4f normalConeApex;
            Vector4f boundSphere;
            Vector4f selfErrorSphere;
        };
        String                                          identifier;

        Vec<Vector3f>                                   m_vertices;
        Vec<Vector4f>                                   m_verticesAligned;
        Vec<Vector3f>                                   m_normals;
        Vec<Vector4f>                                   m_normalsAligned;
        Vec<Vector2f>                                   m_uvs;
        Vec<Vector4f>                                   m_tangents;
        Vec<u32>                                        m_indices;

        // Cluster data
        Vec<MeshletData>                                m_meshlets;
        Vec<Vector4f>                                   m_normalsCone;
        Vec<Vector4f>                                   m_normalsConeApex;
        Vec<Vector4f>                                   m_boundSphere;
        Vec<u32>                                        m_meshletTriangles;
        Vec<u32>                                        m_meshletVertices;
        Vec<u32>                                        m_meshletInClusterGroup;
        Vec<MeshProcLib::MeshProcess::MeshletCullData>  m_meshCullData;
        Vec<MeshProcLib::MeshProcess::FlattenedBVHNode> m_bvhNodes; // seems not suitable to be here
        Vec<MeshProcLib::MeshProcess::ClusterGroup>     m_clusterGroups;

        // Num meshlets in each lod
        Vec<u32>                                        m_numMeshletsEachLod;

        GPUCPCounter                                    m_cpCounter;
        u32                                             m_maxLod;

        IFRIT_STRUCT_SERIALIZE(m_vertices, m_normals, m_uvs, m_tangents, m_indices);
    };

    struct MeshInstanceTransform
    {
        Matrix4x4f model;
        Matrix4x4f invModel;
        f32        maxScale;
    };

    class IFRIT_APIDECL Mesh : public AssetReferenceContainer, public IAssetCompatible
    {
        using GPUBuffer = Graphics::Rhi::RhiBufferRef;
        using GPUBindId = Graphics::Rhi::RhiDescHandleLegacy;

    public:
        struct GPUObjectBuffer
        {
            Vector4f boundingSphere;
            u32      vertexBufferId;
            u32      normalBufferId;
            u32      tangentBufferId;
            u32      uvBufferId;
            u32      meshletBufferId;
            u32      meshletVertexBufferId;
            u32      meshletIndexBufferId;
            u32      meshletCullBufferId;
            u32      bvhNodeBufferId;
            u32      clusterGroupBufferId;
            u32      meshletInClusterBufferId;
            u32      cpCounterBufferId;
            u32      materialDataId;
            u32      pad2;
            u32      pad3;
        };

        struct GPUResource
        {
            GPUBuffer       vertexBuffer           = nullptr; // should be aligned
            GPUBuffer       normalBuffer           = nullptr; // should be aligned
            GPUBuffer       uvBuffer               = nullptr;
            GPUBuffer       meshletBuffer          = nullptr;
            GPUBuffer       meshletVertexBuffer    = nullptr;
            GPUBuffer       meshletIndexBuffer     = nullptr;
            GPUBuffer       meshletCullBuffer      = nullptr;
            GPUBuffer       bvhNodeBuffer          = nullptr;
            GPUBuffer       clusterGroupBuffer     = nullptr;
            GPUBuffer       meshletInClusterBuffer = nullptr;
            GPUBuffer       cpCounterBuffer        = nullptr;
            GPUBuffer       materialDataBuffer     = nullptr; // currently, opaque is used to hold material data
            GPUBuffer       tangentBuffer          = nullptr;

            GPUObjectBuffer objectData;
            GPUBuffer       objectBuffer = nullptr;

            bool            haveMaterialData = false;

        } m_resource;
        bool                  m_resourceDirty = true;
        Ref<MeshData>         m_data;

        virtual Ref<MeshData> LoadMesh() { return m_data; }

        // Profile result shows that the copy of shared_ptr takes most of
        // game Loop time, so a funcion indicating no ownership transfer
        // might be useful
        virtual MeshData*     LoadMeshUnsafe() { return m_data.get(); }

        inline void           SetGPUResource(GPUResource& resource)
        {
            m_resource.vertexBuffer           = resource.vertexBuffer;
            m_resource.normalBuffer           = resource.normalBuffer;
            m_resource.uvBuffer               = resource.uvBuffer;
            m_resource.meshletBuffer          = resource.meshletBuffer;
            m_resource.meshletVertexBuffer    = resource.meshletVertexBuffer;
            m_resource.meshletIndexBuffer     = resource.meshletIndexBuffer;
            m_resource.meshletCullBuffer      = resource.meshletCullBuffer;
            m_resource.bvhNodeBuffer          = resource.bvhNodeBuffer;
            m_resource.clusterGroupBuffer     = resource.clusterGroupBuffer;
            m_resource.meshletInClusterBuffer = resource.meshletInClusterBuffer;
            m_resource.cpCounterBuffer        = resource.cpCounterBuffer;
            m_resource.materialDataBuffer     = resource.materialDataBuffer;
            m_resource.tangentBuffer          = resource.tangentBuffer;

            m_resource.objectBuffer = resource.objectBuffer;
            m_resource.objectData   = resource.objectData;
        }
        inline void GetGPUResource(GPUResource& resource)
        {
            resource.vertexBuffer           = m_resource.vertexBuffer;
            resource.normalBuffer           = m_resource.normalBuffer;
            resource.uvBuffer               = m_resource.uvBuffer;
            resource.meshletBuffer          = m_resource.meshletBuffer;
            resource.meshletVertexBuffer    = m_resource.meshletVertexBuffer;
            resource.meshletIndexBuffer     = m_resource.meshletIndexBuffer;
            resource.meshletCullBuffer      = m_resource.meshletCullBuffer;
            resource.bvhNodeBuffer          = m_resource.bvhNodeBuffer;
            resource.clusterGroupBuffer     = m_resource.clusterGroupBuffer;
            resource.meshletInClusterBuffer = m_resource.meshletInClusterBuffer;
            resource.cpCounterBuffer        = m_resource.cpCounterBuffer;
            resource.materialDataBuffer     = m_resource.materialDataBuffer;
            resource.tangentBuffer          = m_resource.tangentBuffer;

            resource.objectBuffer = m_resource.objectBuffer;
            resource.objectData   = m_resource.objectData;
        }
        // TODO: static method
        virtual void     CreateMeshLodHierarchy(Ref<MeshData> meshData, const String& cachePath);
        virtual Vector4f GetBoundingSphere(const Vec<Vector3f>& vertices);

        IFRIT_STRUCT_SERIALIZE(m_data, m_assetReference, m_usingAsset);
    };

    // This subjects to change. It's only an alleviation for the coupled design of
    // mesh data and instance making each instance have its own mesh data is not a
    // good idea. However, a cp queue for each instance is still not a good idea
    // Migrating this into persistent culling pass's buffer might be an alternative
    class IFRIT_APIDECL MeshInstance
    {
        using GPUBuffer = Graphics::Rhi::RhiBufferRef;
        using GPUBindId = Graphics::Rhi::RhiDescHandleLegacy;

    public:
        struct GPUObjectBuffer
        {
            u32 cpQueueBufferId;
            u32 cpCounterBufferId;
            u32 filteredMeshletsId;
            u32 pad;
        };

        struct GPUResource
        {
            GPUBuffer       cpQueueBuffer    = nullptr;
            GPUBuffer       filteredMeshlets = nullptr;
            GPUObjectBuffer objectData;
            GPUBuffer       objectBuffer = nullptr;
        } m_resource;

        inline void SetGPUResource(GPUResource& resource)
        {
            m_resource.filteredMeshlets = resource.filteredMeshlets;
            m_resource.cpQueueBuffer    = resource.cpQueueBuffer;

            m_resource.objectBuffer = resource.objectBuffer;
            m_resource.objectData   = resource.objectData;
        }
        inline void GetGPUResource(GPUResource& resource)
        {
            resource.filteredMeshlets = m_resource.filteredMeshlets;
            resource.cpQueueBuffer    = m_resource.cpQueueBuffer;

            resource.objectBuffer = m_resource.objectBuffer;
            resource.objectData   = m_resource.objectData;
        }
    };

    class MeshFilter : public Component
    {
    private:
        bool              m_meshLoaded = false;
        Ref<Mesh>         m_rawData    = nullptr;
        AssetReference    m_meshReference;
        // this points to the actual object used for primitive gathering
        Ref<Mesh>         m_attribute = nullptr;
        Ref<MeshInstance> m_instance  = nullptr;

    public:
        MeshFilter() { m_instance = std::make_shared<MeshInstance>(); }
        MeshFilter(Ref<SceneObject> owner)
            : Component(owner) { m_instance = std::make_shared<MeshInstance>(); }
        virtual ~MeshFilter() = default;
        inline String Serialize() override { return ""; }
        inline void   Deserialize() override {}
        void          LoadMesh();
        inline void   SetMesh(Ref<Mesh> p)
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
                auto mesh   = Common::Utility::CheckedPointerCast<Mesh>(out[0]);
                m_attribute = mesh;
            }
        }
        inline Ref<Mesh>         GetMesh() { return m_attribute; }
        inline Ref<MeshInstance> GetMeshInstance() { return m_instance; }
        IFRIT_COMPONENT_SERIALIZE(m_rawData, m_meshReference);
    };

    class MeshRenderer : public Component
    {
    private:
        Ref<Material>  m_material;
        AssetReference m_materialReference;

    public:
        MeshRenderer() {} // for deserialization
        MeshRenderer(Ref<SceneObject> owner)
            : Component(owner) {}
        virtual ~MeshRenderer() = default;
        inline String        Serialize() override { return ""; }
        inline void          Deserialize() override {}
        inline Ref<Material> GetMaterial() { return m_material; }
        inline void          SetMaterial(Ref<Material> p) { m_material = p; }

        IFRIT_COMPONENT_SERIALIZE(m_materialReference);
    };

} // namespace Ifrit::Core

IFRIT_COMPONENT_REGISTER(Ifrit::Core::MeshFilter);
IFRIT_COMPONENT_REGISTER(Ifrit::Core::MeshRenderer);