#include "ifrit/runtime/geometry/preset/Plane.h"
#include <numbers>

namespace Ifrit::Runtime::Geometry
{
    IFRIT_APIDECL      Plane::Plane(f32 m_Width, f32 m_Height) : m_Width(m_Width), m_Height(m_Height) {}

    IFRIT_APIDECL void Plane::BuildMesh()
    {
        m_SelfData             = MakeRef<MeshData>();
        m_SelfDataRaw          = m_SelfData.get();
        m_SelfData->m_MeshType = MeshType::Surface;

        m_SelfData->m_vertices.resize(4);
        m_SelfData->m_normals.resize(4);
        m_SelfData->m_uvs.resize(4);
        m_SelfData->m_tangents.resize(4);
        m_SelfData->m_verticesAligned.resize(4);
        m_SelfData->m_normalsAligned.resize(4);
        m_SelfData->m_indices.resize(6);

        // Square
        m_SelfData->m_vertices[0] = Vector3f(-m_Width / 2.0f, 0.0f, -m_Height / 2.0f);
        m_SelfData->m_vertices[1] = Vector3f(m_Width / 2.0f, 0.0f, -m_Height / 2.0f);
        m_SelfData->m_vertices[2] = Vector3f(m_Width / 2.0f, 0.0f, m_Height / 2.0f);
        m_SelfData->m_vertices[3] = Vector3f(-m_Width / 2.0f, 0.0f, m_Height / 2.0f);
        m_SelfData->m_normals[0]  = Vector3f(0.0f, 1.0f, 0.0f);
        m_SelfData->m_normals[1]  = Vector3f(0.0f, 1.0f, 0.0f);
        m_SelfData->m_normals[2]  = Vector3f(0.0f, 1.0f, 0.0f);
        m_SelfData->m_normals[3]  = Vector3f(0.0f, 1.0f, 0.0f);

        m_SelfData->m_uvs[0] = Vector2f(0.0f, 1.0f);
        m_SelfData->m_uvs[1] = Vector2f(0.0f, 1.0f);
        m_SelfData->m_uvs[2] = Vector2f(0.0f, 1.0f);
        m_SelfData->m_uvs[3] = Vector2f(0.0f, 1.0f);

        m_SelfData->m_tangents[0] = Vector4f(1.0f, 0.0f, 0.0f, 1.0f);
        m_SelfData->m_tangents[1] = Vector4f(1.0f, 0.0f, 0.0f, 1.0f);
        m_SelfData->m_tangents[2] = Vector4f(1.0f, 0.0f, 0.0f, 1.0f);
        m_SelfData->m_tangents[3] = Vector4f(1.0f, 0.0f, 0.0f, 1.0f);

        m_SelfData->m_verticesAligned[0] = Vector4f(m_SelfData->m_vertices[0], 1.0f);
        m_SelfData->m_verticesAligned[1] = Vector4f(m_SelfData->m_vertices[1], 1.0f);
        m_SelfData->m_verticesAligned[2] = Vector4f(m_SelfData->m_vertices[2], 1.0f);
        m_SelfData->m_verticesAligned[3] = Vector4f(m_SelfData->m_vertices[3], 1.0f);

        m_SelfData->m_normalsAligned[0] = Vector4f(m_SelfData->m_normals[0], 0.0f);
        m_SelfData->m_normalsAligned[1] = Vector4f(m_SelfData->m_normals[1], 0.0f);
        m_SelfData->m_normalsAligned[2] = Vector4f(m_SelfData->m_normals[2], 0.0f);
        m_SelfData->m_normalsAligned[3] = Vector4f(m_SelfData->m_normals[3], 0.0f);

        // indices
        m_SelfData->m_indices[0] = 0;
        m_SelfData->m_indices[1] = 1;
        m_SelfData->m_indices[2] = 2;
        m_SelfData->m_indices[3] = 0;
        m_SelfData->m_indices[4] = 2;
        m_SelfData->m_indices[5] = 3;
    }

    IFRIT_APIDECL Ref<MeshData> Plane::LoadMesh()
    {
        if (!m_Loaded)
        {
            BuildMesh();
            m_Loaded = true;
        }
        return m_SelfData;
    }

    IFRIT_APIDECL MeshData* Plane::LoadMeshUnsafe()
    {
        if (!m_Loaded)
        {
            BuildMesh();
            m_Loaded = true;
        }
        return m_SelfDataRaw;
    }

    IFRIT_APIDECL u32 Plane::GetNumIndices() { return 6; }
    IFRIT_APIDECL u32 Plane::GetNumVertices() { return 4; }
    IFRIT_APIDECL Vec<u32> Plane::GetIndexBufferHost() { return m_SelfData->m_indices; }
    IFRIT_APIDECL Vec<Vector3f> Plane::GetVertexBufferHost() { return m_SelfData->m_vertices; }

    IFRIT_APIDECL Mesh*         PlaneAsset::GetMesh()
    {
        if (mMesh == nullptr)
        {
            mMesh = MakeOwner<Plane>(mWidth, mHeight);
        }
        return mMesh.get();
    }

} // namespace Ifrit::Runtime::Geometry
