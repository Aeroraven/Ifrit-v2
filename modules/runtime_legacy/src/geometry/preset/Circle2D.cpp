#include "ifrit/runtime/geometry/preset/Circle2D.h"
#include <numbers>

namespace Ifrit::Runtime::Geometry
{
    IFRIT_APIDECL      Circle2D::Circle2D(f32 radius, u32 divisions) : m_Radius(radius), m_Divisions(divisions) {}

    IFRIT_APIDECL void Circle2D::BuildMesh()
    {
        m_SelfData             = MakeRef<MeshData>();
        m_SelfDataRaw          = m_SelfData.get();
        m_SelfData->m_MeshType = MeshType::Surface;

        m_SelfData->m_vertices.resize(m_Divisions + 1);
        m_SelfData->m_normals.resize(m_Divisions + 1);
        m_SelfData->m_uvs.resize(m_Divisions + 1);
        m_SelfData->m_tangents.resize(m_Divisions + 1);
        m_SelfData->m_verticesAligned.resize(m_Divisions + 1);
        m_SelfData->m_normalsAligned.resize(m_Divisions + 1);

        m_SelfData->m_indices.resize(3 * m_Divisions);

        // circle boundary
        for (u32 i = 0; i <= m_Divisions; ++i)
        {
            f32 angle                 = 2.0f * std::numbers::pi_v<f32> * i / m_Divisions;
            m_SelfData->m_vertices[i] = Vector3f(m_Radius * std::cos(angle), m_Radius * std::sin(angle), 0.0f);
            m_SelfData->m_normals[i] = Vector3f(1.0f * i / m_Divisions * 2.0f - 1.0f,
                 1.0f * i / m_Divisions * 2.0f - 1.0f, 1.0f * i / m_Divisions * 2.0f - 1.0f);
            m_SelfData->m_uvs[i]      = Vector2f(0.0f, 0.0f);
            m_SelfData->m_tangents[i] = Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
            m_SelfData->m_verticesAligned[i] =
                Vector4f(m_SelfData->m_vertices[i].x, m_SelfData->m_vertices[i].y, m_SelfData->m_vertices[i].z, 1.0f);
            m_SelfData->m_normalsAligned[i] =
                Vector4f(m_SelfData->m_normals[i].x, m_SelfData->m_normals[i].y, m_SelfData->m_normals[i].z, 1.0f);
        }
        // circle center
        m_SelfData->m_vertices[m_Divisions] = Vector3f(0.0f, 0.0f, 0.0f);
        m_SelfData->m_normals[m_Divisions]  = Vector3f(0.0f, 0.0f, 1.0f);
        m_SelfData->m_uvs[m_Divisions]      = Vector2f(0.0f, 0.0f);
        m_SelfData->m_tangents[m_Divisions] = Vector4f(0.0f, 0.0f, 0.0f, 1.0f);

        m_SelfData->m_verticesAligned[m_Divisions] = Vector4f(m_SelfData->m_vertices[m_Divisions].x,
            m_SelfData->m_vertices[m_Divisions].y, m_SelfData->m_vertices[m_Divisions].z, 1.0f);
        m_SelfData->m_normalsAligned[m_Divisions]  = Vector4f(m_SelfData->m_normals[m_Divisions].x,
             m_SelfData->m_normals[m_Divisions].y, m_SelfData->m_normals[m_Divisions].z, 1.0f);

        // indices
        for (u32 i = 0; i < m_Divisions; ++i)
        {
            m_SelfData->m_indices[3 * i]     = i;
            m_SelfData->m_indices[3 * i + 1] = (i + 1) % (m_Divisions);
            m_SelfData->m_indices[3 * i + 2] = m_Divisions; // center vertex
        }
    }

    IFRIT_APIDECL Ref<MeshData> Circle2D::LoadMesh()
    {
        if (!m_Loaded)
        {
            BuildMesh();
            m_Loaded = true;
        }
        return m_SelfData;
    }

    IFRIT_APIDECL MeshData* Circle2D::LoadMeshUnsafe()
    {
        if (!m_Loaded)
        {
            BuildMesh();
            m_Loaded = true;
        }
        return m_SelfDataRaw;
    }

    IFRIT_APIDECL u32 Circle2D::GetNumIndices() { return 3 * m_Divisions; }
    IFRIT_APIDECL u32 Circle2D::GetNumVertices() { return m_Divisions + 1; }
    IFRIT_APIDECL Vec<u32> Circle2D::GetIndexBufferHost() { return m_SelfData->m_indices; }
    IFRIT_APIDECL Vec<Vector3f> Circle2D::GetVertexBufferHost() { return m_SelfData->m_vertices; }

    IFRIT_APIDECL Mesh*         Circle2DAsset::GetMesh()
    {
        if (mMesh == nullptr)
        {
            mMesh = MakeOwner<Circle2D>(mRadius, mDivisions);
        }
        return mMesh.get();
    }

} // namespace Ifrit::Runtime::Geometry
