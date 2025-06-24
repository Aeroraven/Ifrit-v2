/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "ifrit/runtime/physics/siro/TetrahedralMesh.h"

namespace Ifrit::Runtime::Siro
{
    IFRIT_APIDECL TetrahedralMesh::TetrahedralMesh() : Mesh(), m_TriangularMesh(nullptr)
    {
        m_SelfData             = std::make_shared<MeshData>();
        m_SelfData->m_MeshType = MeshType::Tetrahedral;
    }

    IFRIT_APIDECL TetrahedralMesh::~TetrahedralMesh()
    {
        if (m_TriangularMesh)
        {
            delete m_TriangularMesh;
            m_TriangularMesh = nullptr;
        }
    }

    IFRIT_APIDECL void TetrahedralMesh::SetTriangularMesh(Mesh* mesh) { m_TriangularMesh = mesh; }

    IFRIT_APIDECL Ref<MeshData> TetrahedralMesh::LoadMesh()
    {
        if (m_Loaded)
        {
            return m_SelfData;
        }
        BuildMesh();
        m_Loaded = true;
        return m_SelfData;
    }

    IFRIT_APIDECL MeshData* TetrahedralMesh::LoadMeshUnsafe()
    {
        if (m_Loaded)
        {
            return m_SelfData.get();
        }
        BuildMesh();
        m_Loaded = true;
        return m_SelfData.get();
    }

    IFRIT_APIDECL void TetrahedralMesh::BuildMesh()
    {
        // TODO
    }

    IFRIT_APIDECL u32 TetrahedralMesh::GetNumIndices() { return static_cast<u32>(m_TriangleIndices.size()); }
    IFRIT_APIDECL u32 TetrahedralMesh::GetNumVertices() { return static_cast<u32>(m_Vertices.size()); }
    IFRIT_APIDECL Vec<u32> TetrahedralMesh::GetIndexBufferHost()
    {
        return m_TriangleIndices; // Return the triangle indices
    }
    IFRIT_APIDECL Vec<Vector3f> TetrahedralMesh::GetVertexBufferHost()
    {
        return m_Vertices; // Return the vertex buffer
    }

    IFRIT_APIDECL Vec<u32> TetrahedralMesh::GetTetrahedronIndicesHost()
    {
        return m_Indices; // Return the tetrahedron indices
    }

} // namespace Ifrit::Runtime::Siro