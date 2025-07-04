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
#include "ifrit/runtime/physics/siro/geometry/TetrahedralMesh.h"
#include "ifrit/geomproc/tetrahedralization/MeshTetrahedralizer.h"
#include "ifrit/core/algo/OrderedPairs.h"

using namespace Ifrit::GeometryProc::Tetrahedralization;
using namespace Ifrit::GeometryProc;
using namespace Ifrit::Math;

namespace Ifrit::Runtime::Siro
{
    IFRIT_APIDECL TetrahedralMesh::TetrahedralMesh() : Mesh(), m_TriangularMesh(nullptr)
    {
        m_SelfData             = std::make_shared<MeshData>();
        m_SelfData->m_MeshType = MeshType::Solid;
    }

    IFRIT_APIDECL      TetrahedralMesh::~TetrahedralMesh() {}

    IFRIT_APIDECL void TetrahedralMesh::SetTriangularMesh(Ref<Mesh> mesh) { m_TriangularMesh = mesh; }

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
        auto meshData = m_TriangularMesh->LoadMesh();
        iAssertion(meshData != nullptr, "Mesh data is null");
        iAssertion(meshData->m_MeshType == MeshType::Surface || meshData->m_MeshType == MeshType::VirtualGeometry,
            "Mesh type must be Surface");

        auto           indexBuffer  = m_TriangularMesh->GetIndexBufferHost();
        auto           vertexBuffer = m_TriangularMesh->GetVertexBufferHost();

        MeshDescriptor meshDesc;
        meshDesc.vertexData     = reinterpret_cast<i8*>(vertexBuffer.data());
        meshDesc.indexData      = reinterpret_cast<i8*>(indexBuffer.data());
        meshDesc.vertexCount    = static_cast<i32>(vertexBuffer.size());
        meshDesc.indexCount     = static_cast<i32>(indexBuffer.size());
        meshDesc.vertexStride   = sizeof(Vector3f);
        meshDesc.positionOffset = 0;

        auto tetraMesh = TetrahedralizeMesh(meshDesc);
        m_Indices      = tetraMesh.m_Indices;
        m_Vertices     = tetraMesh.m_Vertices;

        HashMap<OrderedPair3, u32> facets;
        for (u32 i = 0; i < m_Indices.size(); i += 4)
        {
            u32          pA = m_Indices[i];
            u32          pB = m_Indices[i + 1];
            u32          pC = m_Indices[i + 2];
            u32          pD = m_Indices[i + 3];

            OrderedPair3 face1(pA, pB, pC);
            facets[face1] = facets.find(face1) != facets.end() ? facets[face1] + 1 : 1;
            OrderedPair3 face2(pA, pD, pB);
            facets[face2] = facets.find(face2) != facets.end() ? facets[face2] + 1 : 1;
            OrderedPair3 face3(pA, pC, pD);
            facets[face3] = facets.find(face3) != facets.end() ? facets[face3] + 1 : 1;
            OrderedPair3 face4(pC, pB, pD);
            facets[face4] = facets.find(face4) != facets.end() ? facets[face4] + 1 : 1;
        }

        m_TriangleIndices.clear();
        m_SurfaceTriangleIndices.clear();
        for (const auto& [face, count] : facets)
        {
            if (count == 1) // This is a boundary face
            {
                // Add to surface triangles
                m_SurfaceTriangleIndices.push_back(face.ra);
                m_SurfaceTriangleIndices.push_back(face.rb);
                m_SurfaceTriangleIndices.push_back(face.rc);
            }
            m_TriangleIndices.push_back(face.ra);
            m_TriangleIndices.push_back(face.rb);
            m_TriangleIndices.push_back(face.rc);
        }

        Vec<Vector3f> vertexNormals(m_Vertices.size(), Vector3f(0.0f, 0.0f, 0.0f));
        for (u32 i = 0; i < m_SurfaceTriangleIndices.size(); i += 3)
        {
            u32      a = m_SurfaceTriangleIndices[i];
            u32      b = m_SurfaceTriangleIndices[i + 1];
            u32      c = m_SurfaceTriangleIndices[i + 2];

            Vector3f edge1  = m_Vertices[b] - m_Vertices[a];
            Vector3f edge2  = m_Vertices[c] - m_Vertices[a];
            Vector3f normal = Normalize(Cross(edge1, edge2));

            vertexNormals[a] += normal;
            vertexNormals[b] += normal;
            vertexNormals[c] += normal;
        }
        for (auto& normal : vertexNormals)
        {
            normal = Normalize(normal);
        }

        m_SelfData->m_vertices = m_Vertices;
        m_SelfData->m_indices  = m_SurfaceTriangleIndices;

        for (u32 i = 0; i < m_Vertices.size(); i++)
        {
            m_SelfData->m_verticesAligned.push_back(Vector4f(m_Vertices[i].x, m_Vertices[i].y, m_Vertices[i].z, 1.0f));
            m_SelfData->m_uvs.push_back(Vector2f(0.0f, 0.0f)); // Placeholder UVs
            m_SelfData->m_normals.push_back(vertexNormals[i]);
            m_SelfData->m_normalsAligned.push_back(Vector4f(vertexNormals[i], 0.0f)); // Placeholder normals
            m_SelfData->m_tangents.push_back(Vector4f(1.0f, 0.0f, 0.0f, 1.0f));       // Placeholder tangents
        }
    }

    IFRIT_APIDECL u32 TetrahedralMesh::GetNumIndices() { return static_cast<u32>(m_TriangleIndices.size()); }
    IFRIT_APIDECL u32 TetrahedralMesh::GetNumVertices() { return static_cast<u32>(m_Vertices.size()); }
    IFRIT_APIDECL Vec<u32> TetrahedralMesh::GetIndexBufferHost() { return m_TriangleIndices; }
    IFRIT_APIDECL Vec<Vector3f> TetrahedralMesh::GetVertexBufferHost() { return m_Vertices; }
    IFRIT_APIDECL Vec<u32> TetrahedralMesh::GetSolidMeshIndices() { return m_Indices; }
    IFRIT_APIDECL Vec<Vector3f> TetrahedralMesh::GetSolidMeshVertices() { return m_Vertices; }
    IFRIT_APIDECL Vec<u32> TetrahedralMesh::GetSurfaceMeshIndices() { return m_SurfaceTriangleIndices; }

    IFRIT_APIDECL Ref<MeshData> TetrahedralMesh::GetBaseMesh() { return m_TriangularMesh->LoadMesh(); }

} // namespace Ifrit::Runtime::Siro