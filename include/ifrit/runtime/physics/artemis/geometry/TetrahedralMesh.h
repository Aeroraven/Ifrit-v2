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

#pragma once
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/base/Mesh.h"

namespace Ifrit::Runtime::Artemis
{
    struct TetrahedralMeshAttribute
    {
        f32 m_PlaceHolder;

        IFRIT_STRUCT_SERIALIZE(m_PlaceHolder);
    };

    class IFRIT_RUNTIME_API TetrahedralMesh : public Mesh
    {
    private:
        Vec<Vector3f> m_Vertices;
        Vec<u32>      m_Indices;
        Vec<u32>      m_TriangleIndices;
        Vec<u32>      m_SurfaceTriangleIndices;
        Ref<Mesh>     m_TriangularMesh = nullptr;
        Ref<MeshData> m_SelfData;
        bool          m_Loaded = false;

    private:
        void BuildMesh();

    public:
        TetrahedralMesh();
        virtual ~TetrahedralMesh();

        void                  SetTriangularMesh(Ref<Mesh> mesh);

        virtual Ref<MeshData> LoadMesh() override;
        virtual MeshData*     LoadMeshUnsafe() override;
        virtual u32           GetNumIndices() override;
        virtual u32           GetNumVertices() override;
        virtual Vec<u32>      GetIndexBufferHost() override;
        virtual Vec<Vector3f> GetVertexBufferHost() override;

        virtual Vec<u32>      GetSolidMeshIndices() override;
        virtual Vec<Vector3f> GetSolidMeshVertices() override;
        virtual Vec<u32>      GetSurfaceMeshIndices() override;

        virtual Ref<MeshData> GetBaseMesh() override;
    };
} // namespace Ifrit::Runtime::Artemis