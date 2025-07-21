#pragma once
#include "ifrit/runtime/base/Mesh.h"

namespace Ifrit::Runtime::Geometry
{
    class IFRIT_APIDECL Circle2D : public Mesh
    {
    private:
        Ref<MeshData> m_SelfData;
        MeshData*     m_SelfDataRaw = nullptr;
        bool          m_Loaded      = false;
        f32           m_Radius      = 1.0f;
        u32           m_Divisions   = 32;

    private:
        void BuildMesh();

    public:
        Circle2D() {}
        Circle2D(f32 radius, u32 divisions);

        virtual ~Circle2D() = default;

        Ref<MeshData>         LoadMesh() override;
        MeshData*             LoadMeshUnsafe() override;
        inline Mesh&          GetMesh() { return *this; }

        virtual u32           GetNumIndices();
        virtual u32           GetNumVertices();
        virtual Vec<u32>      GetIndexBufferHost();
        virtual Vec<Vector3f> GetVertexBufferHost();
    };
} // namespace Ifrit::Runtime::Geometry