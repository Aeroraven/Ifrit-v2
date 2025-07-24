#pragma once
#include "ifrit/runtime/base/Mesh.h"

namespace Ifrit::Runtime::Geometry
{
    class IFRIT_APIDECL Square2D : public Mesh
    {
    private:
        Ref<MeshData> m_SelfData;
        MeshData*     m_SelfDataRaw = nullptr;
        bool          m_Loaded      = false;
        f32           m_Width       = 1.0f;
        f32           m_Height      = 1.0f;

    private:
        void BuildMesh();

    public:
        Square2D() {}
        Square2D(f32 m_Width, f32 m_Height);

        virtual ~Square2D() = default;

        Ref<MeshData>         LoadMesh() override;
        MeshData*             LoadMeshUnsafe() override;
        inline Mesh&          GetMesh() { return *this; }

        virtual u32           GetNumIndices();
        virtual u32           GetNumVertices();
        virtual Vec<u32>      GetIndexBufferHost();
        virtual Vec<Vector3f> GetVertexBufferHost();
    };
} // namespace Ifrit::Runtime::Geometry
