#pragma once
#include "ifrit/runtime/base/Mesh.h"
#include "ifrit/runtime/asset/MeshAsset.h"
#include "ifrit/core/reflection/ReflAttrs.h"

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

    class IFRIT_APIDECL IF_CLASS() Circle2DAsset : public MeshAsset
    {
    public:
        IF_PROPERTY()
        f32 mRadius;

        IF_PROPERTY()
        u32 mDivisions;

    private:
        Owner<Circle2D> mMesh = nullptr;

    public:
        Circle2DAsset() = default;
        Circle2DAsset(f32 radius, u32 divisions) : mRadius(radius), mDivisions(divisions) {}

        virtual Mesh* GetMesh() override;
    };
} // namespace Ifrit::Runtime::Geometry
