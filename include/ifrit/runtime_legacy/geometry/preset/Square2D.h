#pragma once
#include "ifrit/runtime/base/Mesh.h"
#include "ifrit/runtime/asset/MeshAsset.h"
#include "ifrit/core/reflection/ReflAttrs.h"
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

    class IFRIT_APIDECL IF_CLASS() Square2DAsset : public MeshAsset
    {
    public:
        IF_PROPERTY()
        f32 mWidth = 1.0f;

        IF_PROPERTY()
        f32 mHeight = 1.0f;

    private:
        Owner<Square2D> mMesh = nullptr;

    public:
        Square2DAsset() = default;
        Square2DAsset(f32 width, f32 height) : mWidth(width), mHeight(height) {}

        virtual Mesh* GetMesh() override;
    };
} // namespace Ifrit::Runtime::Geometry
