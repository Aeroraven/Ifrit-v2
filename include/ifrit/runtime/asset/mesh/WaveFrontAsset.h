#pragma once
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/runtime/base/Mesh.h"
#include "ifrit/runtime/asset/MeshAsset.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL ImportedWaveFrontMesh : public Mesh
    {
    private:
        Ref<MeshData> m_selfData;
        MeshData*     m_selfDataRaw = nullptr;
        bool          m_loaded      = false;

    public:
        String mResourcePath;

    public:
        Ref<MeshData>         LoadMesh() override;
        MeshData*             LoadMeshUnsafe() override;
        inline Mesh&          GetMesh() { return *this; }

        virtual u32           GetNumIndices();
        virtual u32           GetNumVertices();
        virtual Vec<u32>      GetIndexBufferHost();
        virtual Vec<Vector3f> GetVertexBufferHost();
    };

    class IFRIT_APIDECL IF_CLASS() WaveFrontAsset : public ImportedMeshAsset
    {
    private:
        Owner<ImportedWaveFrontMesh> mImportedMesh = nullptr;

    public:
        virtual Mesh* GetMesh() override;
    };
} // namespace Ifrit::Runtime