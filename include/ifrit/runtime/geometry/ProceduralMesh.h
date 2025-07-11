
#pragma once
#include "ifrit/runtime/assetmanager/Asset.h"
#include "ifrit/runtime/base/Mesh.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraph.h"

namespace Ifrit::Runtime::Geometry
{
    class IFRIT_RUNTIME_API ProceduralMesh : public Mesh
    {
    private:
        Ref<MeshData> m_SelfData = nullptr;
        bool          m_Loaded   = false;

    public:
        virtual void          UpdateMesh(FrameGraphBuilder& builder) = 0;
        Ref<MeshData>         LoadMesh() override;
        MeshData*             LoadMeshUnsafe() override;
        inline Mesh&          GetMesh() { return *this; }

        virtual u32           GetNumIndices() final;
        virtual u32           GetNumVertices() final;
        virtual Vec<u32>      GetIndexBufferHost() final;
        virtual Vec<Vector3f> GetVertexBufferHost() final;
    };
} // namespace Ifrit::Runtime::Geometry