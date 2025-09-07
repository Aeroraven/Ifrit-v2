#include "ifrit/runtime/geometry/ProceduralMesh.h"

using namespace Ifrit::RHI;

namespace Ifrit::Runtime::Geometry
{
    Ref<MeshData> ProceduralMesh::LoadMesh()
    {
        if (m_Loaded)
        {
            return m_SelfData;
        }
        m_SelfData                   = MakeRef<MeshData>();
        m_SelfData->m_MeshType       = MeshType::Surface;
        m_SelfData->m_GenerationType = MeshGeneratorType::Procedual;
        m_Loaded                     = true;
        return m_SelfData;
    }

    MeshData* ProceduralMesh::LoadMeshUnsafe()
    {
        if (m_Loaded)
        {
            return m_SelfData.get();
        }
        m_SelfData                   = MakeRef<MeshData>();
        m_SelfData->m_MeshType       = MeshType::Surface;
        m_SelfData->m_GenerationType = MeshGeneratorType::Procedual;
        m_Loaded                     = true;
        return m_SelfData.get();
    }

    u32 ProceduralMesh::GetNumIndices()
    {
        IF_LOG_ASSERTION("ProceduralMesh", false, "");
        return 0;
    }

    u32 ProceduralMesh::GetNumVertices()
    {
        IF_LOG_ASSERTION("ProceduralMesh", false, "");
        return 0;
    }

    Vec<u32> ProceduralMesh::GetIndexBufferHost()
    {
        IF_LOG_ASSERTION("ProceduralMesh", false, "");
        return {};
    }

    Vec<Vector3f> ProceduralMesh::GetVertexBufferHost()
    {
        IF_LOG_ASSERTION("ProceduralMesh", false, "");
        return {};
    }

} // namespace Ifrit::Runtime::Geometry
