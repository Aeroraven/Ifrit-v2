#include "ifrit/runtime/geometry/MeshUtility.h"

using namespace Ifrit::RHI;
namespace Ifrit::Runtime::Geometry
{
    IFRIT_APIDECL void AllocateMeshGPUResources(RHI::RhiBackend* rhi, Mesh* mesh, u32 maxVertices, u32 maxIndices)
    {
        // TODO
        auto meshDataRef = mesh->LoadMesh();
        iAssertion(meshDataRef != nullptr, "Mesh data is null");
        iAssertion(meshDataRef->m_GenerationType != MeshGeneratorType::Static,
            "Static mesh generation is managed by the system");

        Mesh::GPUResource meshResource;
        mesh->GetGPUResource(meshResource);
        iAssertion(meshResource.objectBuffer == nullptr, "Mesh GPU resource has already been allocated.");
        iAssertion(meshDataRef->m_MeshType == MeshType::Surface, "Surface mesh is required.");

        auto defaultUsage  = RhiBufferUsage_CopyDst | RhiBufferUsage_SSBO;
        auto indexUsage    = defaultUsage | RhiBufferUsage_Index;
        auto indirectUsage = defaultUsage | RhiBufferUsage_Indirect;

        auto reqVxSize      = SizeCast<u32>(sizeof(Vector4f) * maxVertices);
        auto reqIxSize      = SizeCast<u32>(sizeof(u32) * maxIndices);
        auto reqNormalSize  = SizeCast<u32>(sizeof(Vector4f) * maxVertices);
        auto reqTangentSize = SizeCast<u32>(sizeof(Vector4f) * maxVertices);
        auto reqUvSize      = SizeCast<u32>(sizeof(Vector2f) * maxVertices);
        auto reqIndSize     = SizeCast<u32>(sizeof(u32) * 5);

        meshResource.vertexBuffer  = rhi->CreateBufferDevice("Mesh_Vertex_Proc", reqVxSize, defaultUsage, true);
        meshResource.normalBuffer  = rhi->CreateBufferDevice("Mesh_Normal_Proc", reqNormalSize, defaultUsage, true);
        meshResource.uvBuffer      = rhi->CreateBufferDevice("Mesh_UV_Proc", reqUvSize, defaultUsage, true);
        meshResource.tangentBuffer = rhi->CreateBufferDevice("Mesh_Tangent_Proc", reqTangentSize, defaultUsage, true);
        meshResource.indexBuffer   = rhi->CreateBufferDevice("Mesh_Index_Proc", reqIxSize, indexUsage, true);
        meshResource.procIndirectDrawBuffer =
            rhi->CreateBufferDevice("Mesh_IndirectDraw_Proc", reqIndSize, indirectUsage, true);

        Mesh::GPUObjectBuffer& objectBuffer   = meshResource.objectData;
        objectBuffer.vertexBufferId           = rhi->GetUAVDescriptor(meshResource.vertexBuffer.get());
        objectBuffer.normalBufferId           = rhi->GetUAVDescriptor(meshResource.normalBuffer.get());
        objectBuffer.uvBufferId               = rhi->GetUAVDescriptor(meshResource.uvBuffer.get());
        objectBuffer.tangentBufferId          = rhi->GetUAVDescriptor(meshResource.tangentBuffer.get());
        objectBuffer.indexBufferId            = rhi->GetUAVDescriptor(meshResource.indexBuffer.get());
        objectBuffer.bvhNodeBufferId          = ~0u;
        objectBuffer.clusterGroupBufferId     = ~0u;
        objectBuffer.meshletBufferId          = ~0u;
        objectBuffer.meshletVertexBufferId    = ~0u;
        objectBuffer.meshletIndexBufferId     = ~0u;
        objectBuffer.meshletInClusterBufferId = ~0u;
        objectBuffer.cpCounterBufferId        = ~0u;

        meshResource.objectBuffer =
            rhi->CreateBufferDevice("ObjectBuffer", sizeof(Mesh::GPUObjectBuffer), defaultUsage, true);
        mesh->SetGPUResource(meshResource);
    }

    IFRIT_APIDECL void ForceMeshObjectBufferSync(RHI::RhiBackend* rhi, Mesh* mesh)
    {
        Mesh::GPUResource meshResource;
        mesh->GetGPUResource(meshResource);

        auto tq                 = rhi->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
        auto stagedObjectBuffer = rhi->CreateStagedSingleBuffer(meshResource.objectBuffer.get());
        tq->RunSyncCommand([&](const RhiCommandList* cmd) {
            stagedObjectBuffer->CmdCopyToDevice(cmd, &meshResource.objectData, sizeof(Mesh::GPUObjectBuffer), 0);
        });
    }

} // namespace Ifrit::Runtime::Geometry