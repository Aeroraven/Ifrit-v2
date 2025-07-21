#include "ifrit/runtime/geometry/surfaceop/ParticleSurfaceProceduralMesh.h"
#include "ifrit/runtime/geometry/MeshUtility.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/geometry/internal/InternalShaderRegistry.Geometry.h"

#include "ifrit.shader.neo/Meshing/SurfaceRecon/SurfRecon.Common.hlsli"

using namespace Ifrit::RHI;
using namespace Ifrit::Runtime::FrameGraphUtils;
using namespace Ifrit::Math;

namespace Ifrit::Runtime::Geometry
{
    struct FSurfReconGridData
    {
        Vector4f        m_MinBound;
        Vector4f        m_MaxBound;
        Vector4i        m_NumCells;
        int             m_BlockWidthInCellUnits;
        RHI::RhiUAVDesc m_CellParticleCount;  // UAV
        RHI::RhiUAVDesc m_ActiveCellsInBlock; // UAV
        RHI::RhiUAVDesc m_SurfaceVertices;    // UAV
    };

    struct ParticleSurfaceProceduralMeshPrivateData
    {
        ParticleSurfaceProceduralMesh* m_Parent = nullptr;
        FSurfReconGridData             m_GridData;

        // External
        RHI::RhiBufferRef              m_ParticleSrcDataBuffer  = nullptr;
        RHI::RhiBufferRef              m_ParticleSrcCountBuffer = nullptr;

        // Managed
        bool                           m_ResourcePrepared         = false;
        RHI::RhiBufferRef              m_ParticleDataCopyDispArgs = nullptr;
        RHI::RhiBufferRef              m_CellParticleCount        = nullptr;
        RHI::RhiBufferRef              m_ActiveCellsInBlock       = nullptr;
        RHI::RhiBufferRef              m_SurfaceVertices          = nullptr;
        RHI::RhiBufferRef              m_GridDataBuffer           = nullptr;
        RHI::RhiBufferRef              m_DebugDataBuffer          = nullptr;

        // RDG
        FGBufferNodeRef                m_RDGParticleSrcDataBuffer;
        FGBufferNodeRef                m_RDGParticleSrcCountBuffer;
        FGBufferNodeRef                m_RDGParticleDataCopyDispArgs;

        FGBufferNodeRef                m_RDGCellParticleCount;
        FGBufferNodeRef                m_RDGActiveCellsInBlock;
        FGBufferNodeRef                m_RDGSurfaceVertices;
        FGBufferNodeRef                m_RDGGridData;

        FGBufferNodeRef                m_RDGMeshVertexBuffer;
        FGBufferNodeRef                m_RDGMeshNormalBuffer;
        FGBufferNodeRef                m_RDGMeshTangentBuffer;
        FGBufferNodeRef                m_RDGMeshUVBuffer;
        FGBufferNodeRef                m_RDGMeshIndexBuffer;
        FGBufferNodeRef                m_RDGMeshIndirectDrawArgs;

        FGBufferNodeRef                m_RDGDebugDataBuffer;

        // RDG Managed
        FGBufferNodeRef                m_RDGActiveBlockList;
        FGBufferNodeRef                m_RDGActiveBlockCounter;
        FGBufferNodeRef                m_RDGTriangleCounter; // Actually num indices

        // Temp
        int                            m_NumBlocks = 0;
        Vector3i                       m_NumBlocksPerAxis;

        void                           PrepareRDGResources(FrameGraphBuilder& builder);
        void                           PrepareGridBuildDispArgs(FrameGraphBuilder& builder);
        void                           ResetGrids(FrameGraphBuilder& builder);
        void                           GridBuild(FrameGraphBuilder& builder);
        void                           FilterBlocks(FrameGraphBuilder& builder);
        void                           FilterCells(FrameGraphBuilder& builder);
        void                           VoxelMeshing(FrameGraphBuilder& builder);
        void                           PrepareDrawArgs(FrameGraphBuilder& builder);
    };

    void ParticleSurfaceProceduralMeshPrivateData::PrepareRDGResources(FrameGraphBuilder& builder)
    {
        m_RDGParticleSrcDataBuffer = &builder.ImportBuffer("RDG.ParticleSrcDataBuffer", m_ParticleSrcDataBuffer.get());
        m_RDGParticleSrcCountBuffer =
            &builder.ImportBuffer("RDG.ParticleSrcCountBuffer", m_ParticleSrcCountBuffer.get());
        m_RDGParticleDataCopyDispArgs =
            &builder.ImportBuffer("RDG.ParticleDataCopyDispArgs", m_ParticleDataCopyDispArgs.get());

        m_RDGMeshVertexBuffer = &builder.ImportBuffer(
            "RDG.ParticleSurfaceProceduralMesh.VertexBuffer", m_Parent->m_resource.vertexBuffer.get());
        m_RDGMeshNormalBuffer = &builder.ImportBuffer(
            "RDG.ParticleSurfaceProceduralMesh.NormalBuffer", m_Parent->m_resource.normalBuffer.get());
        m_RDGMeshTangentBuffer = &builder.ImportBuffer(
            "RDG.ParticleSurfaceProceduralMesh.TangentBuffer", m_Parent->m_resource.tangentBuffer.get());
        m_RDGMeshUVBuffer =
            &builder.ImportBuffer("RDG.ParticleSurfaceProceduralMesh.UVBuffer", m_Parent->m_resource.uvBuffer.get());
        m_RDGMeshIndexBuffer = &builder.ImportBuffer(
            "RDG.ParticleSurfaceProceduralMesh.IndexBuffer", m_Parent->m_resource.indexBuffer.get());
        m_RDGMeshIndirectDrawArgs = &builder.ImportBuffer(
            "RDG.ParticleSurfaceProceduralMesh.IndirectDrawArgs", m_Parent->m_resource.procIndirectDrawBuffer.get());

        m_RDGCellParticleCount =
            &builder.ImportBuffer("RDG.ParticleSurfaceProceduralMesh.CellParticleCount", m_CellParticleCount.get());
        m_RDGActiveCellsInBlock =
            &builder.ImportBuffer("RDG.ParticleSurfaceProceduralMesh.ActiveCellsInBlock", m_ActiveCellsInBlock.get());
        m_RDGSurfaceVertices =
            &builder.ImportBuffer("RDG.ParticleSurfaceProceduralMesh.SurfaceVertices", m_SurfaceVertices.get());

        m_RDGGridData = &builder.ImportBuffer("RDG.ParticleSurfaceProceduralMesh.GridData", m_GridDataBuffer.get());

        m_RDGDebugDataBuffer =
            &builder.ImportBuffer("RDG.ParticleSurfaceProceduralMesh.DebugData", m_DebugDataBuffer.get());

        m_RDGActiveBlockList    = &builder.DeclareBuffer("RDG.ParticleSurfaceProceduralMesh.ActiveBlockList",
               FrameGraphBufferDesc(sizeof(u32) * m_NumBlocks, RhiBufferUsage_CopyDst | RhiBufferUsage_SSBO));
        m_RDGActiveBlockCounter = &builder.DeclareBuffer("RDG.ParticleSurfaceProceduralMesh.ActiveBlockCounter",
            FrameGraphBufferDesc(
                sizeof(u32) * 3, RhiBufferUsage_CopyDst | RhiBufferUsage_SSBO | RhiBufferUsage_Indirect));
        m_RDGTriangleCounter    = &builder.DeclareBuffer("RDG.ParticleSurfaceProceduralMesh.TriangleCounter",
               FrameGraphBufferDesc(sizeof(u32), RhiBufferUsage_CopyDst | RhiBufferUsage_SSBO));
    }
    void ParticleSurfaceProceduralMeshPrivateData::ResetGrids(FrameGraphBuilder& builder)
    {
        AddClearUAVPass(builder, "ParticleSurfaceProceduralMesh.ResetGrids", *m_RDGCellParticleCount, 0);
        AddClearUAVPass(builder, "ParticleSurfaceProceduralMesh.ResetActiveCellsInBlock", *m_RDGActiveCellsInBlock, 0);
        AddClearUAVPass(builder, "ParticleSurfaceProceduralMesh.ResetSurfaceVertices", *m_RDGSurfaceVertices, 0);
    }
    void ParticleSurfaceProceduralMeshPrivateData::PrepareGridBuildDispArgs(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_ParticleCounterSrc; // SRV!
            u32 m_ParticleCounterDst; // UAV!
        } pc{};

        AddComputePass<PushConst>(builder, "ParticleSurfaceProceduralMesh.PrepareDispArgs",
            ShaderVariantDesc(Internal::InternalShaderTableGeometry::SurfReconPrepareDispArgsCS, {}), Vector3i(1, 1, 1),
            pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticleCounterSrc = ctx.m_FgDesc->GetSRV(*m_RDGParticleSrcCountBuffer);
                pc.m_ParticleCounterDst = ctx.m_FgDesc->GetUAV(*m_RDGParticleDataCopyDispArgs);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGParticleSrcCountBuffer)
            .AddWriteResource(*m_RDGParticleDataCopyDispArgs);
    }

    void ParticleSurfaceProceduralMeshPrivateData::GridBuild(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_ParticleCounter;
            u32 m_ParticleCounterIndirect; // SRV
            u32 m_ParticleLocation;        // SRV
            u32 m_Grid;                    // SRV
            u32 m_DebugData;               // UAV
        } pc;

        pc.m_ParticleCounter = ~0u;

        AddIndirectComputePass<PushConst>(builder, "ParticleSurfaceProceduralMesh.GridBuild",
            ShaderVariantDesc(Internal::kIntShaderTableGeometry.SurfReconGridBuildCS, {}),
            *m_RDGParticleDataCopyDispArgs, sizeof(u32), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticleCounterIndirect = ctx.m_FgDesc->GetSRV(*m_RDGParticleDataCopyDispArgs);
                pc.m_ParticleLocation        = ctx.m_FgDesc->GetSRV(*m_RDGParticleSrcDataBuffer);
                pc.m_Grid                    = ctx.m_FgDesc->GetSRV(*m_RDGGridData);
                pc.m_DebugData               = ctx.m_FgDesc->GetUAV(*m_RDGDebugDataBuffer);

                SetRootConstant(pc, ctx);
            })
            .AddReadWriteResource(*m_RDGCellParticleCount)
            .AddReadWriteResource(*m_RDGActiveCellsInBlock)
            .AddReadResource(*m_RDGParticleSrcDataBuffer)
            .AddReadResource(*m_RDGParticleDataCopyDispArgs)
            .AddWriteResource(*m_RDGSurfaceVertices)
            .AddReadResource(*m_RDGGridData);
    }

    void ParticleSurfaceProceduralMeshPrivateData::FilterBlocks(FrameGraphBuilder& builder)
    {
        AddClearUAVPass(builder, "ParticleSurfaceProceduralMesh.ClearActiveBlockCounter", *m_RDGActiveBlockCounter, 0);

        struct PushConst
        {
            RHI::RhiUAVDesc m_ActiveGridBlockCounter;
            RHI::RhiUAVDesc m_ActiveGridBlockList;
            RHI::RhiSRVDesc m_Grid; // SRV
        } pc{};

        Vector3i dispatchArgs = m_NumBlocksPerAxis;
        dispatchArgs.x = DivRoundUp(dispatchArgs.x, IfritShader::Meshing::SurfRecon::kSurfReconFilterBlockTGSz3D);
        dispatchArgs.y = DivRoundUp(dispatchArgs.y, IfritShader::Meshing::SurfRecon::kSurfReconFilterBlockTGSz3D);
        dispatchArgs.z = DivRoundUp(dispatchArgs.z, IfritShader::Meshing::SurfRecon::kSurfReconFilterBlockTGSz3D);

        AddComputePass<PushConst>(builder, "ParticleSurfaceProceduralMesh.FilterBlocks",
            ShaderVariantDesc(Internal::kIntShaderTableGeometry.SurfReconFilterBlocksCS, {}), dispatchArgs, pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ActiveGridBlockCounter = ctx.m_FgDesc->GetUAV(*m_RDGActiveBlockCounter);
                pc.m_ActiveGridBlockList    = ctx.m_FgDesc->GetUAV(*m_RDGActiveBlockList);
                pc.m_Grid                   = ctx.m_FgDesc->GetSRV(*m_RDGGridData);

                SetRootConstant(pc, ctx);
            })
            .AddReadWriteResource(*m_RDGActiveBlockCounter)
            .AddReadWriteResource(*m_RDGActiveBlockList)
            .AddReadResource(*m_RDGActiveCellsInBlock)
            .AddReadResource(*m_RDGGridData);
    }

    void ParticleSurfaceProceduralMeshPrivateData::FilterCells(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            RHI::RhiSRVDesc m_ActiveGridBlockList;
            RHI::RhiSRVDesc m_Grid; // SRV
        } pc{};

        AddIndirectComputePass<PushConst>(builder, "ParticleSurfaceProceduralMesh.FilterCells",
            ShaderVariantDesc(Internal::kIntShaderTableGeometry.SurfReconFilterCellsCS, {}), *m_RDGActiveBlockCounter,
            0, pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ActiveGridBlockList = ctx.m_FgDesc->GetSRV(*m_RDGActiveBlockList);
                pc.m_Grid                = ctx.m_FgDesc->GetSRV(*m_RDGGridData);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGCellParticleCount)
            .AddReadWriteResource(*m_RDGSurfaceVertices)
            .AddReadResource(*m_RDGActiveCellsInBlock)
            .AddReadResource(*m_RDGActiveBlockList)
            .AddReadResource(*m_RDGGridData);
    }

    void ParticleSurfaceProceduralMeshPrivateData::VoxelMeshing(FrameGraphBuilder& builder)
    {
        AddClearUAVPass(builder, "ParticleSurfaceProceduralMesh.ClearTriangleCounter", *m_RDGTriangleCounter, 0);

        struct PushConst
        {
            RHI::RhiSRVDesc m_ActiveGridBlockList;
            RHI::RhiSRVDesc m_Grid;
            RHI::RhiUAVDesc m_IndexBuffer;
            RHI::RhiUAVDesc m_VertexBuffer;
            RHI::RhiUAVDesc m_NormalBuffer;
            RHI::RhiUAVDesc m_TangentBuffer;
            RHI::RhiUAVDesc m_UVBuffer;

            RHI::RhiUAVDesc m_TriangleCounter;
        } pc{};

        AddIndirectComputePass<PushConst>(builder, "ParticleSurfaceProceduralMesh.VoxelMeshing",
            ShaderVariantDesc(Internal::kIntShaderTableGeometry.SurfReconVoxelMeshingCS, {}), *m_RDGActiveBlockCounter,
            0, pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ActiveGridBlockList = ctx.m_FgDesc->GetSRV(*m_RDGActiveBlockList);
                pc.m_Grid                = ctx.m_FgDesc->GetSRV(*m_RDGGridData);
                pc.m_IndexBuffer         = ctx.m_FgDesc->GetUAV(*m_RDGMeshIndexBuffer);
                pc.m_VertexBuffer        = ctx.m_FgDesc->GetUAV(*m_RDGMeshVertexBuffer);
                pc.m_NormalBuffer        = ctx.m_FgDesc->GetUAV(*m_RDGMeshNormalBuffer);
                pc.m_TangentBuffer       = ctx.m_FgDesc->GetUAV(*m_RDGMeshTangentBuffer);
                pc.m_UVBuffer            = ctx.m_FgDesc->GetUAV(*m_RDGMeshUVBuffer);
                pc.m_TriangleCounter     = ctx.m_FgDesc->GetUAV(*m_RDGTriangleCounter);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGActiveBlockList)
            .AddReadResource(*m_RDGGridData)
            .AddReadResource(*m_RDGCellParticleCount)
            .AddReadResource(*m_RDGSurfaceVertices)
            .AddReadWriteResource(*m_RDGMeshIndexBuffer)
            .AddReadWriteResource(*m_RDGMeshVertexBuffer)
            .AddReadWriteResource(*m_RDGMeshNormalBuffer)
            .AddReadWriteResource(*m_RDGMeshTangentBuffer)
            .AddReadWriteResource(*m_RDGMeshUVBuffer)
            .AddReadWriteResource(*m_RDGTriangleCounter);
    }

    void ParticleSurfaceProceduralMeshPrivateData::PrepareDrawArgs(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            RHI::RhiSRVDesc m_TriangleCounter;
            RHI::RhiUAVDesc m_TargetIndirectIndexedDrawArgs;
        } pc{};

        AddComputePass<PushConst>(builder, "ParticleSurfaceProceduralMesh.PrepareDrawArgs",
            ShaderVariantDesc(Internal::kIntShaderTableGeometry.SurfReconPrepareDrawArgsCS, {}), Vector3i(1, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_TriangleCounter               = ctx.m_FgDesc->GetSRV(*m_RDGTriangleCounter);
                pc.m_TargetIndirectIndexedDrawArgs = ctx.m_FgDesc->GetUAV(*m_RDGMeshIndirectDrawArgs);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGTriangleCounter)
            .AddWriteResource(*m_RDGMeshIndirectDrawArgs);
    }

    IFRIT_APIDECL void ParticleSurfaceProceduralMesh::UpdateMesh(FrameGraphBuilder& builder)
    {
        if (m_Data->m_ParticleSrcDataBuffer == nullptr)
        {
            return;
        }

        IFRIT_FRAMEGRAPH_EVENT_SCOPE(builder, "ParticleSurfaceProceduralMesh.UpdateMesh");
        m_Data->PrepareRDGResources(builder);
        m_Data->ResetGrids(builder);
        m_Data->PrepareGridBuildDispArgs(builder);
        m_Data->GridBuild(builder);
        m_Data->FilterBlocks(builder);
        m_Data->FilterCells(builder);
        m_Data->VoxelMeshing(builder);
        m_Data->PrepareDrawArgs(builder);
    }

    IFRIT_APIDECL void ParticleSurfaceProceduralMesh::SetParticleData(
        RHI::RhiBufferRef particleDataBuffer, RHI::RhiBufferRef particleCount)
    {
        m_Data->m_ParticleSrcDataBuffer  = particleDataBuffer;
        m_Data->m_ParticleSrcCountBuffer = particleCount;
    }

    IFRIT_APIDECL void ParticleSurfaceProceduralMesh::Init(
        RHI::RhiBackend* rhi, u32 maxParticles, u32 maxIndices, Vector4i gridSize, Vector3f minBound, Vector3f maxBound)
    {
        AllocateMeshGPUResources(rhi, this, maxParticles, maxIndices);
        ForceMeshObjectBufferSync(rhi, this);

        auto debugSz       = SizeCast<u32>(maxParticles * sizeof(u32) * 16);
        auto indirectArgSz = SizeCast<u32>(sizeof(u32) * 4);
        auto indirectUsage = RhiBufferUsage_CopyDst | RhiBufferUsage_Indirect | RhiBufferUsage_SSBO;

        m_Data->m_GridData.m_MinBound              = Vector4f(minBound, 0.0f);
        m_Data->m_GridData.m_MaxBound              = Vector4f(maxBound, 0.0f);
        m_Data->m_GridData.m_NumCells              = gridSize;
        m_Data->m_GridData.m_BlockWidthInCellUnits = 4;

        auto totalCells = gridSize.x * gridSize.y * gridSize.z;
        auto totalGrids = DivRoundUp(gridSize.x, m_Data->m_GridData.m_BlockWidthInCellUnits)
            * DivRoundUp(gridSize.y, m_Data->m_GridData.m_BlockWidthInCellUnits)
            * DivRoundUp(gridSize.z, m_Data->m_GridData.m_BlockWidthInCellUnits);
        auto totalVertexCells = (gridSize.x + 1) * (gridSize.y + 1) * (gridSize.z + 1);
        auto gridDataSz       = SizeCast<u32>(sizeof(FSurfReconGridData));

        m_Data->m_NumBlocks        = totalGrids;
        m_Data->m_NumBlocksPerAxis = Vector3i(DivRoundUp(gridSize.x, m_Data->m_GridData.m_BlockWidthInCellUnits),
            DivRoundUp(gridSize.y, m_Data->m_GridData.m_BlockWidthInCellUnits),
            DivRoundUp(gridSize.z, m_Data->m_GridData.m_BlockWidthInCellUnits));

        auto cellParticleCountSz  = SizeCast<u32>(totalCells * sizeof(u32));
        auto activeCellsInBlockSz = SizeCast<u32>(totalGrids * sizeof(u32));
        auto surfaceVertexSz      = SizeCast<u32>(totalVertexCells * sizeof(u32));
        auto uavUsage             = RhiBufferUsage_CopyDst | RhiBufferUsage_SSBO;

        m_Data->m_ParticleDataCopyDispArgs =
            rhi->CreateBufferDevice("ParticleDataCopyDispArgs", indirectArgSz, indirectUsage, true);
        m_Data->m_CellParticleCount = rhi->CreateBufferDevice(
            "ParticleSurfaceProceduralMesh.CellParticleCount", cellParticleCountSz, uavUsage, true);
        m_Data->m_ActiveCellsInBlock = rhi->CreateBufferDevice(
            "ParticleSurfaceProceduralMesh.ActiveCellsInBlock", activeCellsInBlockSz, uavUsage, true);
        m_Data->m_SurfaceVertices =
            rhi->CreateBufferDevice("ParticleSurfaceProceduralMesh.SurfaceVertices", surfaceVertexSz, uavUsage, true);
        m_Data->m_GridDataBuffer =
            rhi->CreateBufferDevice("ParticleSurfaceProceduralMesh.GridDataBuffer", gridDataSz, uavUsage, true);
        m_Data->m_DebugDataBuffer =
            rhi->CreateBufferDevice("ParticleSurfaceProceduralMesh.DebugDataBuffer", debugSz, uavUsage, true);

        m_Data->m_GridData.m_CellParticleCount  = rhi->GetUAVDescriptor(m_Data->m_CellParticleCount.get());
        m_Data->m_GridData.m_ActiveCellsInBlock = rhi->GetUAVDescriptor(m_Data->m_ActiveCellsInBlock.get());
        m_Data->m_GridData.m_SurfaceVertices    = rhi->GetUAVDescriptor(m_Data->m_SurfaceVertices.get());

        auto tq                   = rhi->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
        auto stagedGridDataBuffer = rhi->CreateStagedSingleBuffer(m_Data->m_GridDataBuffer.get());
        tq->RunSyncCommand([&](const RhiCommandList* cmd) {
            stagedGridDataBuffer->CmdCopyToDevice(cmd, &m_Data->m_GridData, sizeof(FSurfReconGridData), 0);
        });
    }

    IFRIT_APIDECL ParticleSurfaceProceduralMesh::ParticleSurfaceProceduralMesh()
    {
        m_Data           = new ParticleSurfaceProceduralMeshPrivateData();
        m_Data->m_Parent = this;

        m_resourceDirty = false;
    }

    IFRIT_APIDECL ParticleSurfaceProceduralMesh::~ParticleSurfaceProceduralMesh()
    {
        if (m_Data)
        {
            delete m_Data;
            m_Data = nullptr;
        }
    }

} // namespace Ifrit::Runtime::Geometry
