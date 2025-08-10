#pragma once
#include "ifrit.shader.neo/Common.hlsli"

#ifndef __cplusplus
#include "ifrit.shader.neo/Bindless.hlsli"
#include "ifrit.shader.neo/Math.LinAlg.hlsli"
#endif

namespace IfritShader{
namespace Meshing{
namespace SurfRecon{

    IFSHADER_DEFINE_CONST_UINT32(kSurfReconGridBuildTGSz, 128);
    IFSHADER_DEFINE_CONST_UINT32(kSurfReconVertexCompactTGSz, 128);
    IFSHADER_DEFINE_CONST_UINT32(kSurfReconVertexDensityTGSz, 128);

    IFSHADER_DEFINE_CONST_UINT32(kSurfReconFilterBlockTGSz3D, 4);
    IFSHADER_DEFINE_CONST_UINT32(kSurfReconFilterBlockTGSz2D, 8);

    IFSHADER_DEFINE_CONST_UINT32(kSurfReconFilterCellTGSz3D, 4);
    IFSHADER_DEFINE_CONST_UINT32(kSurfReconFilterCellTGSz2D, 8);

    IFSHADER_DEFINE_CONST_INT32(kSurfReconMaxParticlesPerCell, 16);

#ifndef __cplusplus
#ifndef IFSHADER_SURFRECON_2D
    IFSHADER_DEFINE_CONST_INT32(kSurfReconProbDimension, 3);
    IFSHADER_TYPEALIAS(FSpatialVector, float3);
    IFSHADER_TYPEALIAS(FSpatialIndex, int3);
    IFSHADER_TYPEALIAS(FSpatialVectorAligned, float4);
    IFSHADER_TYPEALIAS(FSpatialMatrix, float3x3);
    IFSHADER_TYPEALIAS(FSpatialHomogeneousTransform, float4x4);

    FSpatialVector FromAlignedVector(FSpatialVectorAligned vec)
    {
        return FSpatialVector(vec.x, vec.y, vec.z);
    }

    FSpatialVectorAligned ToAlignedVector(FSpatialVector vec)
    {
        return FSpatialVectorAligned(vec.x, vec.y, vec.z, 0.0f);
    }
    FSpatialIndex ToSpatialIndex(int x,int y, int z)
    {
        return FSpatialIndex(int(x), int(y), int(z));
    }
    FSpatialIndex ToSpatialIndex(int4 vec)
    {
        return FSpatialIndex(int(vec.x), int(vec.y), int(vec.z));
    }
    FSpatialVector ToSpatialVector(float x, float y, float z)
    {
        return FSpatialVector(x, y, z);
    }
    FSpatialVector ToSpatialVector(float4 vec)
    {
        return FSpatialVector(vec.x, vec.y, vec.z);
    }
#else
    IFSHADER_DEFINE_CONST_INT32(kSurfReconProbDimension, 2);
    IFSHADER_TYPEALIAS(FSpatialVector, float2);
    IFSHADER_TYPEALIAS(FSpatialIndex, int2);
    IFSHADER_TYPEALIAS(FSpatialVectorAligned, float2);
    IFSHADER_TYPEALIAS(FSpatialMatrix, float2x2);
    IFSHADER_TYPEALIAS(FSpatialHomogeneousTransform, float3x3);

    FSpatialVector FromAlignedVector(FSpatialVectorAligned vec)
    {
        return FSpatialVector(vec.x, vec.y);
    }

    FSpatialVectorAligned ToAlignedVector(FSpatialVector vec)
    {
        return FSpatialVectorAligned(vec.x, vec.y);
    }
    FSpatialIndex ToSpatialIndex(int x, int y,int z)
    {
        return FSpatialIndex(int(x), int(y));
    }
    FSpatialIndex ToSpatialIndex(int4 vec)
    {
        return FSpatialIndex(int(vec.x), int(vec.y));
    }
    FSpatialVector ToSpatialVector(float x, float y, float z)
    {
        return FSpatialVector(x, y);
    }
    FSpatialVector ToSpatialVector(float4 vec)
    {
        return FSpatialVector(vec.x, vec.y);
    }
#endif

struct FRWSpatialVectorHandle
{
    TRWStructuredBufferHandle<FSpatialVectorAligned> m_Handle;
    
    FSpatialVector Load(uint index)
    {
        return FromAlignedVector(m_Handle.Load(index));
    }

    void Store(uint index, FSpatialVector value)
    {
        m_Handle.Store(ToAlignedVector(value), index);  
    }
};

struct FSpatialVectorHandle
{
    TStructuredBufferHandle<FSpatialVectorAligned> m_Handle;
    FSpatialVector Load(uint index)
    {
        return FromAlignedVector(m_Handle.Load(index));
    }

};


struct FSurfReconGrid
{
    float4 m_MinBound;
    float4 m_MaxBound;
    int4 m_NumCells;
    int m_BlockWidthInCellUnits;
    TAtomicRWStructuredBufferHandle<int> m_CellParticleCount;
    TAtomicRWStructuredBufferHandle<int> m_ActiveCellsInBlock;
    TRWStructuredBufferHandle<int> m_SurfaceVertices;

    void MarkVertexBufferIndex(int vxSlotId)
    {
        m_SurfaceVertices.Store(1, vxSlotId);
    }

    bool IsVertexBufferIndexMarked(int vxSlotId)
    {
        return m_SurfaceVertices.Load(vxSlotId) > 0;
    }

    FSpatialIndex GetCellIndex(FSpatialVector position)
    {
        FSpatialIndex cellIndex;
        cellIndex.x = int((position.x - m_MinBound.x) / (m_MaxBound.x - m_MinBound.x) * m_NumCells.x);
        cellIndex.y = int((position.y - m_MinBound.y) / (m_MaxBound.y - m_MinBound.y) * m_NumCells.y);
#ifndef IFSHADER_SURFRECON_2D
        cellIndex.z = int((position.z - m_MinBound.z) / (m_MaxBound.z - m_MinBound.z) * m_NumCells.z);
#endif
        return cellIndex;
    }

    FSpatialVector GetCellCenter(FSpatialIndex cellIndex)
    {
        FSpatialVector cellCenter;
        cellCenter.x = m_MinBound.x + (cellIndex.x + 0.5f) * (m_MaxBound.x - m_MinBound.x) / m_NumCells.x;
        cellCenter.y = m_MinBound.y + (cellIndex.y + 0.5f) * (m_MaxBound.y - m_MinBound.y) / m_NumCells.y;
#ifndef IFSHADER_SURFRECON_2D
        cellCenter.z = m_MinBound.z + (cellIndex.z + 0.5f) * (m_MaxBound.z - m_MinBound.z) / m_NumCells.z;
#endif
        return cellCenter;
    }


    int GetCellFlattenId(FSpatialIndex cellIndex)
    {
        // TODO: consider morton encoding
        int cellId = cellIndex.x + cellIndex.y * m_NumCells.x;
#ifndef IFSHADER_SURFRECON_2D
        cellId += cellIndex.z * m_NumCells.x * m_NumCells.y;
#endif
        return cellId;
    }

    int GetCellVertexFlattenId(FSpatialIndex cellIndex)
    {
        int vxId = cellIndex.x + cellIndex.y * (m_NumCells.x + 1);
#ifndef IFSHADER_SURFRECON_2D
        vxId += cellIndex.z * (m_NumCells.x + 1) * (m_NumCells.y + 1);
#endif
        return vxId;
    }

    FSpatialIndex GetCellVertexFromFlattenId(int vxId)
    {
#ifndef IFSHADER_SURFRECON_2D
        FSpatialIndex cellIndex;
        cellIndex.z = vxId / (m_NumCells.x + 1) / (m_NumCells.y + 1);
        vxId -= cellIndex.z * (m_NumCells.x + 1) * (m_NumCells.y + 1);
        cellIndex.y = vxId / (m_NumCells.x + 1);
        cellIndex.x = vxId % (m_NumCells.x + 1);
#else
        FSpatialIndex cellIndex;
        cellIndex.y = vxId / (m_NumCells.x + 1);
        cellIndex.x = vxId % (m_NumCells.x + 1);
#endif
        return cellIndex;
    }

    FSpatialVector GetCellVertexPosition(FSpatialIndex cellIndex)
    {
        FSpatialVector cellVertex;
        cellVertex.x = m_MinBound.x + cellIndex.x * (m_MaxBound.x - m_MinBound.x) / m_NumCells.x;
        cellVertex.y = m_MinBound.y + cellIndex.y * (m_MaxBound.y - m_MinBound.y) / m_NumCells.y;
#ifndef IFSHADER_SURFRECON_2D
        cellVertex.z = m_MinBound.z + cellIndex.z * (m_MaxBound.z - m_MinBound.z) / m_NumCells.z;
#endif
        return cellVertex;
    }   


    FSpatialIndex GetBlockIndex(FSpatialIndex cellIndex)
    {
        FSpatialIndex blockIndex;
        blockIndex.x = cellIndex.x / m_BlockWidthInCellUnits;
        blockIndex.y = cellIndex.y / m_BlockWidthInCellUnits;
#ifndef IFSHADER_SURFRECON_2D
        blockIndex.z = cellIndex.z / m_BlockWidthInCellUnits;
#endif
        return blockIndex;
    }

    int GetBlockFlattenId(FSpatialIndex blockIndex)
    {
        int blockId = blockIndex.x + blockIndex.y * (m_NumCells.x / m_BlockWidthInCellUnits);
#ifndef IFSHADER_SURFRECON_2D
        blockId += blockIndex.z * (m_NumCells.x / m_BlockWidthInCellUnits) * (m_NumCells.y / m_BlockWidthInCellUnits);
#endif
        return blockId;
    }

    FSpatialIndex GetBlockCoordFromFlattenId(int blockId)
    {
#ifndef IFSHADER_SURFRECON_2D
        FSpatialIndex blockIndex;
        blockIndex.z = blockId / (m_NumCells.x / m_BlockWidthInCellUnits) / (m_NumCells.y / m_BlockWidthInCellUnits);
        blockId -= blockIndex.z * (m_NumCells.x / m_BlockWidthInCellUnits) * (m_NumCells.y / m_BlockWidthInCellUnits);
        blockIndex.y = blockId / (m_NumCells.x / m_BlockWidthInCellUnits);
        blockIndex.x = blockId % (m_NumCells.x / m_BlockWidthInCellUnits);
#else
        FSpatialIndex blockIndex;
        blockIndex.y = blockId / (m_NumCells.x / m_BlockWidthInCellUnits);
        blockIndex.x = blockId % (m_NumCells.x / m_BlockWidthInCellUnits);
#endif
        return blockIndex;
    }

    int MarkCellActive(int cellId)
    {
        int curId = m_CellParticleCount.AtomicAdd(cellId, 1);
        return curId + 1;
    }

    int MarkBlockActive(int blockId)
    {
        return m_ActiveCellsInBlock.AtomicAdd(blockId, 1) + 1;
    }

    bool IsBlockActive(int blockId)
    {
        return m_ActiveCellsInBlock.Load(blockId) > 0;
    }
    bool IsBlockInternal(int blockId)
    {
        int NumActiveCells = m_ActiveCellsInBlock.Load(blockId);
        if(NumActiveCells >= GetNumCellsInBlock())
            return true;
        return false;
    }

    bool IsCellActive(int cellId)
    {
        return m_CellParticleCount.Load(cellId) > 0;
    }

    int GetNumParticlesInCell(int CellId)
    {
        return m_CellParticleCount.Load(CellId);
    }

    bool IsInRange(FSpatialVector position)
    {
        bool inRange = true;
        inRange &= position.x >= m_MinBound.x && position.x <= m_MaxBound.x;
        inRange &= position.y >= m_MinBound.y && position.y <= m_MaxBound.y;
#ifndef IFSHADER_SURFRECON_2D
        inRange &= position.z >= m_MinBound.z && position.z <= m_MaxBound.z;
#endif  
        return inRange;
    }

    FSpatialIndex GetCellInBlockBoundaryState(FSpatialIndex CellIndex)
    {
        FSpatialIndex BlockIndex = GetBlockIndex(CellIndex);
        FSpatialIndex InBlockOffset = CellIndex - BlockIndex * m_BlockWidthInCellUnits;
        FSpatialIndex Result = FSpatialIndex(0);
        if(InBlockOffset.x == 0) Result.x = -1;
        else if(InBlockOffset.x == m_BlockWidthInCellUnits - 1) Result.x = 1;

        if(InBlockOffset.y == 0) Result.y = -1;
        else if(InBlockOffset.y == m_BlockWidthInCellUnits - 1) Result.y = 1;

#ifndef IFSHADER_SURFRECON_2D
        if(InBlockOffset.z == 0) Result.z = -1;
        else if(InBlockOffset.z == m_BlockWidthInCellUnits - 1) Result.z = 1;   
#endif
        return Result;
    }

    FSpatialIndex GetMaxBlockIndex()
    {
        FSpatialIndex maxBlockIndex;
        maxBlockIndex.x = (m_NumCells.x - 1) / m_BlockWidthInCellUnits;
        maxBlockIndex.y = (m_NumCells.y - 1) / m_BlockWidthInCellUnits;
#ifndef IFSHADER_SURFRECON_2D
        maxBlockIndex.z = (m_NumCells.z - 1) / m_BlockWidthInCellUnits;
#endif
        return maxBlockIndex;
    }

    int GetNumCellsInBlock()
    {
#ifndef IFSHADER_SURFRECON_2D
        return (m_BlockWidthInCellUnits+2) * (m_BlockWidthInCellUnits+2) * (m_BlockWidthInCellUnits+2);
#else
        return (m_BlockWidthInCellUnits+2) * (m_BlockWidthInCellUnits+2);
#endif
    }

    int GetNumCellVertices()
    {
#ifndef IFSHADER_SURFRECON_2D
        return (m_NumCells.x + 1) * (m_NumCells.y + 1) * (m_NumCells.z + 1);
#else
        return (m_NumCells.x + 1) * (m_NumCells.y + 1);
#endif
    }

    FSpatialIndex GetGridNumCellsInAxis()
    {
#ifndef IFSHADER_SURFRECON_2D
        return FSpatialIndex(m_NumCells.x,m_NumCells.y,m_NumCells.z);
#else
        return FSpatialIndex(m_NumCells.x,m_NumCells.y);
#endif
    }

};
#endif

}}}
