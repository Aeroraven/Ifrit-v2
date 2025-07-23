#pragma once
#include "ifrit.shader.neo/Common.hlsli"

#ifndef __cplusplus
#include "ifrit.shader.neo/Bindless.hlsli"
#include "ifrit.shader.neo/Math.LinAlg.hlsli"
#include "ifrit.shader.neo/Math.LinAlg.SVD.hlsli"
#endif

#define IFSHADER_MPM_WAVE_INTRINSIC_ENABLED 1

// Pascal's fp32 shared atomics use locks. For performance, fixed-point atomics are used instead.
// Reference: https://forums.developer.nvidia.com/t/worse-atomic-performance-in-shared-than-global-memory/52150/2
#define IFSHADER_MPM_GRID_ATTRIBUTE_FIXEDPOINT 0


namespace IfritShader{
namespace Artemis{
namespace MPM{

    IFSHADER_DEFINE_CONST_UINT32(kMpmTGSizeX, 128);
    IFSHADER_DEFINE_CONST_INT32(kMpmGridSearchRange,1);

    IFSHADER_DEFINE_CONST_INT32(kMpmMaterial_Jelly,0);
    IFSHADER_DEFINE_CONST_INT32(kMpmMaterial_Fluid,1);
    IFSHADER_DEFINE_CONST_INT32(kMpmMaterial_Snow,2);
    IFSHADER_DEFINE_CONST_INT32(kMpmMaterial_Visco,3);


#ifndef __cplusplus
    IFSHADER_TYPEALIAS(FScalar, float);
    IFSHADER_DEFINE_CONST_FLOAT(kMpmFixedPointComponent, 1e11f);
    IFSHADER_DEFINE_CONST_INT32(kMpmBlockPageSize, 128);
    IFSHADER_DEFINE_CONST_INT32(kMpmBlockWidth, 4);


    int ToFixedPoint(float Value)
    {
        return int(Value * kMpmFixedPointComponent);
    }

    float ToFloatPoint(int Value)
    {
        return float(Value) / kMpmFixedPointComponent;
    }
    
#ifdef IFSHADER_MPM_3D
    IFSHADER_DEFINE_CONST_INT32(kMpmProbDimension, 3);

    IFSHADER_TYPEALIAS(FSpatialVector, float3);
    IFSHADER_TYPEALIAS(FSpatialIndex, int3);
    IFSHADER_TYPEALIAS(FSpatialVectorAligned, float4);
    IFSHADER_TYPEALIAS(FSpatialTransform, float3x3);
    IFSHADER_TYPEALIAS(FSpatialHomogeneousTransform, float4x4);

    struct FPackedMatrixHandle
    {
        TRWStructuredBufferHandle<float4> PackedMatrix;
        
        FSpatialTransform GetMatrix(uint Index)
        {
            float4 Row0 = PackedMatrix.Load(Index * 3 + 0);
            float4 Row1 = PackedMatrix.Load(Index * 3 + 1);
            float4 Row2 = PackedMatrix.Load(Index * 3 + 2);
            return FSpatialTransform(
                Row0.x, Row0.y, Row0.z,
                Row1.x, Row1.y, Row1.z,
                Row2.x, Row2.y, Row2.z
            );
        }

        void StoreMatrix(uint Index, FSpatialTransform Matrix)
        {
            PackedMatrix.Store(
                float4(Matrix[0].x, Matrix[0].y, Matrix[0].z, 0.0f),
                Index * 3 + 0
            );
            PackedMatrix.Store(
                float4(Matrix[1].x, Matrix[1].y, Matrix[1].z, 0.0f),
                Index * 3 + 1
            );
            PackedMatrix.Store(
                float4(Matrix[2].x, Matrix[2].y, Matrix[2].z, 0.0f),
                Index * 3 + 2
            );
        }
    };

    struct FSpatialVectorHandle
    {
        TStructuredBufferHandle<FSpatialVectorAligned> SpatialVectors;

        FSpatialVector Load(uint Index)
        {
            return SpatialVectors.Load(Index).xyz;
        }
    };

    struct FRWSpatialVectorHandle
    {
        TRWStructuredBufferHandle<FSpatialVectorAligned> SpatialVectors;

        FSpatialVector Load(uint Index)
        {
            return SpatialVectors.Load(Index).xyz;
        }

        void Store(uint Index, FSpatialVector Value)
        {
            SpatialVectors.Store(FSpatialVectorAligned(Value.x, Value.y, Value.z, 0.0f), Index);
        }
    };

    struct FAtomicSpatialVectorHandle
    {
        TAtomicRWStructuredBufferHandle<FScalar> SpatialVectors;

        FSpatialVector Load(uint Index)
        {
            FScalar X = SpatialVectors.Load(Index * 4 + 0);
            FScalar Y = SpatialVectors.Load(Index * 4 + 1);
            FScalar Z = SpatialVectors.Load(Index * 4 + 2);
            return FSpatialVector(X, Y, Z);
        }

        void Store(uint Index, FSpatialVector Value)
        {
            SpatialVectors.Store(Value.x, Index * 4 + 0);
            SpatialVectors.Store(Value.y, Index * 4 + 1);
            SpatialVectors.Store(Value.z, Index * 4 + 2);
        }

        void AtomicAdd(uint Index, FSpatialVector Value)
        {
            SpatialVectors.AtomicAdd(Index * 4 + 0, Value.x);
            SpatialVectors.AtomicAdd(Index * 4 + 1, Value.y);
            SpatialVectors.AtomicAdd(Index * 4 + 2, Value.z);
        }
    };

    struct FAtomicSpatialVectorHandleFixedPoint
    {
        TAtomicRWStructuredBufferHandle<int> SpatialVectors;

        FSpatialVector Load(uint Index)
        {
            int X = SpatialVectors.Load(Index * 4 + 0);
            int Y = SpatialVectors.Load(Index * 4 + 1);
            int Z = SpatialVectors.Load(Index * 4 + 2);
            return FSpatialVector(ToFloatPoint(X), ToFloatPoint(Y), ToFloatPoint(Z));
        }

        void Store(uint Index, FSpatialVector Value)
        {
            SpatialVectors.Store(ToFixedPoint(Value.x), Index * 4 + 0);
            SpatialVectors.Store(ToFixedPoint(Value.y), Index * 4 + 1);
            SpatialVectors.Store(ToFixedPoint(Value.z), Index * 4 + 2);
        }

        void AtomicAdd(uint Index, FSpatialVector Value)
        {
            SpatialVectors.AtomicAdd(Index * 4 + 0, ToFixedPoint(Value.x));
            SpatialVectors.AtomicAdd(Index * 4 + 1, ToFixedPoint(Value.y));
            SpatialVectors.AtomicAdd(Index * 4 + 2, ToFixedPoint(Value.z));
        }
    };

    FSpatialVector ToSpatialVector(float4 Value)
    {
        return FSpatialVector(Value.x, Value.y, Value.z);
    }

#define IFSHADER_MPM_RANGESEARCH_BEGIN(dx,dy,dz) \
    {\
        int dx=-kMpmGridSearchRange; \
        int dy=-kMpmGridSearchRange; \
        int dz=-kMpmGridSearchRange; \
        for(;dx<=kMpmGridSearchRange;dx++) \
        { \
            for(int dy=-kMpmGridSearchRange;dy<=kMpmGridSearchRange;dy++) \
            { \
                for(int dz=-kMpmGridSearchRange;dz<=kMpmGridSearchRange;dz++) \
                { 
                    
#define IFSHADER_MPM_RANGESEARCH_END() }}}}

#else
    IFSHADER_DEFINE_CONST_INT32(kMpmProbDimension, 2);

    IFSHADER_TYPEALIAS(FSpatialVector, float2);
    IFSHADER_TYPEALIAS(FSpatialIndex, int2);
    IFSHADER_TYPEALIAS(FSpatialVectorAligned, float2);
    IFSHADER_TYPEALIAS(FSpatialTransform, float2x2);
    IFSHADER_TYPEALIAS(FSpatialHomogeneousTransform, float3x3);

    struct FPackedMatrixHandle
    {
        TRWStructuredBufferHandle<float4> PackedMatrix;

        FSpatialTransform GetMatrix(uint Index)
        {
            float4 Row = PackedMatrix.Load(Index);
            return FSpatialTransform(
                Row.x, Row.y,
                Row.z, Row.w
            );
        }

        void StoreMatrix(uint Index, FSpatialTransform Matrix)
        {
            PackedMatrix.Store(
                float4(
                    Matrix[0].x, Matrix[0].y,
                    Matrix[1].x, Matrix[1].y
                ),
                Index
            );
        }
    };

    struct FRWSpatialVectorHandle
    {
        TRWStructuredBufferHandle<FSpatialVectorAligned> SpatialVectors;

        FSpatialVector Load(uint Index)
        {
            return SpatialVectors.Load(Index).xy;
        }

        void Store(uint Index, FSpatialVector Value)
        {
            SpatialVectors.Store(FSpatialVectorAligned(Value.x, Value.y), Index);
        }
    };

    struct FSpatialVectorHandle
    {
        TStructuredBufferHandle<FSpatialVectorAligned> SpatialVectors;

        FSpatialVector Load(uint Index)
        {
            return SpatialVectors.Load(Index).xy;
        }
    };

    struct FAtomicSpatialVectorHandle
    {
        TAtomicRWStructuredBufferHandle<FScalar> SpatialVectors;

        FSpatialVector Load(uint Index)
        {
            FScalar X = SpatialVectors.Load(Index * 2 + 0);
            FScalar Y = SpatialVectors.Load(Index * 2 + 1);
            return FSpatialVector(X, Y);
        }

        void Store(uint Index, FSpatialVector Value)
        {
            SpatialVectors.Store(Value.x, Index * 2 + 0);
            SpatialVectors.Store(Value.y, Index * 2 + 1);
        }

        void AtomicAdd(uint Index, FSpatialVector Value)
        {
            SpatialVectors.AtomicAdd(Index * 2 + 0, Value.x);
            SpatialVectors.AtomicAdd(Index * 2 + 1, Value.y);
        }
    };

    struct FAtomicSpatialVectorHandleFixedPoint
    {
        TAtomicRWStructuredBufferHandle<int> SpatialVectors;

        FSpatialVector Load(uint Index)
        {
            int X = SpatialVectors.Load(Index * 2 + 0);
            int Y = SpatialVectors.Load(Index * 2 + 1);
            return FSpatialVector(ToFloatPoint(X), ToFloatPoint(Y)); 
        }

        void Store(uint Index, FSpatialVector Value)
        {
            SpatialVectors.Store(ToFixedPoint(Value.x), Index * 2 + 0);
            SpatialVectors.Store(ToFixedPoint(Value.y), Index * 2 + 1);
        }

        void AtomicAdd(uint Index, FSpatialVector Value)
        {
            SpatialVectors.AtomicAdd(Index * 2 + 0, ToFixedPoint(Value.x));
            SpatialVectors.AtomicAdd(Index * 2 + 1, ToFixedPoint(Value.y));
        }
    };

    FSpatialVector ToSpatialVector(float4 Value)
    {
        return FSpatialVector(Value.x, Value.y);
    }
    

#define IFSHADER_MPM_RANGESEARCH_BEGIN(dx,dy,dz) \
    {\
        int dx=-kMpmGridSearchRange; \
        int dy=-kMpmGridSearchRange; \
        int dz=0; \
        for(;dx<=kMpmGridSearchRange;dx++) \
        { \
            for(int dy=-kMpmGridSearchRange;dy<=kMpmGridSearchRange;dy++) \
            { 
                

#define IFSHADER_MPM_RANGESEARCH_END() }}}

#endif

    IFSHADER_TYPEALIAS(FSpatialCondBitSet, FSpatialIndex);

    struct FMpmParticleMaterial
    {
        float m_Youngs;
        float m_Possion;
        int m_Type;
    };

    struct FRWScalarHandle
    {
        TRWStructuredBufferHandle<FScalar> Scalars;

        FScalar Load(uint Index)
        {
            return Scalars.Load(Index);
        }

        void Store(uint Index, FScalar Value)
        {
            Scalars.Store(Value, Index);
        }
    };

    
    struct FScalarHandle
    {
        TStructuredBufferHandle<FScalar> Scalars;

        FScalar Load(uint Index)
        {
            return Scalars.Load(Index);
        }

    };

    struct FAtomicScalarHandle
    {
        TAtomicRWStructuredBufferHandle<FScalar> Scalars;

        FScalar Load(uint Index)
        {
            return Scalars.Load(Index);
        }

        void Store(uint Index, FScalar Value)
        {
            Scalars.Store(Value, Index);
        }

        void AtomicAdd(uint Index, FScalar Value)
        {
            Scalars.AtomicAdd(Index, Value);
        }
    };

    struct FAtomicScalarHandleFixedPoint
    {
        TAtomicRWStructuredBufferHandle<int> Scalars;

        FScalar Load(uint Index)
        {
            return ToFloatPoint(Scalars.Load(Index));
        }
        void Store(uint Index, FScalar Value)
        {
            Scalars.Store(ToFixedPoint(Value), Index);
        }
        void AtomicAdd(uint Index, FScalar Value)
        {
            Scalars.AtomicAdd(Index, ToFixedPoint(Value));
        }
    };

    struct FParticleCounterHandle
    {
        TAtomicRWStructuredBufferHandle<int> ParticleCounter;

        int Allocate(int Count)
        {
            int Index = ParticleCounter.AtomicAdd(0,Count);
            int CurSize = Index + Count;
            int RequiredTGs = DivRoundUp(CurSize, (int)kMpmTGSizeX);
            ParticleCounter.AtomicMax(1, RequiredTGs);
            return Index;
        }

        int Set(int Count)
        {
            ParticleCounter.Store(Count,0);
            int CurSize = Count;
            int RequiredTGs = DivRoundUp(CurSize, (int)kMpmTGSizeX);
            ParticleCounter.AtomicMax(1, RequiredTGs);
            return 0;
        }

        int GetNumParticles()
        {
            return ParticleCounter.Load(0);
        }
    }

    struct FDenseGridStructure
    {
        int4 m_GridSize;
        int4 m_GridBoundaryWidth;
        float4 m_GridTranslation;

    #if IFSHADER_MPM_GRID_ATTRIBUTE_FIXEDPOINT
        FAtomicSpatialVectorHandleFixedPoint m_GridVelocity;
        FAtomicSpatialVectorHandleFixedPoint m_GridForce;
        FAtomicScalarHandleFixedPoint m_GridMass;
    #else
        FAtomicSpatialVectorHandle m_GridVelocity;
        FAtomicSpatialVectorHandle m_GridForce;
        FAtomicScalarHandle m_GridMass;
    #endif

        FScalar m_GridSpacing;

        int EncodeSpatialIndex(FSpatialIndex Index)
        {
            float4 GridSize = m_GridSize;
#ifdef IFSHADER_MPM_3D
            return Index.x + Index.y * int(GridSize.x) + Index.z * int(GridSize.x * GridSize.y);
#else
            return Index.x + Index.y * int(GridSize.x);
#endif
        }

        FSpatialIndex DecodeSpatialIndex(int EncodedIndex)
        {
            float4 GridSize = m_GridSize;
#ifdef IFSHADER_MPM_3D
            int Z = EncodedIndex / int(GridSize.x * GridSize.y);   
            int Y = (EncodedIndex - Z * int(GridSize.x * GridSize.y)) / int(GridSize.x);
            int X = EncodedIndex - Y * int(GridSize.x) - Z * int(GridSize.x * GridSize.y);
            return FSpatialIndex(X, Y, Z);
#else
            int Y = EncodedIndex / int(GridSize.x);
            int X = EncodedIndex % int(GridSize.x);
            return FSpatialIndex(X, Y); 
#endif
        }

        bool IsValidGrid(FSpatialIndex Index)
        {
            int4 GridSize = m_GridSize;
#ifdef IFSHADER_MPM_3D
            return Index.x >= 0 && Index.x < GridSize.x &&
                   Index.y >= 0 && Index.y < GridSize.y &&
                   Index.z >= 0 && Index.z < GridSize.z;
#else
            return Index.x >= 0 && Index.x < GridSize.x &&
                     Index.y >= 0 && Index.y < GridSize.y;
#endif
        }

        FSpatialIndex GetCentralGrid(FSpatialVector Position)
        {
            FScalar GridSpacing = m_GridSpacing;
            FSpatialVector GridTranslation = ToSpatialVector(m_GridTranslation);
            FSpatialVector GridPosition = (Position - GridTranslation) / GridSpacing;
            FSpatialIndex CentralGrid = FSpatialIndex(GridPosition);
            return CentralGrid;
        }

        FSpatialIndex GetCentralGridRounded(FSpatialVector Position)
        {
            FScalar GridSpacing = m_GridSpacing;
            FSpatialVector GridTranslation = ToSpatialVector(m_GridTranslation);
            FSpatialVector GridPosition = (Position - GridTranslation) / GridSpacing;
            FSpatialIndex CentralGrid = FSpatialIndex(GridPosition);
            return CentralGrid;
        }


        FScalar GetGridSpacing()
        {
            return m_GridSpacing;
        }

        void AddVelocity(FSpatialIndex Index, FSpatialVector Velocity)
        {
            int EncodedIndex = EncodeSpatialIndex(Index);
            m_GridVelocity.AtomicAdd(EncodedIndex, Velocity);
        }

        void AddForce(FSpatialIndex Index, FSpatialVector Force)
        {
            int EncodedIndex = EncodeSpatialIndex(Index);
            m_GridForce.AtomicAdd(EncodedIndex, Force);
        }

        void AddMass(FSpatialIndex Index, FScalar Mass)
        {
            int EncodedIndex = EncodeSpatialIndex(Index);
            m_GridMass.AtomicAdd(EncodedIndex, Mass);
        }

        void StoreVelocity(FSpatialIndex Index, FSpatialVector Velocity)
        {
            int EncodedIndex = EncodeSpatialIndex(Index);
            m_GridVelocity.Store(EncodedIndex, Velocity);
        }

        void StoreForce(FSpatialIndex Index, FSpatialVector Force)
        {
            int EncodedIndex = EncodeSpatialIndex(Index);
            m_GridForce.Store(EncodedIndex, Force);
        }

        void StoreMass(FSpatialIndex Index, FScalar Mass)
        {
            int EncodedIndex = EncodeSpatialIndex(Index);
            m_GridMass.Store(EncodedIndex, Mass);
        }

        FSpatialVector LoadVelocity(FSpatialIndex Index)
        {
            int EncodedIndex = EncodeSpatialIndex(Index);
            return m_GridVelocity.Load(EncodedIndex);
        }

        FSpatialVector LoadForce(FSpatialIndex Index)
        {
            int EncodedIndex = EncodeSpatialIndex(Index);
            return m_GridForce.Load(EncodedIndex);
        }

        FScalar LoadMass(FSpatialIndex Index)
        {
            int EncodedIndex = EncodeSpatialIndex(Index);
            return m_GridMass.Load(EncodedIndex);
        }

        FSpatialVector GetGridLocation(FSpatialIndex Index)
        {
            FScalar GridSpacing = m_GridSpacing;
            FSpatialVector GridTranslation = ToSpatialVector(m_GridTranslation);
            return GridTranslation + (FSpatialVector(Index)+0.5f) * GridSpacing;
        }

        int GetTotalGridCount()
        {
            int4 GridSize = m_GridSize;
#ifdef IFSHADER_MPM_3D
            return GridSize.x * GridSize.y * GridSize.z;
#else
            return GridSize.x * GridSize.y;
#endif
        }

        FSpatialCondBitSet IsBoundaryGrid(FSpatialIndex Index)
        {
            FSpatialCondBitSet Result;
            int4 BoundaryWidth = m_GridBoundaryWidth;
#ifdef IFSHADER_MPM_3D
            Result.x = (Index.x < BoundaryWidth.x) ? 1 : (Index.x >= m_GridSize.x - BoundaryWidth.x) ? 2 : 0;
            Result.y = (Index.y < BoundaryWidth.y) ? 1 : (Index.y >= m_GridSize.y - BoundaryWidth.y) ? 2 : 0;
            Result.z = (Index.z < BoundaryWidth.z) ? 1 : (Index.z >= m_GridSize.z - BoundaryWidth.z) ? 2 : 0;
#else
            Result.x = (Index.x < BoundaryWidth.x) ? 1 : (Index.x >= m_GridSize.x - BoundaryWidth.x) ? 2 : 0;
            Result.y = (Index.y < BoundaryWidth.y) ? 1 : (Index.y >= m_GridSize.y - BoundaryWidth.y) ? 2 : 0;
#endif
            return Result;
        }

        FSpatialVector GetMinBound()
        {
            return ToSpatialVector(m_GridTranslation);
        }

        FSpatialVector GetMinBoundWithBoundary()
        {
            FSpatialVector MinBound = ToSpatialVector(m_GridTranslation);
            FSpatialIndex BoundaryWidth = ToSpatialIndex(m_GridBoundaryWidth.x, m_GridBoundaryWidth.y, m_GridBoundaryWidth.z);
            return MinBound + FSpatialVector(m_GridSpacing) * FSpatialVector(BoundaryWidth);
        }

        
        FSpatialVector GetMaxBound()
        {
            FScalar GridSpacing = m_GridSpacing;
            FSpatialVector GridSize = ToSpatialVector(m_GridSize);
            return ToSpatialVector(m_GridTranslation) + GridSize * GridSpacing;
        }

        FSpatialVector GetMaxBoundWithBoundary()
        {
            FSpatialVector MaxBound = GetMaxBound();
            FSpatialIndex BoundaryWidth = ToSpatialIndex(m_GridBoundaryWidth.x, m_GridBoundaryWidth.y, m_GridBoundaryWidth.z);
            return MaxBound - FSpatialVector(m_GridSpacing) * FSpatialVector(BoundaryWidth);
        }

    };

    struct FDenseGridStructureHandle
    {
        TRWStructuredBufferHandle<FDenseGridStructure> m_Data;

        FDenseGridStructure Load()
        {
            return m_Data.Load(0);
        }
    };

    float4 SpatialVectorToFloat4(FSpatialVector Vec)
    {
#ifdef IFSHADER_MPM_3D
        return float4(Vec.x, Vec.y, Vec.z, 0.0f);
#else
        return float4(Vec.x, Vec.y, 0.0f, 0.0f);
#endif
    }

    // ==========================================
    // Grid Block
    // ==========================================

    struct FMpmBlockPageAllocData
    {
        int m_NumPages;
        int m_Offset;
        int m_StartingPageId;
    };

    // Blocks -> 4^D Cells (6^D Influence)
    struct FDenseBlockStructure
    {
        int4 m_BlockSize;
        TAtomicRWStructuredBufferHandle<int> m_BlockDispatchArgs; //(NumBlockPages, 1, 1)
        TRWStructuredBufferHandle<int> m_ParticleIndices;
        TAtomicRWStructuredBufferHandle<int> m_AllocBlockOffset;
        TAtomicRWStructuredBufferHandle<int> m_ParticleCountsInBlock;
        TRWStructuredBufferHandle<int> m_BlockStoreOffset;

        void ClearPages()
        {
            m_BlockDispatchArgs.Store(0, 0);
            m_BlockDispatchArgs.Store(1, 1);
            m_BlockDispatchArgs.Store(1, 2);
        }

        void StoreParticleIndex(int OverallOffset,int ParticleIndex)
        {
            m_ParticleIndices.Store(ParticleIndex, OverallOffset);
        }
    
        int GetParticleIndex(int OverallOffset)
        {
            return m_ParticleIndices.Load(OverallOffset);
        }

        int ScatterToBlock(int BlockId)
        {
            return m_ParticleCountsInBlock.AtomicAdd(BlockId, 1);
        }

        void SetBlockStoreOffset(int BlockId, int Offset)
        {
            m_BlockStoreOffset.Store(Offset, BlockId);
        }

        int GetBlockStoreOffset(int BlockId)
        {
            return m_BlockStoreOffset.Load(BlockId);
        }

        int GetNumParticlesInBlock(int BlockId)
        {
            return m_ParticleCountsInBlock.Load(BlockId);
        }

        

        FMpmBlockPageAllocData AllocatePagesWithElementSize(int ElementCount,int BlockId)
        {
            int NumPages = DivRoundUp(ElementCount, kMpmBlockPageSize);
            int Offset = m_AllocBlockOffset.AtomicAdd(BlockId, NumPages* kMpmBlockPageSize);
            m_BlockDispatchArgs.AtomicAdd(0, NumPages);
            
            FMpmBlockPageAllocData Ret;
            Ret.m_NumPages = NumPages;
            Ret.m_Offset = Offset;
            Ret.m_StartingPageId = Offset / kMpmBlockPageSize;
            return Ret; 
        }

        FSpatialIndex GetBlockSize(FSpatialIndex GridSize)
        {
            FSpatialIndex Ret;
#ifdef IFSHADER_MPM_3D
            Ret.x = DivRoundUp(GridSize.x, kMpmBlockWidth);
            Ret.y = DivRoundUp(GridSize.y, kMpmBlockWidth);
            Ret.z = DivRoundUp(GridSize.z, kMpmBlockWidth);
#else
            Ret.x = DivRoundUp(GridSize.x, kMpmBlockWidth); 
            Ret.y = DivRoundUp(GridSize.y, kMpmBlockWidth);
#endif
            return Ret;
        }

        FSpatialIndex GetBlockCoord(FSpatialIndex Index)
        {
#ifdef IFSHADER_MPM_3D
            return FSpatialIndex(
                Index.x / kMpmBlockWidth,
                Index.y / kMpmBlockWidth,
                Index.z / kMpmBlockWidth
            );
#else
            return FSpatialIndex(
                Index.x / kMpmBlockWidth,
                Index.y / kMpmBlockWidth
            );
#endif

        }

        int GetBlockId(FSpatialIndex BlockCoord)
        {
#ifdef IFSHADER_MPM_3D
            return BlockCoord.x + BlockCoord.y * m_BlockSize.x + BlockCoord.z * m_BlockSize.x * m_BlockSize.y;
#else
            return BlockCoord.x + BlockCoord.y * m_BlockSize.x;   
#endif
        }

        FSpatialIndex DecodeBlockId(int BlockId)
        {
#ifdef IFSHADER_MPM_3D
            int Z = BlockId / (m_BlockSize.x * m_BlockSize.y);
            int Y = (BlockId - Z * m_BlockSize.x * m_BlockSize.y) / m_BlockSize.x;
            int X = BlockId - Y * m_BlockSize.x - Z * m_BlockSize.x * m_BlockSize.y;
            return FSpatialIndex(X, Y, Z);
#else
            int Y = BlockId / m_BlockSize.x;
            int X = BlockId % m_BlockSize.x;
            return FSpatialIndex(X, Y);
#endif
        }

        int GetTotalBlockCount()
        {
#ifdef IFSHADER_MPM_3D
            return m_BlockSize.x * m_BlockSize.y * m_BlockSize.z;
#else
            return m_BlockSize.x * m_BlockSize.y;
#endif
        }

    };

    struct FDenseBlockStructureHandle
    {
        TRWStructuredBufferHandle<FDenseBlockStructure> m_Data;

        FDenseBlockStructure Load()
        {
            return m_Data.Load(0);
        }
    };


    struct FValidGridCounter
    {
        TAtomicRWStructuredBufferHandle<int> m_Data;
        
        int Allocate(int Count)
        {
            int Index = m_Data.AtomicAdd(0, Count);
            int CurSize = Index + Count;
            int RequiredTGs = DivRoundUp(CurSize, kMpmTGSizeX);
            m_Data.AtomicMax(1, RequiredTGs);
            m_Data.Store(1, 2);
            m_Data.Store(1, 3);
            return Index;
        }

        int GetValidGridCount()
        {
            return m_Data.Load(0);
        }
    };

    
    FSpatialIndex ToSpatialIndex(int dx, int dy, int dz)
    {
#ifdef IFSHADER_MPM_3D
        return FSpatialIndex(dx, dy, dz);
#else
        return FSpatialIndex(dx, dy);
#endif
    }

    FSpatialTransform GetIdentitySpatialTransform()
    {
#ifdef IFSHADER_MPM_3D
        return Math::Identity3();
#else
        return Math::Identity2();
#endif
    }

    
    // ==========================================
    // Eulerian interpolation functions
    // ==========================================

    FScalar EulerianInterpQuadratic1D(FScalar Val)
    {
        FScalar AbsX = abs(Val);
        FScalar AbsX2 = AbsX * AbsX;
        if(AbsX < 0.5f)
            return 0.75f - AbsX2;
        else if(AbsX < 1.5f)
            return 0.5f * (AbsX2 - 3.0f * AbsX + 2.25f);
        else
            return 0.0f;
    }

    FScalar EulerianInterpQuadratic1DGradient(FScalar Val)
    {
        FScalar AbsX = abs(Val);
        FScalar ValSign = sign(Val);
        if(AbsX < 0.5f)
            return -2.0f * Val;
        else if(AbsX < 1.5f)
            return Val - 1.5f * ValSign;
        else
            return 0.0f;
        
    }

    FScalar EulerianInterpCubic1D(FScalar Val)
    {
        FScalar AbsX = abs(Val);
        FScalar AbsX2 = AbsX * AbsX;
        FScalar AbsX3 = AbsX2 * AbsX;
        if(AbsX < 1.0f)
            return 0.5f * AbsX3 - AbsX2 + (2.0f / 3.0f);
        else if(AbsX < 2.0f)
            return (1.0f / 6.0f) * (-AbsX3 + 6.0f * AbsX2 - 12.0f * AbsX + 8.0f);
        else
            return 0.0f;
    }

    FScalar EulerianInterpCubic1DGradient(FScalar Val)
    {
        FScalar AbsX = abs(Val);
        FScalar ValSign = sign(Val);
        FScalar AbsX2 = AbsX * AbsX;
        if(AbsX < 1.0f)
            return (1.5f * AbsX2 - 2.0f * AbsX) * ValSign;
        else if(AbsX < 2.0f)
            return (1.0f / 6.0f) * (-3.0f * AbsX2 + 12.0f * AbsX - 12.0f) * ValSign;
        else
            return 0.0f;
    }

    FScalar EulerianInterpQuadraticSpatial(FSpatialVector Val)
    {
#ifdef IFSHADER_MPM_3D
        FScalar Wx = EulerianInterpQuadratic1D(Val.x);
        FScalar Wy = EulerianInterpQuadratic1D(Val.y);
        FScalar Wz = EulerianInterpQuadratic1D(Val.z);
        return Wx * Wy * Wz;
#else
        FScalar Wx = EulerianInterpQuadratic1D(Val.x);
        FScalar Wy = EulerianInterpQuadratic1D(Val.y);
        return Wx * Wy;
#endif
    }

    FSpatialVector EulerianInterpQuadraticSpatialGradient(FSpatialVector Val, FScalar GridSpacing)
    {
#ifdef IFSHADER_MPM_3D
        FScalar Wx = EulerianInterpQuadratic1D(Val.x);
        FScalar Wy = EulerianInterpQuadratic1D(Val.y);
        FScalar Wz = EulerianInterpQuadratic1D(Val.z);
        FScalar dWx = EulerianInterpQuadratic1DGradient(Val.x);
        FScalar dWy = EulerianInterpQuadratic1DGradient(Val.y);
        FScalar dWz = EulerianInterpQuadratic1DGradient(Val.z);
        return FSpatialVector(
            dWx * Wy * Wz,
            Wx * dWy * Wz,
            Wx * Wy * dWz
        ) * rcp(GridSpacing);
#else
        FScalar Wx = EulerianInterpQuadratic1D(Val.x);
        FScalar Wy = EulerianInterpQuadratic1D(Val.y);
        FScalar dWx = EulerianInterpQuadratic1DGradient(Val.x);
        FScalar dWy = EulerianInterpQuadratic1DGradient(Val.y);
        return FSpatialVector(
            dWx * Wy,
            Wx * dWy
        ) * rcp(GridSpacing);
#endif
    }


    FScalar EulerianInterpCubicSpatial(FSpatialVector Val)
    {
#ifdef IFSHADER_MPM_3D
        FScalar Wx = EulerianInterpCubic1D(Val.x);
        FScalar Wy = EulerianInterpCubic1D(Val.y);
        FScalar Wz = EulerianInterpCubic1D(Val.z);
        return Wx * Wy * Wz;
#else
        FScalar Wx = EulerianInterpCubic1D(Val.x);  
        FScalar Wy = EulerianInterpCubic1D(Val.y);
        return Wx * Wy;
#endif
    }

    FSpatialVector EulerianInterpCubicSpatialGradient(FSpatialVector Val, FScalar GridSpacing)
    {
#ifdef IFSHADER_MPM_3D
        FScalar Wx = EulerianInterpCubic1D(Val.x);
        FScalar Wy = EulerianInterpCubic1D(Val.y);
        FScalar Wz = EulerianInterpCubic1D(Val.z);
        FScalar dWx = EulerianInterpCubic1DGradient(Val.x);
        FScalar dWy = EulerianInterpCubic1DGradient(Val.y);
        FScalar dWz = EulerianInterpCubic1DGradient(Val.z);
        return FSpatialVector(
            dWx * Wy * Wz,
            Wx * dWy * Wz,
            Wx * Wy * dWz
        ) * rcp(GridSpacing);
#else
        FScalar Wx = EulerianInterpCubic1D(Val.x);
        FScalar Wy = EulerianInterpCubic1D(Val.y);
        FScalar dWx = EulerianInterpCubic1DGradient(Val.x);
        FScalar dWy = EulerianInterpCubic1DGradient(Val.y);
        return FSpatialVector(
            dWx * Wy,
            Wx * dWy
        ) * rcp(GridSpacing);
#endif
    }

    FScalar GetEulerianInterpWeight(FSpatialVector Val)
    {
#ifdef IFSHADER_MPM_INTERP_CUBIC
        return EulerianInterpCubicSpatial(Val);
#else
        return EulerianInterpQuadraticSpatial(Val);
#endif
    }

    FScalar GetDInv(FScalar GridSpacing)
    {
#ifdef IFSHADER_MPM_INTERP_CUBIC
        return 3.0f * rcp(GridSpacing*GridSpacing);
#else
        return 4.0f * rcp(GridSpacing*GridSpacing);
#endif
    }

    FSpatialVector GetEulerianInterpGradient(FSpatialVector Val, FScalar GridSpacing)
    {
#ifdef IFSHADER_MPM_INTERP_CUBIC
        return EulerianInterpCubicSpatialGradient(Val, GridSpacing);
#else
        return EulerianInterpQuadraticSpatialGradient(Val, GridSpacing);
#endif
    }


    // ==========================================
    // Constitutive models
    // ==========================================

    void ToLameParameter(
        FScalar YoungModulus, 
        FScalar PoissonRatio, 
        out FScalar Mu, 
        out FScalar Lambda)
    {
        Mu = YoungModulus / (2.0f * (1.0f + PoissonRatio));
        Lambda = YoungModulus * PoissonRatio / ((1.0f + PoissonRatio) * (1.0f - 2.0f * PoissonRatio));
    }

    FSpatialTransform NeoHookeanStressFT(FSpatialTransform F, FScalar Mu, FScalar Lambda)
    {
        FSpatialTransform F_T = Math::Transpose(F);
        FSpatialTransform F_InvT = Math::Inverse(F_T);
        FScalar J = Math::Determinant(F);
        FSpatialTransform P = Mu*(F-F_InvT) + Lambda*log(J)*F_InvT;
        return mul(P, F_T); 
    }

    FSpatialTransform NeoHookeanStressFT(FSpatialTransform F, FScalar J, FScalar Mu, FScalar Lambda)
    {
        FSpatialTransform F_T = Math::Transpose(F);
        FSpatialTransform F_InvT = Math::Inverse(F_T);
        FSpatialTransform P = Mu*(F-F_InvT) + Lambda*log(J)*F_InvT;
        return mul(P, F_T);
    }

    FSpatialTransform FixedCorotatedStressFT(FSpatialTransform F, FScalar Mu, FScalar Lambda)
    {
        FScalar J = Math::Determinant(F);
        FSpatialTransform I = GetIdentitySpatialTransform();
        FSpatialTransform F_T = Math::Transpose(F);
        
        FSpatialTransform R;
        FSpatialTransform S;
        Math::PolarDecomposition(F,R,S);
        FSpatialTransform P = 2.0f * Mu * mul((F - R),F_T) + Lambda * (J - 1.0f) * J * I;
        return P;
    }

    FSpatialTransform FixedCorotatedStressFT(FSpatialTransform F, FScalar J, FScalar Mu, FScalar Lambda)
    {
        FSpatialTransform F_T = Math::Transpose(F);
        FSpatialTransform I = GetIdentitySpatialTransform();
        FSpatialTransform R;
        FSpatialTransform S;
        Math::PolarDecomposition(F,R,S);
        FSpatialTransform P = 2.0f * Mu * mul((F - R),F_T) + Lambda * (J - 1.0f) * J * I;
        return P;
    }

    // ==========================================
    // Particle types
    // ==========================================
    
    bool IsElasticMaterial(FMpmParticleMaterial Material)
    {
        return Material.m_Type == kMpmMaterial_Jelly ||
               Material.m_Type == kMpmMaterial_Visco;
    }

    bool IsViscoMaterial(FMpmParticleMaterial Material)
    {
        return Material.m_Type == kMpmMaterial_Visco;
    }

    bool IsLiquidMaterial(FMpmParticleMaterial Material)
    {
        return Material.m_Type == kMpmMaterial_Fluid;
    }

#endif
}}}

