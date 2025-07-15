#pragma ifrit.multi_compile IFSHADER_MPM_3D
#include "ifrit.shader.neo/Siro/MPM/CPIC/CPIC.Common.hlsli"

namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        FDenseGridStructureHandle m_Grid;
        FDenseBlockStructureHandle m_Block;
        TStructuredBufferHandle<int> m_BlockPageData; // store the block id
        TStructuredBufferHandle<int> m_BlockPageNumParticles;

        TStructuredBufferHandle<int> m_CpicBlockRigidList;
        TStructuredBufferHandle<int> m_CpicBlockRigidCounter;

        TStructuredBufferHandle<float> m_CpicGMinDist; // g_d
        TStructuredBufferHandle<int> m_CpicGAffinity; // g_A
        TStructuredBufferHandle<int> m_CpicGSign; // g_T

        TRWStructuredBufferHandle<int> m_CpicPAffinity; // p_A
        TRWStructuredBufferHandle<int> m_CpicPSign; // p_T

        TStructuredBufferHandle<int> m_ParticleIndex;
        FSpatialVectorHandle m_ParticlePosition;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

#ifdef IFSHADER_MPM_3D
    groupshared float sCpicCellCdfMinDist[kMpmBlockWidth*kMpmBlockWidth*kMpmBlockWidth];
    groupshared int sCpicCellCdfAffinity[kMpmBlockWidth*kMpmBlockWidth*kMpmBlockWidth*kCpicMaxRigidPerCell/32];
    groupshared int sCpicCellCdfSign[kMpmBlockWidth*kMpmBlockWidth*kMpmBlockWidth*kCpicMaxRigidPerCell/32];

#else
    groupshared float sCpicCellCdfMinDist[kMpmBlockWidth*kMpmBlockWidth];
    groupshared int sCpicCellCdfAffinity[kMpmBlockWidth*kMpmBlockWidth*kCpicMaxRigidPerCell/32];
    groupshared int sCpicCellCdfSign[kMpmBlockWidth*kMpmBlockWidth*kCpicMaxRigidPerCell/32];
#endif
    groupshared int sCurRigidList[kCpicMaxRigidPerCell];

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmBlockPageSize,1,1)
    void CpicReconParticleCdfCS(uint3 ThreadId:SV_DispatchThreadID,
                            uint3 GroupId:SV_GroupID,
                            uint3 GroupThreadId:SV_GroupThreadID)
    {
        // each block requires a 6x6x6 shared memory
        // neighbouring cells should get the rigid mask refiltered
        // TODO: find an elegant way to do this
        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        FDenseBlockStructure Block = PushConst.m_Block.Load();
        int BlockId = PushConst.m_BlockPageData.Load(GroupId.x);

        int CurRigidCount = PushConst.m_CpicBlockRigidCounter.Load(BlockId);
        if(GroupThreadId.x < CurRigidCount)
        {
            int RigidOffset = PushConst.m_CpicBlockRigidList.Load(EncodeBlockRigidIdList(BlockId, GroupThreadId.x));
            sCurRigidList[GroupThreadId.x] = RigidOffset;
        }

        // load the cell data into shared memory
        int TotalAffectingCells = 0;
        FSpatialIndex BaseGridCoordInBlock = Block.GetBlockCoord(BlockId) * kMpmBlockWidth;
        FSpatialIndex MinCornerGrid = BaseGridCoordInBlock - 1;
#ifdef IFSHADER_MPM_3D
        TotalAffectingCells = kMpmBlockWidth * kMpmBlockWidth * kMpmBlockWidth;
#else
        TotalAffectingCells = kMpmBlockWidth * kMpmBlockWidth;
#endif
        for(int i=0;i<TotalAffectingCells;i+=kMpmBlockPageSize)
        {
            int GridIdX = i % kMpmBlockWidth;
            int GridIdY = (i / kMpmBlockWidth) % kMpmBlockWidth;
            int GridIdZ = 0;
#ifdef IFSHADER_MPM_3D
            GridIdZ = i / (kMpmBlockWidth * kMpmBlockWidth);
#endif  
            FSpatialIndex AffectingGridOffsetR = ToSpatialIndex(GridIdX, GridIdY, GridIdZ);
            FSpatialIndex AffectingGridOffset = AffectingGridOffsetR - 1;
            
            FSpatialIndex GridCoord = BaseGridCoordInBlock + AffectingGridOffset;

            FSpatialIndex GridBlockCoord = GridCoord / kMpmBlockWidth;
            int GridBlockId = Block.GetBlockId(GridBlockCoord);
            int GridId = Grid.EncodeSpatialIndex(GridCoord);

            sCpicCellCdfMinDist[i] = PushConst.m_CpicGMinDist.Load(GridId);
            int NumRigidBitsPerCellIn32Bits = kCpicMaxRigidPerCell / 32;

            for(int j=0; j<NumRigidBitsPerCellIn32Bits; ++j)
            {
                int LoadOffset = EncodeBlockRigidBitAttrIdList(GridId, j);
                int Affinity = PushConst.m_CpicGAffinity.Load(LoadOffset);
                sCpicCellCdfAffinity[i*NumRigidBitsPerCellIn32Bits + j] = Affinity;

                int Sign = PushConst.m_CpicGSign.Load(LoadOffset);
                sCpicCellCdfSign[i*NumRigidBitsPerCellIn32Bits + j] = Sign;
            }
        }


        // foreach particle, compute p_A and p_T
        int ParticleInPageIndex = ThreadId.x;
        int ParticleIndex = PushConst.m_ParticleIndex.Load(ParticleInPageIndex);
        int NumParticlesInPage = PushConst.m_BlockPageNumParticles.Load(GroupId.x);

        if(GroupThreadId.x >= NumParticlesInPage)
            return;

        FSpatialVector ParticlePosition = PushConst.m_ParticlePosition.Load(ParticleIndex);

        FSpatialIndex ParticleGrid = Grid.GetCentralGridRounded(ParticlePosition);

        const int NumRigidAttrBits = kCpicMaxRigidPerCell/32;
        int PA[NumRigidAttrBits];
        int PT[NumRigidAttrBits];
        float PTS[NumRigidAttrBits * 32];

        for(int i = 0; i < NumRigidAttrBits; ++i)
        {
            PA[i] = 0;
            PT[i] = 0;
        }
        for(int i = 0; i < NumRigidAttrBits * 32; ++i)
        {
            PTS[i] = 0;
        }

        FScalar GridSpacing = Grid.GetGridSpacing();
        FScalar RcpGridSpacing = rcp(GridSpacing);
        
        IFSHADER_MPM_RANGESEARCH_BEGIN(dx,dy,dz)
            FSpatialIndex GridOffset = ToSpatialIndex(dx, dy, dz);
            FSpatialIndex GridCoord = ParticleGrid + GridOffset;
            FSpatialVector GridCenter = Grid.GetGridLocation(GridCoord);
            FSpatialVector G2PVector = GridCenter - ParticlePosition;

            if(!Grid.IsValidGrid(GridCoord)) continue;

            FScalar ContribWeight = 0.0f;
#ifdef IFSHADER_MPM_INTERP_CUBIC
            ContribWeight = EulerianInterpCubicSpatial(G2PVector*RcpGridSpacing);
#else
            ContribWeight = EulerianInterpQuadraticSpatial(G2PVector*RcpGridSpacing);
#endif

            FSpatialIndex GridCoordRelToMinCorner = GridCoord - MinCornerGrid;
            int SharedMemId = GridCoordRelToMinCorner.x + 
                GridCoordRelToMinCorner.y * kMpmBlockWidth;
#ifdef IFSHADER_MPM_3D
            SharedMemId += GridCoordRelToMinCorner.z * kMpmBlockWidth * kMpmBlockWidth;
#endif
            float GD = sCpicCellCdfMinDist[SharedMemId];
            int GT[NumRigidAttrBits];
            int GA[NumRigidAttrBits];
            for(int i = 0; i < NumRigidAttrBits; ++i)
            {
                GT[i] = sCpicCellCdfSign[SharedMemId * NumRigidAttrBits + i];
                GA[i] = sCpicCellCdfAffinity[SharedMemId * NumRigidAttrBits + i];
            }

            if(GD >= 1e25f)
            {
                continue;
            }

            for(int i = 0; i < NumRigidAttrBits; ++i)
            {
                PA[i] |= GA[i];
            }

            for(int i = 0; i < NumRigidAttrBits * 32; ++i)
            {
                int GTx = GT[i / 32];
                int GTBit = (GTx >> (i % 32)) & 1;
                int GABit = (GA[i / 32] >> (i % 32)) & 1;
                if(GTBit == 0) GTBit = -1;

                PTS[i] += ContribWeight * (GTBit * GABit) * GD;
            }
        IFSHADER_MPM_RANGESEARCH_END()

        for(int i = 0; i < NumRigidAttrBits * 32; ++i)
        {
            PT[i / 32] |= ((PTS[i] >= 0.0f ? 1 : 0) << (i % 32));
        }

        // store the results
        for(int i = 0; i < NumRigidAttrBits; ++i)
        {
            PushConst.m_CpicPAffinity.Store(PA[i], ParticleInPageIndex * NumRigidAttrBits + i);
            PushConst.m_CpicPSign.Store(PT[i], ParticleInPageIndex * NumRigidAttrBits + i);
        }
    
    }

}}}