#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS
#pragma ifrit.multi_compile IFSHADER_MPM_PBMPM
#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"

#ifdef IFSHADER_MPM_MLS
    #define IFSHADER_MPM_IGNORE_DEFORM_GRAD_UPDATE
#endif

#ifdef IFSHADER_MPM_PBMPM
#ifndef IFSHADER_MPM_IGNORE_DEFORM_GRAD_UPDATE
    #define IFSHADER_MPM_IGNORE_DEFORM_GRAD_UPDATE
#endif
#endif

// This G2P shader is only supported by MLS-MPM or PB-MPM. x_x

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        FParticleCounterHandle m_ParticleCounter;
        FScalar m_DeltaTime;
        FDenseGridStructureHandle m_Grid;
        FPackedMatrixHandle m_ParticleDeformationGrad;
        FRWSpatialVectorHandle m_ParticleLocation;
        FPackedMatrixHandle m_ParticleB;
        FRWSpatialVectorHandle m_ParticleVelocity;

        // Blocking 
        FDenseBlockStructureHandle m_Block;
        TStructuredBufferHandle<int> m_ParticleIndexesInPages;
        TStructuredBufferHandle<int> m_ParticleCountInPages;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    static const int kAffRange = kMpmBlockWidth + 2;

#ifdef IFSHADER_MPM_3D
    static const int kAffNumCells = kAffRange * kAffRange * kAffRange;
    groupshared float sInBlockVelocities[kAffNumCells * 4];
    groupshared FSpatialIndex sMinBlockInGridAffRange;
#else
    static const int kAffNumCells = kAffRange * kAffRange;
    groupshared float sInBlockVelocities[kAffNumCells * 2];
    groupshared FSpatialIndex sMinBlockInGridAffRange;
#endif

    int GetLocalGridIndex(FSpatialIndex RelativeGridIndex)
    {
#ifdef IFSHADER_MPM_3D
        return RelativeGridIndex.x * kAffRange * kAffRange + RelativeGridIndex.y * kAffRange + RelativeGridIndex.z;
#else
        return RelativeGridIndex.x * kAffRange + RelativeGridIndex.y;
#endif
    }

    FSpatialIndex GetLocalGridFromIndex(int LocalGridIndex)
    {
#ifdef IFSHADER_MPM_3D
        FSpatialIndex LocalGrid;
        LocalGrid.x = LocalGridIndex / (kAffRange * kAffRange);
        LocalGrid.y = (LocalGridIndex / kAffRange) % kAffRange;
        LocalGrid.z = LocalGridIndex % kAffRange;
        return LocalGrid;
#else
        FSpatialIndex LocalGrid;
        LocalGrid.x = LocalGridIndex / kAffRange;
        LocalGrid.y = LocalGridIndex % kAffRange;
        return LocalGrid;
#endif
    }

    FSpatialVector GetInBlockVelocity(int LocalCellId)
    {
#ifdef IFSHADER_MPM_3D
        return FSpatialVector(
            sInBlockVelocities[LocalCellId * 4 + 0],
            sInBlockVelocities[LocalCellId * 4 + 1],
            sInBlockVelocities[LocalCellId * 4 + 2],
        );
#else
        return FSpatialVector(
            sInBlockVelocities[LocalCellId * 2 + 0],
            sInBlockVelocities[LocalCellId * 2 + 1],
        );
#endif
    }


    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmBlockGridToParticleCS(uint3 ThreadId:SV_DispatchThreadID,
                                uint3 GroupId:SV_GroupID,
                                uint3 GroupThreadId:SV_GroupThreadID)
    {
        int PageId = GroupId.x;
        int NumParticlesInPage = PushConst.m_ParticleCountInPages.Load(PageId);
        int PageOffset = PageId * kMpmBlockPageSize + GroupThreadId.x;
        int ParticleId = PushConst.m_ParticleIndexesInPages.Load(PageOffset);
        bool IsValidThread = !(GroupThreadId.x >= NumParticlesInPage);

        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        FDenseBlockStructure Block = PushConst.m_Block.Load();

        if(GroupThreadId.x == 0 && IsValidThread)
        {
            FSpatialVector ParticleLocation = PushConst.m_ParticleLocation.Load(ParticleId);
            FSpatialIndex CenterGrid = Grid.GetCentralGridRounded(ParticleLocation);            
            FSpatialIndex CenterBlock = Block.GetBlockCoord(CenterGrid);
            FSpatialIndex MinBlockInGrid = CenterBlock * kMpmBlockWidth;
            FSpatialIndex MinBlockInGridAffRange = MinBlockInGrid - 1;
            sMinBlockInGridAffRange = MinBlockInGridAffRange;
        }
        GroupMemoryBarrierWithGroupSync();

        for(int i = 0; i < kAffNumCells; i += kMpmTGSizeX)
        {
            int LocalCellId = GroupThreadId.x + i;
            if(LocalCellId < kAffNumCells)
            {
                FSpatialIndex LocalGridIndex = GetLocalGridFromIndex(LocalCellId);
                FSpatialIndex GridLocIndex = sMinBlockInGridAffRange + LocalGridIndex;
                if(!Grid.IsValidGrid(GridLocIndex))
                {
#ifdef IFSHADER_MPM_3D
                    sInBlockVelocities[LocalCellId * 4 + 0] = 0.0f;
                    sInBlockVelocities[LocalCellId * 4 + 1] = 0.0f;
                    sInBlockVelocities[LocalCellId * 4 + 2] = 0.0f;
                    sInBlockVelocities[LocalCellId * 4 + 3] = 0.0f;
#else
                    sInBlockVelocities[LocalCellId * 2 + 0] = 0.0f; 
                    sInBlockVelocities[LocalCellId * 2 + 1] = 0.0f;
#endif
                }
                else
                {
                    FSpatialVector GridVelocity = Grid.LoadVelocity(GridLocIndex);
#ifdef IFSHADER_MPM_3D
                    sInBlockVelocities[LocalCellId * 4 + 0] = GridVelocity.x;
                    sInBlockVelocities[LocalCellId * 4 + 1] = GridVelocity.y;
                    sInBlockVelocities[LocalCellId * 4 + 2] = GridVelocity.z;
                    sInBlockVelocities[LocalCellId * 4 + 3] = 0.0f;
#else
                    sInBlockVelocities[LocalCellId * 2 + 0] = GridVelocity.x;
                    sInBlockVelocities[LocalCellId * 2 + 1] = GridVelocity.y;
#endif
                }

            }
        }
        GroupMemoryBarrierWithGroupSync();

  
        if(IsValidThread)
        {
            FSpatialTransform DeformGradChange = GetIdentitySpatialTransform();
            FSpatialVector ParticleLocation = PushConst.m_ParticleLocation.Load(ParticleId);
            FDenseGridStructure Grid = PushConst.m_Grid.Load();
    
            FSpatialIndex CenterGrid = Grid.GetCentralGridRounded(ParticleLocation);
            FSpatialTransform NewB = 0.0f;
            FSpatialVector NewVelocity = 0.0f;
            FScalar GridSpacing = Grid.GetGridSpacing();
            FScalar RcpSpacing = rcp(GridSpacing);
            FScalar DInv = GetDInv(GridSpacing);
            IFSHADER_MPM_RANGESEARCH_BEGIN(dx,dy,dz)
                FSpatialIndex GridOffset = ToSpatialIndex(dx,dy,dz);
                FSpatialIndex GridCell = GridOffset + CenterGrid;
                FSpatialIndex LocalGridIndex = GridCell - sMinBlockInGridAffRange;
                int LocalCellId = GetLocalGridIndex(LocalGridIndex);
    
                FSpatialVector GridCenter = Grid.GetGridLocation(GridCell);
                FSpatialVector G2PVector = GridCenter - ParticleLocation;
                FSpatialVector G2PVectorWRcp = G2PVector * RcpSpacing;
    
                FSpatialVector GradW = GetEulerianInterpGradient(G2PVectorWRcp, GridSpacing);
                FScalar W = GetEulerianInterpWeight(G2PVectorWRcp);
    
                FSpatialVector GridVelocity = GetInBlockVelocity(LocalCellId);  
                
                NewVelocity += W * GridVelocity;
                NewB += W * Math::OuterProduct(GridVelocity, G2PVector);
            IFSHADER_MPM_RANGESEARCH_END()
            PushConst.m_ParticleVelocity.Store(ParticleId, NewVelocity);
            PushConst.m_ParticleB.StoreMatrix(ParticleId, NewB);
        }
    }

}}}