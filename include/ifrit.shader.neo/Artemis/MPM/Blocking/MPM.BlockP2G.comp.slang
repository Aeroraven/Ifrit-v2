#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS
#pragma ifrit.multi_compile IFSHADER_MPM_PBMPM

#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"
#include "ifrit.shader.neo/Artemis/MPM/MPM.MlsParticleIntegrateUtil.hlsli"
#include "ifrit.shader.neo/Artemis/MPM/CPIC/CPIC.Compatibility.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        FParticleCounterHandle m_ParticleCounter;
        float m_DeltaTime;
        FRWSpatialVectorHandle m_ParticleVelocity;
        FRWSpatialVectorHandle m_ParticleLocation;
        FRWScalarHandle m_ParticleMass;
        FPackedMatrixHandle m_ParticleB;
        FDenseGridStructureHandle m_Grid;
        FPackedMatrixHandle m_ParticleDeformationGrad;
        FRWScalarHandle m_ParticlePlasticity; //m_ParticleDeformationGradDet
        FRWSpatialVectorHandle m_ParticleDebug;
        FPackedMatrixHandle m_ParticleStressContrib;
        TRWStructuredBufferHandle<FMpmParticleMaterial> m_ParticleMaterial;

        // PBMPM
        uint m_IsFirstOrLastRun;

        // Blocking
        FDenseBlockStructureHandle m_Block;
        TStructuredBufferHandle<int> m_ParticleIndexesInPages;
        TStructuredBufferHandle<int> m_ParticleCountInPages;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;
    static const int kAffRange = kMpmBlockWidth + 2;

//#define IFSHADER_MPM_BLOCKING_FIXED_POINT_MATH 1
    static const float kFixedPointScale = 2e10f;
    static const float kFixedPointScaleInv = 1.0f / kFixedPointScale;

    
#ifdef IFSHADER_MPM_3D
    static const int kAffNumCells = kAffRange * kAffRange * kAffRange;
    #ifdef IFSHADER_MPM_BLOCKING_FIXED_POINT_MATH
        groupshared int sInBlockVelocities[kAffNumCells * 4];
    #else
        groupshared float sInBlockVelocities[kAffNumCells * 4];
    #endif
    groupshared FSpatialIndex sMinBlockInGridAffRange;
#else
    static const int kAffNumCells = kAffRange * kAffRange;
    #ifdef IFSHADER_MPM_BLOCKING_FIXED_POINT_MATH
        groupshared int sInBlockVelocities[kAffNumCells * 2];
        groupshared int sInBlockMasses[kAffNumCells];
    #else
        groupshared float sInBlockVelocities[kAffNumCells * 2];
        groupshared float sInBlockMasses[kAffNumCells];
    #endif
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

    FScalar FixedPointToFloat(int FixedPointValue)
    {
        return FixedPointValue * kFixedPointScaleInv;
    }

    int FloatToFixedPoint(FScalar Value)
    {
        return int(Value * kFixedPointScale);
    }


    void InterlockedAddMass(int LocalGrid, FScalar Mass)
    {
#ifdef IFSHADER_MPM_3D

    #ifdef IFSHADER_MPM_BLOCKING_FIXED_POINT_MATH
        InterlockedAdd(sInBlockVelocities[LocalGrid * 4 + 3], FloatToFixedPoint(Mass));
    #else
        InterlockedAdd(sInBlockVelocities[LocalGrid * 4 + 3], Mass);
    #endif
#else
    #ifdef IFSHADER_MPM_BLOCKING_FIXED_POINT_MATH
        InterlockedAdd(sInBlockMasses[LocalGrid], FloatToFixedPoint(Mass));
    #else
        InterlockedAdd(sInBlockMasses[LocalGrid], Mass);
    #endif
#endif
    }

    FScalar GetMass(int LocalGrid)
    {
#ifdef IFSHADER_MPM_3D
    #ifdef IFSHADER_MPM_BLOCKING_FIXED_POINT_MATH
        return FixedPointToFloat(sInBlockVelocities[LocalGrid * 4 + 3]);
    #else
        return sInBlockVelocities[LocalGrid * 4 + 3];
    #endif
#else
    #ifdef IFSHADER_MPM_BLOCKING_FIXED_POINT_MATH
        return FixedPointToFloat(sInBlockMasses[LocalGrid]);
    #else
        return sInBlockMasses[LocalGrid];
    #endif
#endif
    }

    FSpatialVector GetVelocity(int LocalGrid)
    {
#ifdef IFSHADER_MPM_3D
        FSpatialVector Velocity;
    #ifdef IFSHADER_MPM_BLOCKING_FIXED_POINT_MATH
        Velocity.x = FixedPointToFloat(sInBlockVelocities[LocalGrid * 4 + 0]);
        Velocity.y = FixedPointToFloat(sInBlockVelocities[LocalGrid * 4 + 1]);
        Velocity.z = FixedPointToFloat(sInBlockVelocities[LocalGrid * 4 + 2]);
    #else
        Velocity.x = sInBlockVelocities[LocalGrid * 4 + 0];
        Velocity.y = sInBlockVelocities[LocalGrid * 4 + 1];
        Velocity.z = sInBlockVelocities[LocalGrid * 4 + 2];
    #endif
        return Velocity;
#else
        FSpatialVector Velocity;
    #ifdef IFSHADER_MPM_BLOCKING_FIXED_POINT_MATH
        Velocity.x = FixedPointToFloat(sInBlockVelocities[LocalGrid * 2 + 0]);
        Velocity.y = FixedPointToFloat(sInBlockVelocities[LocalGrid * 2 + 1]);
    #else
        Velocity.x = sInBlockVelocities[LocalGrid * 2 + 0];
        Velocity.y = sInBlockVelocities[LocalGrid * 2 + 1];
    #endif
        return Velocity;
#endif
    }


    void InterlockedAddVelocity(int LocalGrid, FSpatialVector Velocity)
    {
#ifdef IFSHADER_MPM_3D
    #ifdef IFSHADER_MPM_BLOCKING_FIXED_POINT_MATH
        InterlockedAdd(sInBlockVelocities[LocalGrid * 4 + 0], FloatToFixedPoint(Velocity.x));
        InterlockedAdd(sInBlockVelocities[LocalGrid * 4 + 1], FloatToFixedPoint(Velocity.y));
        InterlockedAdd(sInBlockVelocities[LocalGrid * 4 + 2], FloatToFixedPoint(Velocity.z));
    #else
        InterlockedAdd(sInBlockVelocities[LocalGrid * 4 + 0], Velocity.x);
        InterlockedAdd(sInBlockVelocities[LocalGrid * 4 + 1], Velocity.y);
        InterlockedAdd(sInBlockVelocities[LocalGrid * 4 + 2], Velocity.z);
    #endif
#else
    #ifdef IFSHADER_MPM_BLOCKING_FIXED_POINT_MATH
        InterlockedAdd(sInBlockVelocities[LocalGrid * 2 + 0], FloatToFixedPoint(Velocity.x));
        InterlockedAdd(sInBlockVelocities[LocalGrid * 2 + 1], FloatToFixedPoint(Velocity.y));
    #else
        InterlockedAdd(sInBlockVelocities[LocalGrid * 2 + 0], Velocity.x);
        InterlockedAdd(sInBlockVelocities[LocalGrid * 2 + 1], Velocity.y);
    #endif
#endif
    }


    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmBlockParticleToGridCS(uint3 ThreadId:SV_DispatchThreadID,
                                uint3 GroupId:SV_GroupID,
                                uint3 GroupThreadId:SV_GroupThreadID)
    {
        int PageId = GroupId.x;
        int NumParticlesInPage = PushConst.m_ParticleCountInPages.Load(PageId);
        int PageOffset = PageId * kMpmBlockPageSize + GroupThreadId.x;
        int ParticleId = PushConst.m_ParticleIndexesInPages.Load(PageOffset);
        bool IsValidThread = !(GroupThreadId.x >= NumParticlesInPage);
        bool IsFirstRun = (PushConst.m_IsFirstOrLastRun & 1) != 0;

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

        for(int i = 0; i < kAffNumCells; i+=kMpmTGSizeX)
        {
            int LocalCellId = i + GroupThreadId.x;
            if(LocalCellId < kAffNumCells){
                FSpatialIndex LocalGridIndex = GetLocalGridFromIndex(LocalCellId);
                FSpatialIndex GridLocIndex = sMinBlockInGridAffRange + LocalGridIndex;
#ifdef IFSHADER_MPM_3D
                sInBlockVelocities[LocalCellId * 4 + 0] = 0;
                sInBlockVelocities[LocalCellId * 4 + 1] = 0;
                sInBlockVelocities[LocalCellId * 4 + 2] = 0;
                sInBlockVelocities[LocalCellId * 4 + 3] = 0;
#else
                sInBlockVelocities[LocalCellId * 2 + 0] = 0;
                sInBlockVelocities[LocalCellId * 2 + 1] = 0;
                sInBlockMasses[LocalCellId] = 0;
#endif
            }
        }
        GroupMemoryBarrierWithGroupSync();
        
        if(IsValidThread)
        {
            FSpatialVector ParticleVelocity = PushConst.m_ParticleVelocity.Load(ParticleId);
            FSpatialVector ParticleLocation = PushConst.m_ParticleLocation.Load(ParticleId);
            FSpatialTransform ParticleB = PushConst.m_ParticleB.GetMatrix(ParticleId);
            FScalar ParticleMass = PushConst.m_ParticleMass.Load(ParticleId);
            
            FSpatialIndex CenterGrid = Grid.GetCentralGridRounded(ParticleLocation);
            
            // Convert to local coordinate
            FSpatialIndex MinBlockInGridAffRange = sMinBlockInGridAffRange;

            FScalar GridSpacing = Grid.GetGridSpacing();
            FScalar DInv = GetDInv(GridSpacing);
            FSpatialTransform ParticleC = ParticleB * DInv;

#ifdef IFSHADER_MPM_MLS
            PerformMlsParticleDeformUpdate(
                ParticleId,PushConst.m_DeltaTime,ParticleC,PushConst.m_ParticleDeformationGrad,
                PushConst.m_ParticleMaterial,PushConst.m_ParticlePlasticity,PushConst.m_ParticleStressContrib
            );
#endif 
            FScalar RcpGridSpacing = rcp(GridSpacing);
            FSpatialVector BaseMomentum = ParticleMass * ParticleVelocity;

            IFSHADER_MPM_RANGESEARCH_BEGIN(dx,dy,dz)
                FSpatialIndex GridOffset = ToSpatialIndex(dx,dy,dz);
                FSpatialIndex GridLocIndex = CenterGrid + GridOffset;
                FSpatialIndex LocalGridIndex = GridLocIndex - MinBlockInGridAffRange;   
                int LocalGridId = GetLocalGridIndex(LocalGridIndex);

                FSpatialVector GridCenter = Grid.GetGridLocation(GridLocIndex);
                FSpatialVector G2PVector = GridCenter - ParticleLocation;
                
                FScalar ContribWeight  = GetEulerianInterpWeight(G2PVector*RcpGridSpacing);
                FScalar ContribMass = ContribWeight * ParticleMass;
                FSpatialVector ContribMomentumBase = ContribWeight * BaseMomentum;
                FSpatialVector ContribMomentumAffine = ContribMass * mul(ParticleC, G2PVector);
                FSpatialVector ContribMomentum = ContribMomentumBase + ContribMomentumAffine;
                
#ifdef IFSHADER_MPM_PBMPM
                if(IsFirstRun)
                {   
                    InterlockedAddMass(LocalGridId, ContribMass);
                }
                InterlockedAddVelocity(LocalGridId, ContribMomentum);
#else
                InterlockedAddMass(LocalGridId, ContribMass);
                InterlockedAddVelocity(LocalGridId, ContribMomentum);
#endif
            IFSHADER_MPM_RANGESEARCH_END()
        }

        GroupMemoryBarrierWithGroupSync();

        for(int i = 0; i < kAffNumCells; i+=kMpmTGSizeX)
        {
            int LocalCellId = i + GroupThreadId.x;
            if(LocalCellId >= kAffNumCells) continue;


            FSpatialIndex LocalGridIndex = GetLocalGridFromIndex(LocalCellId);
            FSpatialIndex GridLocIndex = sMinBlockInGridAffRange + LocalGridIndex;
            if(!Grid.IsValidGrid(GridLocIndex)) continue;

            FSpatialVector ContribMomentum = GetVelocity(LocalCellId);

#ifdef IFSHADER_MPM_PBMPM
            if(IsFirstRun)
            {   
                bool HasMass = GetMass(LocalCellId) > 1e-6f;
                if(HasMass)
                {
                    Grid.AddMass(GridLocIndex, GetMass(LocalCellId));
                    Grid.AddVelocity(GridLocIndex, ContribMomentum);
                }
            }else
            {
                FScalar CurrentMass = Grid.LoadMass(GridLocIndex);
                bool HasVelocity = abs(ContribMomentum.x) > 1e-12f;
                HasVelocity = HasVelocity || abs(ContribMomentum.y) > 1e-12f;
#ifdef IFSHADER_MPM_3D
                HasVelocity = HasVelocity || abs(ContribMomentum.z) > 1e-12f;
#endif
                if(CurrentMass > 1e-6f && HasVelocity) 
                {
                    Grid.AddVelocity(GridLocIndex, ContribMomentum / CurrentMass);
                }
            }
#else
            Grid.AddMass(GridLocIndex, GetMass(LocalCellId));
            Grid.AddVelocity(GridLocIndex, ContribMomentum);
#endif            
        }
    
    
    }

}}}
