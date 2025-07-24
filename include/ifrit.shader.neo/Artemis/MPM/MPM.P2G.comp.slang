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
        FPackedMatrixHandle m_ParticleDebug;
        FPackedMatrixHandle m_ParticleStressContrib;
        TRWStructuredBufferHandle<FMpmParticleMaterial> m_ParticleMaterial;

        // PBMPM
        uint m_IsFirstOrLastRun;

        // CPIC
#ifdef IFSHADER_MPM_CPIC
        TStructuredBufferHandle<int> m_CpicGAffinity; // g_A
        TStructuredBufferHandle<int> m_CpicGSign; // g_T
        TStructuredBufferHandle<int> m_CpicPAffinity; // p_A
        TStructuredBufferHandle<int> m_CpicPSign; // p_T
#endif
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmParticleToGridCS(uint3 ThreadId:SV_DispatchThreadID,
                             uint3 GroupId:SV_GroupID,
                             uint3 GroupThreadId:SV_GroupThreadID)
    {
        // Step1. P2G
        
        if(ThreadId.x >= PushConst.m_ParticleCounter.GetNumParticles())
            return;

        FSpatialVector ParticleVelocity = PushConst.m_ParticleVelocity.Load(ThreadId.x);
        FSpatialVector ParticleLocation = PushConst.m_ParticleLocation.Load(ThreadId.x);
        FSpatialTransform ParticleB = PushConst.m_ParticleB.GetMatrix(ThreadId.x);
        FScalar ParticleMass = PushConst.m_ParticleMass.Load(ThreadId.x);
        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        
        FSpatialIndex CenterGrid = Grid.GetCentralGridRounded(ParticleLocation);

        FScalar GridSpacing = Grid.GetGridSpacing();
        FScalar DInv = GetDInv(GridSpacing);
        FSpatialTransform ParticleC = ParticleB * DInv;

#ifdef IFSHADER_MPM_CPIC
        FCpicSignTagSet CpicParticleTagSet = GetCpicParticleAffinityAndTags(
            ThreadId.x, PushConst.m_CpicPAffinity, PushConst.m_CpicPSign,
        );
#endif


#ifdef IFSHADER_MPM_MLS
        PerformMlsParticleDeformUpdate(
            ThreadId.x,PushConst.m_DeltaTime,ParticleC,PushConst.m_ParticleDeformationGrad,
            PushConst.m_ParticleMaterial,PushConst.m_ParticlePlasticity,PushConst.m_ParticleStressContrib
        );
#endif
    
        bool IsFirstRun = (PushConst.m_IsFirstOrLastRun & 1) != 0;
        FScalar RcpGridSpacing = rcp(GridSpacing);
        FSpatialVector BaseMomentum = ParticleMass * ParticleVelocity;
        IFSHADER_MPM_RANGESEARCH_BEGIN(dx,dy,dz)
            FSpatialIndex GridOffset = ToSpatialIndex(dx,dy,dz);
            FSpatialIndex GridLocIndex = CenterGrid + GridOffset;
            int GridId = Grid.EncodeSpatialIndex(GridLocIndex);

            if(!Grid.IsValidGrid(GridLocIndex)) continue;

#ifdef IFSHADER_MPM_CPIC
            FCpicSignTagSet CpicGridTagSet = GetCpicGridAffinityAndTags(
                GridId, PushConst.m_CpicGAffinity, PushConst.m_CpicGSign
            );

            bool CpicCompatible = CheckCpicCompatibility(CpicGridTagSet, CpicParticleTagSet);
            if(!CpicCompatible) continue;
#endif

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
                Grid.AddMass(GridLocIndex, ContribMass);
                Grid.AddVelocity(GridLocIndex, ContribMomentum);
            }else
            {
                FScalar CurrentMass = Grid.LoadMass(GridLocIndex);
                if(CurrentMass> 1e-6f)
                {
                    Grid.AddVelocity(GridLocIndex, ContribMomentum/ CurrentMass);
                }
            }
#else
            Grid.AddMass(GridLocIndex, ContribMass);
            Grid.AddVelocity(GridLocIndex, ContribMomentum);
#endif

        IFSHADER_MPM_RANGESEARCH_END()
    }

}}}