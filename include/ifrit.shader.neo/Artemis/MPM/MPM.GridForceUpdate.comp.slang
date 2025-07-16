#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS
#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        FParticleCounterHandle m_ParticleCounter;
        FDenseGridStructureHandle m_Grid;
        FRWSpatialVectorHandle m_ParticleLocation;
        FRWScalarHandle m_ParticleVolume;
        FPackedMatrixHandle m_ParticleDeformationGrad;
        FRWScalarHandle m_ParticleDeformationGradDet;
        FPackedMatrixHandle m_ParticleStressContrib;
        TRWStructuredBufferHandle<FMpmParticleMaterial> m_ParticleMaterial;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;


    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmGridForceUpdateCS(uint3 ThreadId:SV_DispatchThreadID,
                              uint3 GroupId:SV_GroupID,
                              uint3 GroupThreadId:SV_GroupThreadID)
    {
        // Step5. grid force update
        if(ThreadId.x >= PushConst.m_ParticleCounter.GetNumParticles())
            return;

        FSpatialVector ParticleLocation = PushConst.m_ParticleLocation.Load(ThreadId.x);
        FScalar ParticleVolume = PushConst.m_ParticleVolume.Load(ThreadId.x);
        FSpatialTransform ParticleDeformGrad = PushConst.m_ParticleDeformationGrad.GetMatrix(ThreadId.x);
        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        FScalar GridSpacing = Grid.GetGridSpacing();
        FSpatialIndex CenterGrid = Grid.GetCentralGridRounded(ParticleLocation);
        FScalar J = PushConst.m_ParticleDeformationGradDet.Load(ThreadId.x);
        FMpmParticleMaterial ParticleMaterial = PushConst.m_ParticleMaterial.Load(ThreadId.x);
        
        FScalar DInv = GetDInv(GridSpacing);

        FSpatialTransform PFT = 0.0f;
#ifdef IFSHADER_MPM_MLS
        PFT = PushConst.m_ParticleStressContrib.GetMatrix(ThreadId.x);
#else
        FScalar Young = ParticleMaterial.m_Youngs;
        FScalar Possion =  ParticleMaterial.m_Possion;
        FScalar Mu, Lambda;
        ToLameParameter(Young, Possion,Mu, Lambda);

        //PFT = NeoHookeanStressFT(ParticleDeformGrad, Mu, Lambda);
        PFT = FixedCorotatedStressFT(ParticleDeformGrad,Mu,Lambda);
#endif

        IFSHADER_MPM_RANGESEARCH_BEGIN(dx,dy,dz)
            FSpatialIndex GridOffset = ToSpatialIndex(dx,dy,dz);
            FSpatialIndex GridCell = CenterGrid + GridOffset;
            if(!Grid.IsValidGrid(GridCell)) continue;

            FSpatialVector GridCenter = Grid.GetGridLocation(GridCell);
            FSpatialVector G2PVector = GridCenter - ParticleLocation;
            
#ifdef IFSHADER_MPM_MLS
            FScalar W = GetEulerianInterpWeight(G2PVector/GridSpacing);
            Grid.AddForce(GridCell, -ParticleVolume * W * DInv * mul(PFT, G2PVector));
#else
            FSpatialVector GradW = GetEulerianInterpGradient(G2PVector/GridSpacing, GridSpacing);
            Grid.AddForce(GridCell, -ParticleVolume*mul(PFT, GradW));
#endif

        IFSHADER_MPM_RANGESEARCH_END()

    }

}}}