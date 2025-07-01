#pragma ifrit.multi_compile IFSHADER_MPM_3D
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"

namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        int m_NumParticles;
        FSpatialVectorHandle m_ParticleVelocity;
        FSpatialVectorHandle m_ParticleLocation;
        FScalarHandle m_ParticleMass;
        FPackedMatrixHandle m_ParticleB;
        FDenseGridStructureHandle m_Grid;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmParticleToGridCS(uint3 ThreadId:SV_DispatchThreadID,
                             uint3 GroupId:SV_GroupID,
                             uint3 GroupThreadId:SV_GroupThreadID)
    {
        // Step1. P2G
        
        if(ThreadId.x >= PushConst.m_NumParticles)
            return;

        FSpatialVector ParticleVelocity = PushConst.m_ParticleVelocity.Load(ThreadId.x);
        FSpatialVector ParticleLocation = PushConst.m_ParticleLocation.Load(ThreadId.x);
        FSpatialTransform ParticleB = PushConst.m_ParticleB.GetMatrix(ThreadId.x);
        FScalar ParticleMass = PushConst.m_ParticleMass.Load(ThreadId.x);
        FDenseGridStructure Grid = PushConst.m_Grid.Load();

        FSpatialIndex CenterGrid = Grid.GetCentralGrid(ParticleLocation);
        
        IFSHADER_MPM_RANGESEARCH_BEGIN(dx,dy,dz)
            FSpatialIndex GridOffset = ToSpatialIndex(dx,dy,dz);
            FSpatialIndex GridLocIndex = CenterGrid + GridOffset;

            if(!Grid.IsValidGrid(GridLocIndex)) continue;

            FSpatialVector GridCenter = Grid.GetGridLocation(GridLocIndex);
            FSpatialVector G2PVector = GridCenter - ParticleLocation;

            FScalar ContribWeight = 0.0f;
            FScalar DInv = 0.0f;
#ifdef IFSHADER_MPM_INTERP_CUBIC
            ContribWeight = EulerianInterpCubicSpatial(G2PVector);
            DInv = 3.0f * rcp(Grid.GetGridSpacing()*Grid.GetGridSpacing());
#else
            ContribWeight = EulerianInterpQuadraticSpatial(G2PVector);
            DInv = 4.0f * rcp(Grid.GetGridSpacing()*Grid.GetGridSpacing());
#endif
            FScalar ContribMass = ContribWeight * ParticleMass;
            FSpatialVector ContribMomentumBase = ContribWeight * ParticleVelocity * ParticleMass;

            ParticleB *= DInv;
            FSpatialVector ContribMomentumAffine = ContribWeight * mul(ParticleB, G2PVector) * ParticleMass;
            FSpatialVector ContribMomentum = ContribMomentumBase + ContribMomentumAffine;

            Grid.AddMass(GridLocIndex, ContribMass);
            Grid.AddVelocity(GridLocIndex, ContribMomentum);

        IFSHADER_MPM_RANGESEARCH_END()
    }

}}}