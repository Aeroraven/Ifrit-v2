#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"

namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        int m_NumParticles;
        float m_DeltaTime;
        FSpatialVectorHandle m_ParticleVelocity;
        FSpatialVectorHandle m_ParticleLocation;
        FScalarHandle m_ParticleMass;
        FPackedMatrixHandle m_ParticleB;
        FDenseGridStructureHandle m_Grid;
        FPackedMatrixHandle m_ParticleDeformationGrad;
        FScalarHandle m_ParticleDeformationGradDet;
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

        FScalar DInv = 0.0f;
        FScalar GridSpacing = Grid.GetGridSpacing();

#ifdef IFSHADER_MPM_INTERP_CUBIC
        DInv = 3.0f * rcp(GridSpacing*GridSpacing);
#else
        DInv = 4.0f * rcp(GridSpacing*GridSpacing);
#endif
        FSpatialTransform ParticleC = ParticleB * DInv;

#ifdef IFSHADER_MPM_MLS
        FSpatialTransform ParticleDeformGrad = PushConst.m_ParticleDeformationGrad.GetMatrix(ThreadId.x);
        FSpatialTransform DeformGradChange = PushConst.m_DeltaTime * mul(ParticleC, ParticleDeformGrad);
        ParticleDeformGrad += DeformGradChange;
        PushConst.m_ParticleDeformationGrad.StoreMatrix(ThreadId.x, ParticleDeformGrad);

        FScalar TraceC = Math::Trace(ParticleC);
        FScalar J = PushConst.m_ParticleDeformationGradDet.Load(ThreadId.x);
        J += PushConst.m_DeltaTime * TraceC * J;
        PushConst.m_ParticleDeformationGradDet.Store(ThreadId.x, J);
#endif
        
        IFSHADER_MPM_RANGESEARCH_BEGIN(dx,dy,dz)
            FSpatialIndex GridOffset = ToSpatialIndex(dx,dy,dz);
            FSpatialIndex GridLocIndex = CenterGrid + GridOffset;

            if(!Grid.IsValidGrid(GridLocIndex)) continue;

            FSpatialVector GridCenter = Grid.GetGridLocation(GridLocIndex);
            FSpatialVector G2PVector = GridCenter - ParticleLocation;
            
            FScalar ContribWeight = 0.0f;
            
#ifdef IFSHADER_MPM_INTERP_CUBIC
            ContribWeight = EulerianInterpCubicSpatial(G2PVector/GridSpacing);
#else
            ContribWeight = EulerianInterpQuadraticSpatial(G2PVector/GridSpacing);
#endif
            FScalar ContribMass = ContribWeight * ParticleMass;
            FSpatialVector ContribMomentumBase = ContribWeight * ParticleVelocity * ParticleMass;
            FSpatialVector ContribMomentumAffine = ContribWeight * mul(ParticleC, G2PVector) * ParticleMass;
            FSpatialVector ContribMomentum = ContribMomentumBase + ContribMomentumAffine;

            Grid.AddMass(GridLocIndex, ContribMass);
            Grid.AddVelocity(GridLocIndex, ContribMomentum);

        IFSHADER_MPM_RANGESEARCH_END()
    }

}}}