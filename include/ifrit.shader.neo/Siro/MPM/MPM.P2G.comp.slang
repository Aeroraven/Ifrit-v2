#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS
#pragma ifrit.multi_compile IFSHADER_MPM_PBMPM
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
        FScalarHandle m_ParticlePlasticity; //m_ParticleDeformationGradDet
        FPackedMatrixHandle m_ParticleDebug;
        FPackedMatrixHandle m_ParticleStressContrib;
        TRWStructuredBufferHandle<FMpmParticleMaterial> m_ParticleMaterial;

        // PBMPM
        uint m_IsFirstOrLastRun;
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
        FSpatialTransform I = GetIdentitySpatialTransform();
        FScalar ParticleMass = PushConst.m_ParticleMass.Load(ThreadId.x);
        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        FMpmParticleMaterial ParticleMaterial = PushConst.m_ParticleMaterial.Load(ThreadId.x);

        FSpatialIndex CenterGrid = Grid.GetCentralGridRounded(ParticleLocation);

        FScalar DInv = 0.0f;
        FScalar GridSpacing = Grid.GetGridSpacing();

        bool IsFluid = ParticleMaterial.m_Type == kMpmMaterial_Fluid;
        bool IsSnow = ParticleMaterial.m_Type == kMpmMaterial_Snow;
        bool IsJelly = ParticleMaterial.m_Type == kMpmMaterial_Jelly;

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

        {
            FSpatialTransform R,S;
            Math::PolarDecomposition(ParticleDeformGrad, R, S);
        }

        FScalar TraceC = Math::Trace(ParticleC);
        FScalar Jp = PushConst.m_ParticlePlasticity.Load(ThreadId.x);
        FScalar Hardening = exp(10 * (1.0 - Jp));  // From MPM-99

        //if(IsJelly) Hardening = 0.3f;

        FSpatialTransform U,S,V;
        Math::SVD(ParticleDeformGrad, U, S, V);
        S = Math::ClampDiag(S, 1e-6f, 1e30f);
        FSpatialTransform Sx = S;

        if(IsSnow)
        {
            Sx = Math::ClampDiag(Sx, 1-2.5e-2f, 1+4.5e-3f);
            Jp *= Math::DiagProductRelative(S,Sx);
        }
        PushConst.m_ParticlePlasticity.Store(ThreadId.x, Jp);
        FScalar J = Math::DiagProduct(Sx,1e-6f);

        if(IsFluid)
        {
    #ifdef IFSHADER_MPM_3D
            ParticleDeformGrad = GetIdentitySpatialTransform() * pow(J, 1.0f/3.0f);
    #else
            ParticleDeformGrad = GetIdentitySpatialTransform() * sqrt(J);
    #endif
        }
        if(IsSnow)
        {
            ParticleDeformGrad = mul(U, mul(Sx, Math::Transpose(V)));
        }
        PushConst.m_ParticleDeformationGrad.StoreMatrix(ThreadId.x, ParticleDeformGrad);

        FScalar Young = ParticleMaterial.m_Youngs;
        FScalar Possion = ParticleMaterial.m_Possion;
        FScalar Mu, Lambda;
        ToLameParameter( Young, Possion,Mu, Lambda);
        if(IsFluid)
        {
            Mu = 0.0f;
        }
        Mu *= Hardening;
        Lambda *= Hardening;

        FSpatialTransform PFT = 0.0f;
        PFT = FixedCorotatedStressFT(ParticleDeformGrad, J, Mu,Lambda);
        PushConst.m_ParticleStressContrib.StoreMatrix(ThreadId.x, PFT);
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