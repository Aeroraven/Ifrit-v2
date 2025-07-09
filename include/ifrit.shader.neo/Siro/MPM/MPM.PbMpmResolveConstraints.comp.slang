#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS

#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"
#include "ifrit.shader.neo/Random/Random.Simplex2d.hlsli"
namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        FParticleCounterHandle m_ParticleCounter;
        float m_DeltaTime;
        FRWSpatialVectorHandle m_ParticleVelocity;
        FPackedMatrixHandle m_ParticleB; // Deformation deplacement VARIATION RATE!
        FPackedMatrixHandle m_ParticleDeformationGrad;
        TRWStructuredBufferHandle<FMpmParticleMaterial> m_ParticleMaterial;
        FDenseGridStructureHandle m_Grid;
        FPackedMatrixHandle m_ParticleDebug;
        FRWScalarHandle m_ParticleLiquidDensity;

        // PBMPM
        float m_ElasticityInterpolationFactor;
        float m_ElasticityRelaxationFactor;
        float m_LiquidViscosity;
        float m_LiquidRelaxation;
        float m_ViscoPlasticity;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void PbMpmResolveConstraintsCS(uint3 ThreadId:SV_DispatchThreadID,
                           uint3 GroupId:SV_GroupID,
                           uint3 GroupThreadId:SV_GroupThreadID)
    {
        if(ThreadId.x >= PushConst.m_ParticleCounter.GetNumParticles())
            return;

        FScalar DeltaTime = PushConst.m_DeltaTime;
        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        FScalar DInv = 0.0f;
        FScalar GridSpacing = Grid.GetGridSpacing();
#ifdef IFSHADER_MPM_INTERP_CUBIC
        DInv = 3.0f * rcp(GridSpacing*GridSpacing);
#else
        DInv = 4.0f * rcp(GridSpacing*GridSpacing);
#endif

        FMpmParticleMaterial ParticleMaterial = PushConst.m_ParticleMaterial.Load(ThreadId.x);
        FSpatialTransform ParticleDeformDisplacement = PushConst.m_ParticleB.GetMatrix(ThreadId.x) * DInv;
        FSpatialTransform ParticleDeformGrad = PushConst.m_ParticleDeformationGrad.GetMatrix(ThreadId.x);   
        FSpatialTransform I = GetIdentitySpatialTransform();

        bool IsElastic = IsElasticMaterial(ParticleMaterial);
        bool IsLiquid = IsLiquidMaterial(ParticleMaterial);

        if(IsElastic)
        {
            FSpatialTransform Fs = mul(I+ParticleDeformDisplacement, ParticleDeformGrad);
            FSpatialTransform R,S;
            Math::PolarDecomposition(Fs,R,S);

            FScalar DetFs = Math::Determinant(Fs);
            FScalar ADetFs = clamp(abs(DetFs), 0.1f, 1000.0f);
#ifdef IFSHADER_MPM_3D
            FSpatialTransform Q = I / (sign(DetFs) * pow(ADetFs, 1.0f/3.0f));
#else
            FSpatialTransform Q = I / (sign(DetFs) * sqrt(ADetFs));
#endif  
            FSpatialTransform Avol = mul(Q,Fs);
            FSpatialTransform Ashape = R;
            
            FScalar beta = PushConst.m_ElasticityInterpolationFactor;
            FSpatialTransform Tgt = (beta * Ashape + (1.0f - beta) * Avol);
            FSpatialTransform InvF = Math::Inverse(ParticleDeformGrad);
            FSpatialTransform Dp = mul(Tgt, InvF) - I;

            FScalar DetR = Math::Determinant(Tgt);
            PushConst.m_ParticleDebug.StoreMatrix(ThreadId.x,ParticleDeformDisplacement);

            FSpatialTransform Diff = Dp - ParticleDeformDisplacement;
            ParticleDeformDisplacement += PushConst.m_ElasticityRelaxationFactor * Diff;
        }

        if(IsLiquid)
        {
            FSpatialTransform Deviatoric = -(ParticleDeformDisplacement+Math::Transpose(ParticleDeformDisplacement));
            ParticleDeformDisplacement += (PushConst.m_LiquidViscosity * 0.5f) * Deviatoric;
            FScalar LiquidDensity = PushConst.m_ParticleLiquidDensity.Load(ThreadId.x);
            FScalar Alpha = 0.5f * (1.0f / LiquidDensity - Math::Trace(ParticleDeformDisplacement) - 1.0f);
            ParticleDeformDisplacement += (PushConst.m_LiquidRelaxation * Alpha) * I;
        }

        FSpatialTransform NewB = ParticleDeformDisplacement / DInv;
        PushConst.m_ParticleB.StoreMatrix(ThreadId.x, NewB);
    }

}}}

