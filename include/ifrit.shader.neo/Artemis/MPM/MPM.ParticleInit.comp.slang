#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS
#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"
#include "ifrit.shader.neo/Random/Random.Simplex2d.hlsli"
namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        int m_IgnoreParticlePosition;
        float m_DefaultYoungs;
        float m_DefaultPossion;
        int m_DefaultMatType;

        int m_NumParticles;
        float m_Mass;
        float m_Density;
        FDenseGridStructureHandle m_Grid;

        FRWSpatialVectorHandle m_ParticleLocation;
        FRWSpatialVectorHandle m_ParticleVelocity;
        FRWScalarHandle m_ParticleMass;
        FRWScalarHandle m_ParticleVolume;
        FPackedMatrixHandle m_ParticleB;
        FPackedMatrixHandle m_ParticleDeformationGrad;
        FRWScalarHandle m_ParticleDeformationGradDet;
        TRWStructuredBufferHandle<FMpmParticleMaterial> m_ParticleMaterial;
        FRWScalarHandle m_ParticleLiquidDensity;
        FParticleCounterHandle m_ParticleCounter;
        TRWStructuredBufferHandle<float4> m_ParticleColor;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    float3 Hash31(float p)
    {
        // https://www.shadertoy.com/view/4djSRW
        float3 p3 = fract(float3(p) * float3(.1031, .1030, .0973));
        p3 += dot(p3, p3.yzx+33.33);
        return fract((p3.xxy+p3.yzz)*p3.zyx); 
    }


    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmParticleInitCS(uint3 ThreadId:SV_DispatchThreadID,
                           uint3 GroupId:SV_GroupID,
                           uint3 GroupThreadId:SV_GroupThreadID)
    {
        if(ThreadId.x >= PushConst.m_NumParticles)
            return;

        if(ThreadId.x == 0)
        {
            PushConst.m_ParticleCounter.Set(PushConst.m_NumParticles);
        }

        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        FSpatialVector GridMinBound = Grid.GetMinBound();
        FSpatialVector GridMaxBound = Grid.GetMaxBound();


        float4 RandomOffset = float4(Hash31(float(ThreadId.x)*11.45141f),0.0f);
        FSpatialVector RandomOffsetVec = ToSpatialVector(RandomOffset);

        FSpatialVector RandomUV = lerp(FSpatialVector(0.35f),FSpatialVector(0.65f), RandomOffsetVec);
        FSpatialVector Location = lerp(GridMinBound, GridMaxBound, RandomUV);

        if(PushConst.m_IgnoreParticlePosition == 0)
        {
            PushConst.m_ParticleLocation.Store(ThreadId.x, Location);
        }
        PushConst.m_ParticleVelocity.Store(ThreadId.x, FSpatialVector(0.0f));
        PushConst.m_ParticleMass.Store(ThreadId.x, PushConst.m_Mass);
        PushConst.m_ParticleVolume.Store(ThreadId.x, PushConst.m_Mass / PushConst.m_Density);
        
        FSpatialTransform I = GetIdentitySpatialTransform();
        FSpatialTransform B = 0;

        PushConst.m_ParticleB.StoreMatrix(ThreadId.x, B);
        PushConst.m_ParticleDeformationGrad.StoreMatrix(ThreadId.x, I);

        FScalar DInv = 0.0f;
        FScalar GridSpacing = Grid.GetGridSpacing();
#ifdef IFSHADER_MPM_INTERP_CUBIC
        DInv = 3.0f * rcp(GridSpacing*GridSpacing);
#else
        DInv = 4.0f * rcp(GridSpacing*GridSpacing);
#endif

        FMpmParticleMaterial Material;
        Material.m_Possion = PushConst.m_DefaultPossion;
        Material.m_Youngs = PushConst.m_DefaultYoungs;
        Material.m_Type = PushConst.m_DefaultMatType;

        PushConst.m_ParticleMaterial.Store(Material,ThreadId.x);

        PushConst.m_ParticleDeformationGradDet.Store(ThreadId.x, Math::Determinant(I));
        PushConst.m_ParticleLiquidDensity.Store(ThreadId.x, 1.0f);

        PushConst.m_ParticleColor.Store( float4(1.0f, 1.0f, 1.0f, 1.0f), ThreadId.x);
    }

}}}