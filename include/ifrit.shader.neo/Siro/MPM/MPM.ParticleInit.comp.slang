#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"
#include "ifrit.shader.neo/Random/Random.Simplex2d.hlsli"
namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        float m_DefaultYoungs;
        float m_DefaultPossion;
        int m_DefaultMatType;

        int m_NumParticles;
        float m_Mass;
        float m_Density;
        FDenseGridStructureHandle m_Grid;

        FSpatialVectorHandle m_ParticleLocation;
        FSpatialVectorHandle m_ParticleVelocity;
        FScalarHandle m_ParticleMass;
        FScalarHandle m_ParticleVolume;
        FPackedMatrixHandle m_ParticleB;
        FPackedMatrixHandle m_ParticleDeformationGrad;
        FScalarHandle m_ParticleDeformationGradDet;
        TRWStructuredBufferHandle<FMpmParticleMaterial> m_ParticleMaterial;
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

        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        FSpatialVector GridMinBound = Grid.GetMinBound();
        FSpatialVector GridMaxBound = Grid.GetMaxBound();


        float4 RandomOffset = float4(Hash31(float(ThreadId.x)*11.45141f),0.0f);
        FSpatialVector RandomOffsetVec = ToSpatialVector(RandomOffset);

        FSpatialVector RandomUV = lerp(FSpatialVector(0.375f),FSpatialVector(0.625f), RandomOffsetVec);
        FSpatialVector Location = lerp(GridMinBound, GridMaxBound, RandomUV);


        PushConst.m_ParticleLocation.Store(ThreadId.x, Location);
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
    }

}}}