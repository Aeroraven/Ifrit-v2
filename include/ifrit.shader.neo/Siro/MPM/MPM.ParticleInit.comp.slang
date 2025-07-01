#pragma ifrit.multi_compile IFSHADER_MPM_3D
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"
#include "ifrit.shader.neo/Random/Random.Simplex2d.hlsli"
namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
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
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

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

        float RandomSeedX = ThreadId.x * 114.514f;
        float RandomSeedY = ThreadId.x * 1919.0f;
        float RandomOffsetX = Random::Simplex2D(float2(RandomSeedX, RandomSeedY));

        float RandomSeedY2 = ThreadId.x * 1919.0f + 1.0f;
        float RandomOffsetY = Random::Simplex2D(float2(RandomSeedY2, RandomSeedX));

        float RandomSeedZ = ThreadId.x * 7234.0f + 721.0f;
        float RandomOffsetZ = Random::Simplex2D(float2(RandomSeedZ, RandomSeedY2));

        float4 RandomOffset = float4(RandomOffsetX, RandomOffsetY, RandomOffsetZ, 0.0f);
        FSpatialVector RandomOffsetVec = ToSpatialVector(RandomOffset);

        FSpatialVector RandomUV = lerp(FSpatialVector(0.375f),FSpatialVector(0.625f), RandomOffsetVec);
        FSpatialVector Location = lerp(GridMinBound, GridMaxBound, RandomUV);


        PushConst.m_ParticleLocation.Store(ThreadId.x, Location);
        PushConst.m_ParticleVelocity.Store(ThreadId.x, FSpatialVector(0.0f));
        PushConst.m_ParticleMass.Store(ThreadId.x, PushConst.m_Mass);
        PushConst.m_ParticleVolume.Store(ThreadId.x, PushConst.m_Mass / PushConst.m_Density);
        
        FSpatialTransform I = GetIdentitySpatialTransform();
        FSpatialTransform B = 0;

        PushConst.m_ParticleB.StoreMatrix(ThreadId.x, I);
        PushConst.m_ParticleDeformationGrad.StoreMatrix(ThreadId.x, B);
    }

}}}