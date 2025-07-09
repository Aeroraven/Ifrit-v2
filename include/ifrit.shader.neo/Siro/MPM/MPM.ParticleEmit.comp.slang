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
        FSpatialVectorHandle m_ParticleLocationSrc;

        FSpatialVectorHandle m_ParticleLocation;
        FSpatialVectorHandle m_ParticleVelocity;
        FScalarHandle m_ParticleMass;
        FScalarHandle m_ParticleVolume;
        FPackedMatrixHandle m_ParticleB;
        FPackedMatrixHandle m_ParticleDeformationGrad;
        FScalarHandle m_ParticleDeformationGradDet;
        TRWStructuredBufferHandle<FMpmParticleMaterial> m_ParticleMaterial;
        FScalarHandle m_ParticleLiquidDensity;
        FParticleCounterHandle m_ParticleCounter;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;
    groupshared uint sParticleSlotStart;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmParticleEmitCS(uint3 ThreadId:SV_DispatchThreadID,
                           uint3 GroupId:SV_GroupID,
                           uint3 GroupThreadId:SV_GroupThreadID)
    {

        if(GroupThreadId.x == 0)
        {
            int TgMaxThreadId = ThreadId.x + kMpmTGSizeX;
            int SlotsToAllocate = min(kMpmTGSizeX, PushConst.m_NumParticles - ThreadId.x);
            sParticleSlotStart = PushConst.m_ParticleCounter.Allocate(SlotsToAllocate);
        }
        GroupMemoryBarrierWithGroupSync();
        uint ParticleSlot = sParticleSlotStart + GroupThreadId.x;

        if(ThreadId.x >= PushConst.m_NumParticles)
            return;

        FSpatialVector ParticleRefLocation = PushConst.m_ParticleLocationSrc.Load(ThreadId.x);
        
        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        FSpatialVector GridMinBound = Grid.GetMinBound();
        FSpatialVector GridMaxBound = Grid.GetMaxBound();

        PushConst.m_ParticleVelocity.Store(ParticleSlot, FSpatialVector(0.0f));
        PushConst.m_ParticleMass.Store(ParticleSlot, PushConst.m_Mass);
        PushConst.m_ParticleVolume.Store(ParticleSlot, PushConst.m_Mass / PushConst.m_Density);
        PushConst.m_ParticleLocation.Store(ParticleSlot, ParticleRefLocation);
        
        FSpatialTransform I = GetIdentitySpatialTransform();
        FSpatialTransform B = 0;

        PushConst.m_ParticleB.StoreMatrix(ParticleSlot, B);
        PushConst.m_ParticleDeformationGrad.StoreMatrix(ParticleSlot, I);

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

        PushConst.m_ParticleMaterial.Store(Material,ParticleSlot);

        PushConst.m_ParticleDeformationGradDet.Store(ParticleSlot, Math::Determinant(I));
        PushConst.m_ParticleLiquidDensity.Store(ParticleSlot, 1.0f);
    }

}}}
