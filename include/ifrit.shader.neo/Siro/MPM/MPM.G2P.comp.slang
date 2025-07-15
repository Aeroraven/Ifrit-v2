#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS
#pragma ifrit.multi_compile IFSHADER_MPM_PBMPM
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"

#ifdef IFSHADER_MPM_MLS
    #define IFSHADER_MPM_IGNORE_DEFORM_GRAD_UPDATE
#endif

#ifdef IFSHADER_MPM_PBMPM
#ifndef IFSHADER_MPM_IGNORE_DEFORM_GRAD_UPDATE
    #define IFSHADER_MPM_IGNORE_DEFORM_GRAD_UPDATE
#endif
#endif

namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        FParticleCounterHandle m_ParticleCounter;
        FScalar m_DeltaTime;
        FDenseGridStructureHandle m_Grid;
        FPackedMatrixHandle m_ParticleDeformationGrad;
        FRWSpatialVectorHandle m_ParticleLocation;
        FPackedMatrixHandle m_ParticleB;
        FRWSpatialVectorHandle m_ParticleVelocity;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmGridToParticleCS(uint3 ThreadId:SV_DispatchThreadID,
                             uint3 GroupId:SV_GroupID,
                             uint3 GroupThreadId:SV_GroupThreadID)
    {
        // Step6. particle deformation gradient update
        // Step7. G2P
        if(ThreadId.x >= PushConst.m_ParticleCounter.GetNumParticles())
            return;

        FSpatialTransform DeformGradChange = GetIdentitySpatialTransform();
        FSpatialVector ParticleLocation = PushConst.m_ParticleLocation.Load(ThreadId.x);
        FDenseGridStructure Grid = PushConst.m_Grid.Load();

        FSpatialIndex CenterGrid = Grid.GetCentralGridRounded(ParticleLocation);
        FSpatialTransform NewB = 0.0f;
        FSpatialVector NewVelocity = 0.0f;
        FScalar GridSpacing = Grid.GetGridSpacing();
        FScalar DInv = GetDInv(GridSpacing);
        IFSHADER_MPM_RANGESEARCH_BEGIN(dx,dy,dz)
            FSpatialIndex GridOffset = ToSpatialIndex(dx,dy,dz);
            FSpatialIndex GridCell = GridOffset + CenterGrid;
            if(!Grid.IsValidGrid(GridCell)) continue;

            FSpatialVector GridCenter = Grid.GetGridLocation(GridCell);
            FSpatialVector G2PVector = GridCenter - ParticleLocation;

            FSpatialVector GradW = GetEulerianInterpGradient(G2PVector/GridSpacing, GridSpacing);
            FScalar W = GetEulerianInterpWeight(G2PVector/GridSpacing);

            FSpatialVector GridVelocity = Grid.LoadVelocity(GridCell);      

#ifndef IFSHADER_MPM_IGNORE_DEFORM_GRAD_UPDATE
            DeformGradChange += PushConst.m_DeltaTime * Math::OuterProduct(GridVelocity, GradW);
#endif
            NewVelocity += W * GridVelocity;
            NewB += W * Math::OuterProduct(GridVelocity, G2PVector);
        IFSHADER_MPM_RANGESEARCH_END()

#ifndef IFSHADER_MPM_IGNORE_DEFORM_GRAD_UPDATE
        FSpatialTransform ParticleDeformGrad = PushConst.m_ParticleDeformationGrad.GetMatrix(ThreadId.x);
        FSpatialTransform NewDeformGrad = mul(DeformGradChange, ParticleDeformGrad);
        PushConst.m_ParticleDeformationGrad.StoreMatrix(ThreadId.x, NewDeformGrad);
#endif

        PushConst.m_ParticleVelocity.Store(ThreadId.x, NewVelocity);
        PushConst.m_ParticleB.StoreMatrix(ThreadId.x, NewB);
    }

}}}