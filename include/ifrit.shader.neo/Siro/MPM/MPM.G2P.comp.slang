#pragma ifrit.multi_compile IFSHADER_MPM_3D
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"

namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        int m_NumParticles;
        FScalar m_DeltaTime;
        FDenseGridStructureHandle m_Grid;
        FPackedMatrixHandle m_ParticleDeformationGrad;
        FSpatialVectorHandle m_ParticleLocation;
        FPackedMatrixHandle m_ParticleB;
        FSpatialVectorHandle m_ParticleVelocity;
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
        if(ThreadId.x >= PushConst.m_NumParticles)
            return;

        FSpatialTransform DeformGradChange = GetIdentitySpatialTransform();
        FSpatialVector ParticleLocation = PushConst.m_ParticleLocation.Load(ThreadId.x);
        FDenseGridStructure Grid = PushConst.m_Grid.Load();

        FSpatialIndex CenterGrid = Grid.GetCentralGrid(ParticleLocation);
        FSpatialTransform NewB = 0.0f;
        FSpatialVector NewVelocity = 0.0f;
        IFSHADER_MPM_RANGESEARCH_BEGIN(dx,dy,dz)
            FSpatialIndex GridOffset = ToSpatialIndex(dx,dy,dz);
            FSpatialIndex GridCell = GridOffset + CenterGrid;
            if(!Grid.IsValidGrid(GridCell)) continue;

            FSpatialVector GridCenter = Grid.GetGridLocation(GridCell);
            FSpatialVector G2PVector = GridCenter - ParticleLocation;

            FSpatialVector GradW = FSpatialVector(0.0f);
            FScalar W = 0.0f;
#ifdef IFSHADER_MPM_INTERP_CUBIC
            GradW = EulerianInterpCubicSpatialGradient(G2PVector, Grid.GetGridSpacing());
            W = EulerianInterpCubicSpatial(G2PVector);
#else
            GradW = EulerianInterpQuadraticSpatialGradient(G2PVector,Grid.GetGridSpacing());
            W = EulerianInterpQuadraticSpatial(G2PVector);
#endif
            FSpatialVector GridVelocity = Grid.LoadVelocity(GridCell);      

            DeformGradChange += PushConst.m_DeltaTime * Math::OuterProduct(GridVelocity, GradW);
            NewVelocity += W * GridVelocity;
            NewB += W * Math::OuterProduct(GridVelocity, G2PVector);
        IFSHADER_MPM_RANGESEARCH_END()

        FSpatialTransform ParticleDeformGrad = PushConst.m_ParticleDeformationGrad.GetMatrix(ThreadId.x);
        FSpatialTransform NewDeformGrad = mul(DeformGradChange, ParticleDeformGrad);
        PushConst.m_ParticleDeformationGrad.StoreMatrix(ThreadId.x, NewDeformGrad);

        PushConst.m_ParticleVelocity.Store(ThreadId.x, NewVelocity);
        PushConst.m_ParticleB.StoreMatrix(ThreadId.x, NewB);
    }

}}}