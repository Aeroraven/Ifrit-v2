#pragma ifrit.multi_compile IFSHADER_MPM_3D
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"

namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        // Default to Neo-Hookean now
        float4 m_MaterialParameters1;

        int m_NumParticles;
        FDenseGridStructureHandle m_Grid;
        FSpatialVectorHandle m_ParticleLocation;
        FScalarHandle m_ParticleVolume;
        FPackedMatrixHandle m_ParticleDeformationGrad;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;


    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmGridForceUpdateCS(uint3 ThreadId:SV_DispatchThreadID,
                              uint3 GroupId:SV_GroupID,
                              uint3 GroupThreadId:SV_GroupThreadID)
    {
        // Step5. grid force update
        if(ThreadId.x >= PushConst.m_NumParticles)
            return;

        FSpatialVector ParticleLocation = PushConst.m_ParticleLocation.Load(ThreadId.x);
        FScalar ParticleVolume = PushConst.m_ParticleVolume.Load(ThreadId.x);
        FSpatialTransform ParticleDeformGrad = PushConst.m_ParticleDeformationGrad.GetMatrix(ThreadId.x);
        FDenseGridStructure Grid = PushConst.m_Grid.Load();

        FSpatialIndex CenterGrid = Grid.GetCentralGrid(ParticleLocation);

        IFSHADER_MPM_RANGESEARCH_BEGIN(dx,dy,dz)
            FSpatialIndex GridOffset = ToSpatialIndex(dx,dy,dz);
            FSpatialIndex GridCell = CenterGrid + GridOffset;
            if(!Grid.IsValidGrid(GridCell)) continue;

            FSpatialVector GridCenter = Grid.GetGridLocation(GridCell);
            FSpatialVector G2PVector = GridCenter - ParticleLocation;
            FSpatialVector GradW = FSpatialVector(0.0f);
#ifdef IFSHADER_MPM_INTERP_CUBIC
            GradW = EulerianInterpCubicSpatialGradient(G2PVector, Grid.GetGridSpacing());
#else
            GradW = EulerianInterpQuadraticSpatialGradient(G2PVector, Grid.GetGridSpacing());
#endif      
            FScalar Mu = PushConst.m_MaterialParameters1.x;
            FScalar Lambda = PushConst.m_MaterialParameters1.y;
            FSpatialTransform P = NeoHookeanStress(ParticleDeformGrad, Mu, Lambda);
            P = Math::Transpose(P);

            Grid.AddForce(GridCell, -ParticleVolume*mul(P, GradW));
        IFSHADER_MPM_RANGESEARCH_END()

    }

}}}