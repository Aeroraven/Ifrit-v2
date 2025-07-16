#pragma ifrit.multi_compile IFSHADER_MPM_3D
#include "ifrit.shader.neo/Artemis/MPM/CPIC/CPIC.Common.hlsli"
#include "ifrit.shader.neo/Artemis/MPM/CPIC/CPIC.MlsEstimator.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        FDenseGridStructureHandle m_Grid;
        FDenseBlockStructureHandle m_Block;
        TStructuredBufferHandle<int> m_BlockPageData; // store the block id
        TStructuredBufferHandle<int> m_BlockPageNumParticles;

        TStructuredBufferHandle<int> m_CpicBlockRigidList;
        TStructuredBufferHandle<int> m_CpicBlockRigidCounter;

        TStructuredBufferHandle<float> m_CpicGMinDist; // g_d
        TRWStructuredBufferHandle<float> m_CpicPDist; // p_D
        FRWSpatialVectorHandle m_CpicPNormal; // p_N

        TStructuredBufferHandle<int> m_ParticleIndex;
        FSpatialVectorHandle m_ParticlePosition;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmBlockPageSize,1,1)
    void CpicReconParticleNormalCS(uint3 ThreadId:SV_DispatchThreadID,
                            uint3 GroupId:SV_GroupID,
                            uint3 GroupThreadId:SV_GroupThreadID)
    {
        int BlockPageIndex = GroupId.x;
        int BlockId = PushConst.m_BlockPageData.Load(BlockPageIndex);
        int InPageParticleIndex = GroupThreadId.x;
        
        FDenseBlockStructure Block = PushConst.m_Block.Load();
        FDenseGridStructure Grid = PushConst.m_Grid.Load();

        int NumParticlesInPage = PushConst.m_BlockPageNumParticles.Load(BlockPageIndex);
        if(InPageParticleIndex >= NumParticlesInPage)
            return;
        
        int ParticleIndex = PushConst.m_ParticleIndex.Load(ThreadId.x);
        FSpatialVector ParticlePosition = PushConst.m_ParticlePosition.Load(ParticleIndex);


        FSpatialIndex ParticleGrid = Grid.GetCentralGridRounded(ParticlePosition);

        FCpicLinearBasisQ Q;
        FCpicLinearBasisKsi Ksi;
        FCpicLinearBasisMat MInv;
        FCpicNeighbourValues GDs;

        FScalar GridSpacing = Grid.GetGridSpacing();
        FScalar RcpGridSpacing = rcp(GridSpacing);

        IFSHADER_MPM_RANGESEARCH_BEGIN(dx,dy,dz)
            FSpatialIndex GridOffset = ToSpatialIndex(dx, dy, dz);
            FSpatialIndex GridCoord = ParticleGrid + GridOffset;
            int GridId = Grid.EncodeSpatialIndex(GridCoord);    
            if(!Grid.IsValidGrid(GridCoord)) continue;  

            FSpatialVector GridCenter = Grid.GetGridLocation(GridCoord);
            FSpatialVector G2PVector = GridCenter - ParticlePosition;

            int NeighborCellId = EncodeRangeSearchOffset(dx, dy, dz);

            FScalar ContribWeight = 0.0f;
#ifdef IFSHADER_MPM_INTERP_CUBIC
            ContribWeight = EulerianInterpCubicSpatial(G2PVector*RcpGridSpacing);
#else
            ContribWeight = EulerianInterpQuadraticSpatial(G2PVector*RcpGridSpacing);
#endif
            FCpicLinearBasis Basis = EncodeLinearBasis(G2PVector);
            Q.m_Rows[NeighborCellId] = Basis;
            Ksi.m_Diag[NeighborCellId] = ContribWeight;
            
            FScalar GD = PushConst.m_CpicGMinDist.Load(GridId);
            GDs.m_Values[NeighborCellId] = GD;

        IFSHADER_MPM_RANGESEARCH_END()

        MInv = GetCpicLinearMInv(Q,Ksi);
        FCpicLinearBasis DistAndNormal = GetCpicReconValueAndGradient_LinearBasis(MInv, Q, Ksi, GDs);

        FScalar PDist;
        FSpatialVector PNormal;
        GetCpicPDistAndPNormal(DistAndNormal, PDist, PNormal);

        PNormal = normalize(PNormal);

        PushConst.m_CpicPDist.Store(PDist, ParticleIndex);
        PushConst.m_CpicPNormal.Store(ParticleIndex, PNormal);  
    }

}}}
