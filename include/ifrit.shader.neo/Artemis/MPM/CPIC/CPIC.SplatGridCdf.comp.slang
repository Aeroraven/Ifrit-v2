#pragma ifrit.multi_compile IFSHADER_MPM_3D
#include "ifrit.shader.neo/Artemis/MPM/CPIC/CPIC.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        FDenseGridStructureHandle m_Grid;
        FDenseBlockStructureHandle m_Block;
        TRWStructuredBufferHandle<float> m_CpicBlockCdfMinDist;
        TStructuredBufferHandle<int> m_CpicBlockRigidList;
        TStructuredBufferHandle<int> m_CpicBlockRigidCounter;
        TStructuredBufferHandle<FCpicRigidBoundary> m_RigidFaces;

        TRWStructuredBufferHandle<float> m_CpicGMinDist; // g_d
        TRWStructuredBufferHandle<int> m_CpicGAffinity; // g_A
        TRWStructuredBufferHandle<int> m_CpicGSign; // g_T
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
#ifdef IFSHADER_MPM_3D
    IFSHADER_BLOCKSIZE(kMpmBlockWidth, kMpmBlockWidth, kMpmBlockWidth)
#else
    //TODO: at least 50%(NV) or 75%(AMD) of the wavefront is idle
    IFSHADER_BLOCKSIZE(kMpmBlockWidth, kMpmBlockWidth, 1)
#endif
    void CpicSplatGridCdfCS(uint3 ThreadId:SV_DispatchThreadID,
                            uint3 GroupId:SV_GroupID,
                            uint3 GroupThreadId:SV_GroupThreadID)
    {
        // each cell, foreach rigid
        // compute the minimum distance to the rigid boundary => g_A, g_T
        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        FDenseBlockStructure Block = PushConst.m_Block.Load();

        FSpatialIndex BlockCoord = ToSpatialIndex(GroupId.x, GroupId.y, GroupId.z);
        FSpatialIndex GridCoord = ToSpatialIndex(ThreadId.x, ThreadId.y, ThreadId.z);

        int BlockId = Block.GetBlockId(BlockCoord);
        int GridId = Grid.EncodeSpatialIndex(GridCoord);

        int RigidCount = PushConst.m_CpicBlockRigidCounter.Load(BlockId);
        int Rigid_gA[kCpicMaxRigidPerCell/32];
        int Rigid_gT[kCpicMaxRigidPerCell/32];
        float Rigid_gMinDist = 1e30f;

        for(int i = 0; i < kCpicMaxRigidPerCell/32; ++i)
        {
            Rigid_gA[i] = 0;
            Rigid_gT[i] = 0;
        }

        FSpatialVector GridCenter = Grid.GetGridLocation(GridCoord);
        FScalar GridDx = length(Grid.m_GridSize);

        for(int i = 0; i < RigidCount; ++i)
        {
            int RigidOffset = EncodeBlockRigidIdList(BlockId, i);
            int RigidId = PushConst.m_CpicBlockRigidList.Load(RigidOffset);
            FCpicRigidBoundary RigidFace = PushConst.m_RigidFaces.Load(RigidId);

            float Distance = RigidFace.GetDistanceToPoint(GridCenter);
            bool IsValid = Distance <= 1e25f; //=>g_A
            if(abs(Distance) > GridDx)
            {
                IsValid = false;
            }
            bool IsPositive = Distance >= 0.0f; //=>g_T

            int ElementIndex = i / 32;
            Rigid_gA[ElementIndex] |= (IsValid ? 1 : 0) << (i % 32);
            Rigid_gT[ElementIndex] |= (IsPositive ? 1 : 0) << (i % 32);

            if(IsValid)
            {
                float AbsDistance = abs(Distance);
                if(AbsDistance < Rigid_gMinDist)
                {
                    Rigid_gMinDist = AbsDistance;
                }
            }
        }

        for(int i = 0; i < kCpicMaxRigidPerCell/32; ++i)
        {
            int Offset = EncodeBlockRigidBitAttrIdList(GridId, i);
            PushConst.m_CpicGAffinity.Store(Rigid_gA[i], Offset);
            PushConst.m_CpicGSign.Store(Rigid_gT[i], Offset);
        }
        PushConst.m_CpicGMinDist.Store(Rigid_gMinDist, GridId);
    }

}}}
