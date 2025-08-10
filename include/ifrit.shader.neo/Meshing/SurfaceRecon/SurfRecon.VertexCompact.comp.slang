#pragma ifrit.multi_compile IFSHADER_SURFRECON_2D
#include "ifrit.shader.neo/Meshing/SurfaceRecon/SurfRecon.Common.hlsli"

namespace IfritShader{
namespace Meshing{
namespace SurfRecon{

    struct PushConst_T
    {   
        TStructuredBufferHandle<FSurfReconGrid> m_Grid; //SRV
        TAtomicRWStructuredBufferHandle<int> m_ActiveVerticesCounter;
        TRWStructuredBufferHandle<int> m_ActiveVerticesList;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSurfReconVertexCompactTGSz, 1, 1)
    void SurfReconVertexCompactCS(uint3 ThreadId:SV_DispatchThreadID,
                                uint3 GroupId:SV_GroupID,
                                uint3 GroupThreadId:SV_GroupThreadID)
    {
        FSurfReconGrid Grid = PushConst.m_Grid.Load(0);
        int TotalVertices = Grid.GetNumCellVertices();
        bool IsVertexActive = true;
        if(ThreadId.x >= TotalVertices)
        {
            IsVertexActive = false;
        }
        if(IsVertexActive)
        {
            IsVertexActive = Grid.IsVertexBufferIndexMarked(ThreadId.x);
        }
        
        int sActiveVertexCountInWave = WaveActiveCountBits(IsVertexActive);
        int sInWaveOffset = WavePrefixCountBits(IsVertexActive);
        int sGlobalOffset = 0;

        if(WaveIsFirstLane())
        {
            sGlobalOffset = PushConst.m_ActiveVerticesCounter.AtomicAdd(0,sActiveVertexCountInWave);
        }
        int OverallOffset = WaveReadLaneFirst(sGlobalOffset) + sInWaveOffset;
        if(IsVertexActive)
        {
            PushConst.m_ActiveVerticesList.Store(ThreadId.x, OverallOffset);
        }

    }

}}}