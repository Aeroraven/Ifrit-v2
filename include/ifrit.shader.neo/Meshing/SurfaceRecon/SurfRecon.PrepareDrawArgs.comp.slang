#pragma ifrit.multi_compile IFSHADER_SURFRECON_2D
#include "ifrit.shader.neo/Meshing/SurfaceRecon/SurfRecon.Common.hlsli"


/* vk spec
typedef struct VkDrawIndexedIndirectCommand {
    uint32_t    indexCount;
    uint32_t    instanceCount;
    uint32_t    firstIndex;
    int32_t     vertexOffset;
    uint32_t    firstInstance;
} VkDrawIndexedIndirect
*/

namespace IfritShader{
namespace Meshing{
namespace SurfRecon{

    struct PushConst_T
    {   
        TStructuredBufferHandle<int> m_TriangleCounter; 
        TRWStructuredBufferHandle<int> m_TargetIndirectIndexedDrawArgs;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;


    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(1, 1, 1)
    void SurfReconPrepareDrawArgsCS(uint3 ThreadId:SV_DispatchThreadID,
                                    uint3 GroupId:SV_GroupID,
                                    uint3 GroupThreadId:SV_GroupThreadID)
    {
        if(ThreadId.x == 0)
        {
            int NumTriangles = PushConst.m_TriangleCounter.Load(0);
            PushConst.m_TargetIndirectIndexedDrawArgs.Store(NumTriangles, 0); // indexCount
            PushConst.m_TargetIndirectIndexedDrawArgs.Store(1, 1); // instanceCount
            PushConst.m_TargetIndirectIndexedDrawArgs.Store(0, 2); // firstIndex
            PushConst.m_TargetIndirectIndexedDrawArgs.Store(0, 3); // vertexOffset
            PushConst.m_TargetIndirectIndexedDrawArgs.Store(0, 4); // firstInstance
        }
    }

}}}