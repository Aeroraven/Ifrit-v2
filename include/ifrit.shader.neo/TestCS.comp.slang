#include "ifrit.shader.neo/Bindless.hlsli" 
#include "ifrit.shader.neo/Math.MonteCarlo.hlsli"
#include "ifrit.shader.neo/Math.RayTrace.hlsli"
#include "ifrit.shader.neo/Math.SphericalHarmonics.hlsli"
#include "ifrit.shader.neo/Math.Transforms.hlsli"

#include "ifrit.shader.neo/Random/Random.Simplex2d.hlsli"
#include "ifrit.shader.neo/Lighting/CascadedShadowMapping.hlsli"


namespace IfritShader{

    struct PushConst_T
    {   
        TRWStructuredBufferHandle<int> mData;
    };
    
    IFSHADER_PUSHCONST_DECL(PushConst_T,PushConst);
    
    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(1, 1, 1)
    void TestCS(uint3 threadId : SV_DispatchThreadID)
    {
        PushConst.mData.Store(114514, 0);
    }
}


// Test