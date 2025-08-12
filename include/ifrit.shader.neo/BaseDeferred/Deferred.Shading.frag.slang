#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Bindless.hlsli"

namespace IfritShader {
namespace BaseDeferred {

    // Procedural ramps (override via compile-time defines if desired)
    #ifndef IFR_TOON_RAMP_A_SHADOW
    #define IFR_TOON_RAMP_A_SHADOW float3(0.06f, 0.06f, 0.10f)
    #endif
    #ifndef IFR_TOON_RAMP_A_LIGHT
    #define IFR_TOON_RAMP_A_LIGHT  float3(1.00f, 0.96f, 0.88f)
    #endif
    #ifndef IFR_TOON_RAMP_B_SHADOW
    #define IFR_TOON_RAMP_B_SHADOW float3(0.00f, 0.10f, 0.14f)
    #endif
    #ifndef IFR_TOON_RAMP_B_LIGHT
    #define IFR_TOON_RAMP_B_LIGHT  float3(0.72f, 0.90f, 1.00f)
    #endif
    #ifndef IFR_TOON_RAMP_MIX
    #define IFR_TOON_RAMP_MIX 0.35f   // 0 = Ramp A, 1 = Ramp B
    #endif

    // Toon banding constants (override at compile time if needed)
    #ifndef IFR_TOON_STEPS
    #define IFR_TOON_STEPS 4          // >=1
    #endif
    #ifndef IFR_TOON_AMBIENT
    #define IFR_TOON_AMBIENT 0.08f    // 0..1
    #endif

    static const float3 kRampA_Shadow = IFR_TOON_RAMP_A_SHADOW;
    static const float3 kRampA_Light  = IFR_TOON_RAMP_A_LIGHT;
    static const float3 kRampB_Shadow = IFR_TOON_RAMP_B_SHADOW;
    static const float3 kRampB_Light  = IFR_TOON_RAMP_B_LIGHT;
    static const float  kRampMix      = IFR_TOON_RAMP_MIX;

    static const int    kToonSteps    = IFR_TOON_STEPS;
    static const float  kAmbient      = IFR_TOON_AMBIENT;

    struct PixelInput
    {
        IFSHADER_LOCATION(0) float2 vTexCoord : TEXCOORD0;
    };

    struct PixelOutput
    {
        float4 FinalColor : SV_Target0;
    };

    struct PushConstant_T
    {
        TTexture2DHandle<float4> m_Color;
        TTexture2DHandle<float4> m_Normal;
        TTexture2DHandle<float4> m_WorldPos;
        TTexture2DHandle<float> mMask; 
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    PixelOutput BaseDeferredShadingPS(PixelInput Input)
    {
        PixelOutput Output;
        float3 TestLightDir = normalize(float3(-1.0f, -1.0f, 1.0f));

        // Sample the GBuffer
        float4 ColorSample    = PushConst.m_Color.   SampleLevel(ESamplerType::SLinearClamp, Input.vTexCoord, 0);
        float4 NormalSample   = PushConst.m_Normal.  SampleLevel(ESamplerType::SLinearClamp, Input.vTexCoord, 0);
        float4 WorldPosSample = PushConst.m_WorldPos.SampleLevel(ESamplerType::SLinearClamp, Input.vTexCoord, 0);
        float MaskSample      = PushConst.mMask.SampleLevel(ESamplerType::SLinearClamp, Input.vTexCoord, 0);

        

        // Guard against invalid normal
        if (length(NormalSample.xyz) < 1e-6f)
        {
            Output.FinalColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
            return Output;
        }

        float3 N = normalize(NormalSample.xyz);
        float3 L = -TestLightDir;
        float  NdotL = saturate(dot(N, L));
        float lit = max(NdotL, saturate(kAmbient));
        int   steps = max(kToonSteps, 1);
        float s     = (float)steps;
        int   idx   = (int)floor(saturate(lit) * s);
        idx         = min(idx, steps - 1);
        float bandT = (steps > 1) ? (float(idx) / float(steps - 1)) : 0.0f;


        float3 rampA = lerp(kRampA_Shadow, kRampA_Light, bandT);
        float3 rampB = lerp(kRampB_Shadow, kRampB_Light, bandT);
        float3 ramp  = lerp(rampA, rampB, saturate(kRampMix));

        float3 finalRGB = ColorSample.rgb * ramp;

        Output.FinalColor = float4(finalRGB, ColorSample.a);
        if(MaskSample > 0.5f)
        {
            // make a skyblue
            Output.FinalColor = float4(float3(0.2f, 0.4f, 0.6f), ColorSample.a);
        }
        return Output;
    }
}}