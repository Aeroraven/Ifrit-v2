#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Bindless.hlsli"
#include "ifrit.shader.neo/Math.Transforms.hlsli"

namespace IfritShader {
namespace Experimental {

    struct PixelInput
    {
        IFSHADER_LOCATION(0) float2 vTexCoord : TEXCOORD0;
        // IFSHADER_LOCATION(1) float3 vWorldPos : POSITION1;
        // IFSHADER_LOCATION(2) float3 vWorldNormal : NORMAL0;
    };

    struct PixelOutput
    {
        float4 FinalColor : SV_Target0;
    };

    struct PushConstant_T
    {
        float4 mCameraPos;
        float4 mLightDir; 
        float mCameraNear;
        float mCameraFar;
        TTexture2DHandle<float>  mDepthFront;
        TTexture2DHandle<float>  mDepthBack;
        TTexture2DHandle<float>  mMask;
        TTexture2DHandle<float4> mCurColor;
        TTexture2DHandle<float4> mCurNormal;
        TTexture2DHandle<float4> mCurWorldPos;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    inline float SampleR(TTexture2DHandle<float> h, float2 uv)
    {
        return h.SampleLevel(ESamplerType::SLinearClamp, uv, 0);
    }

    inline float4 SampleRGBA(TTexture2DHandle<float4> h, float2 uv)
    {
        return h.SampleLevel(ESamplerType::SLinearClamp, uv, 0);
    }

    // Schlick's Fresnel approximation
    float3 FresnelSchlick(float cosTheta, float3 F0)
    {
        return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    }

    // GGX/Trowbridge-Reitz NDF
    float DistributionGGX(float3 N, float3 H, float roughness)
    {
        float a = roughness * roughness;
        float a2 = a * a;
        float NdotH = max(dot(N, H), 0.0);
        float NdotH2 = NdotH * NdotH;

        float num = a2;
        float denom = (NdotH2 * (a2 - 1.0) + 1.0);
        denom = 3.14159265 * denom * denom;

        return num / denom;
    }

    // Smith's method for geometry function
    float GeometrySchlickGGX(float NdotV, float roughness)
    {
        float r = (roughness + 1.0);
        float k = (r * r) / 8.0;

        float num = NdotV;
        float denom = NdotV * (1.0 - k) + k;

        return num / denom;
    }

    float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
    {
        float NdotV = max(dot(N, V), 0.0);
        float NdotL = max(dot(N, L), 0.0);
        float ggx2 = GeometrySchlickGGX(NdotV, roughness);
        float ggx1 = GeometrySchlickGGX(NdotL, roughness);

        return ggx1 * ggx2;
    }

    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    PixelOutput GelatinTestPS(PixelInput Input)
    {
        
        PixelOutput Out;

        float4 dst = SampleRGBA(PushConst.mCurColor, Input.vTexCoord);

        float mask = SampleR(PushConst.mDepthBack, Input.vTexCoord);
        if (mask <= 0.01)
        {
            Out.FinalColor = dst;
            return Out;
        }

        float zFrontClip = SampleR(PushConst.mDepthFront, Input.vTexCoord);
        float zBackClip  = SampleR(PushConst.mDepthBack,  Input.vTexCoord);
        float3 worldNormal = normalize(SampleRGBA(PushConst.mCurNormal, Input.vTexCoord).xyz);
        float3 worldPos = SampleRGBA(PushConst.mCurWorldPos, Input.vTexCoord).xyz;

        float zNear = PushConst.mCameraNear;
        float zFar  = PushConst.mCameraFar;
        float zFrontView = Math::ClipDepthToViewDepth(zFrontClip, zNear, zFar);
        float zBackView  = Math::ClipDepthToViewDepth(zBackClip,  zNear, zFar);

        float thickness = max(0.0, zBackView - zFrontView);
        
        // Transmission calculation
        float3 sigmaA = float3(0.5, 0.8, 1.2);
        float3 transmittance = exp(-sigmaA * thickness * 15.0);
        float3 baseColor = float3(0.2, 0.4, 0.6);
        float3 transmitted = dst.rgb * transmittance + (1.0 - transmittance) * baseColor;

        // Shiny surface calculation
        float3 N = normalize(worldNormal);
        float3 V = normalize(PushConst.mCameraPos.xyz - worldPos);
        float3 L = normalize(-PushConst.mLightDir.xyz); // assuming light dir points away from light
        float3 H = normalize(V + L);

        // Jelly material properties
        float roughness = 0.05;  // very smooth for wet/glassy look
        float3 F0 = float3(0.04); // dielectric base reflectance
        float IOR = 1.33; // water-like
        
        // Calculate Fresnel
        float cosTheta = max(dot(H, V), 0.0);
        float3 F = FresnelSchlick(cosTheta, F0);
        
        // Calculate specular BRDF
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        
        float3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        float3 specular = numerator / denominator;
        
        // Light contribution
        float NdotL = max(dot(N, L), 0.0);
        float3 lightColor = float3(1.0, 1.0, 1.0); // white light
        float3 radiance = lightColor * NdotL;
        
        // Combine transmission and specular
        float3 kS = F; // specular reflection coefficient
        float3 finalColor = transmitted * (1.0 - kS) + specular * radiance;
        
        // Add some rim lighting for extra shine
        float rimPower = 2.0;
        float rim = 1.0 - max(dot(N, V), 0.0);
        rim = pow(rim, rimPower);
        float3 rimColor = float3(0.8, 0.9, 1.0) * 0.3;
        finalColor += rim * rimColor;

        Out.FinalColor = float4(finalColor, 1.0);
        return Out;
    }
}}