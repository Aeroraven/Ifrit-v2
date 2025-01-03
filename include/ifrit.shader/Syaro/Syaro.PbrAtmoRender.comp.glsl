
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */


#version 450

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro/Syaro.Shared.glsl"
#include "Syaro/Syaro.SharedConst.h"

#include "Atmosphere/PAS.SharedConst.h"
#include "Atmosphere/PAS.Definition.glsl"
#include "Atmosphere/PAS.Function.glsl"
#include "Atmosphere/PAS.Shared.glsl"

layout(local_size_x = cAtmoRenderThreadGroupSizeX, local_size_y = cAtmoRenderThreadGroupSizeY, local_size_z = 1) in;

layout(push_constant) uniform PushConstant{
    vec4 sundir;
    uint perframe;
    uint outTex;
    uint depthTex;
    uint pad2;
    vec4 groundAlbedo;
    uint atmo;
    uint texTransmittance;
    uint texIrradiance;
    uint texScattering;
    uint texMieScattering;
    float earthRadius;
    float bottomAtmoRadius;
} uAtmoRenderPushConstant;

float texFetchDepth(uvec2 uv){
    float depth = texelFetch(GetSampler2D(uAtmoRenderPushConstant.depthTex), ivec2(uv), 0).r;
    return depth;
}

float raySphereIntersect(vec3 ray, vec3 rayOrigin, vec3 sphereCenter, float radius){
    // (o + t*d - c) * (o + t*d - c) = r^2
    // (o-c) * (o-c) + 2t*d*(o-c) + t^2*d*d = r^2
    // => A=d*d, B=2t*d*(o-c), C=(o-c)*(o-c)-r^2

    vec3 oc = rayOrigin - sphereCenter;
    float A = dot(ray, ray);
    float B = 2.0 * dot(ray, oc);
    float C = dot(oc, oc) - radius * radius;
    float discriminant = B*B - 4.0*A*C;
    if(discriminant < 0.0){
        return -1.0;
    }
    float t = (-B - sqrt(discriminant)) / (2.0 * A);
    return t;
}

void main(){
    uint tX = gl_GlobalInvocationID.x;
    uint tY = gl_GlobalInvocationID.y;
    uint renderHeight = uint(GetResource(bPerframeView, uAtmoRenderPushConstant.perframe).data.m_renderHeight);
    uint renderWidth = uint(GetResource(bPerframeView, uAtmoRenderPushConstant.perframe).data.m_renderWidth);
    if(tX >= renderWidth || tY >= renderHeight){
        return;
    }
    float depth = texFetchDepth(uvec2(tX, tY));
    float cameraNear = GetResource(bPerframeView, uAtmoRenderPushConstant.perframe).data.m_cameraNear;
    vec4 cameraPos = GetResource(bPerframeView, uAtmoRenderPushConstant.perframe).data.m_cameraPosition;
    vec2 uv = (vec2(tX, tY)+vec2(0.5)) / vec2(renderWidth, renderHeight);
    AtmosphereParameters atmo = GetResource(bAtmo, uAtmoRenderPushConstant.atmo).data;

    // To world ray direction
    vec2 ndcUV = uv * 2.0 - 1.0;
    vec4 clipPos = vec4(ndcUV, 0.0, 1.0) * cameraNear;
    vec4 worldPos = GetResource(bPerframeView, uAtmoRenderPushConstant.perframe).data.m_clipToWorld * clipPos;
    worldPos /= worldPos.w;
    vec3 rayDir = normalize(worldPos.xyz - cameraPos.xyz);

    vec3 camPosKm = (cameraPos.xyz+vec3(0.0,1.0,0.0)) / 1000.0;
    vec3 earthCenter = vec3(0.0, -uAtmoRenderPushConstant.earthRadius, 0.0);
    vec4 camPosKmRelativeToEarth = vec4(camPosKm - earthCenter, 1.0);
    float cosAngRad = cos(GetResource(bAtmo, uAtmoRenderPushConstant.atmo).data.sun_angular_radius);

    // Solar radiance
    vec3 solarRadiance = vec3(0.0);//GetSolarRadiance(atmo);
    vec3 sunDirection = -normalize(vec3(uAtmoRenderPushConstant.sundir.x,uAtmoRenderPushConstant.sundir.yz));//normalize(vec3(0.612372,0.500000,0.612372));
    //normalize(vec3(-0.612372,0.500000,0.612372));

    // Get ground radiance, intersection with ground
    float dist = raySphereIntersect(rayDir, camPosKmRelativeToEarth.xyz, vec3(0.0), uAtmoRenderPushConstant.earthRadius);
    float groundAlpha = 0.0;
    vec3 groundRadiance = vec3(0.0);
    if(dist > -1e-3){
        vec3 groundPos = camPosKm + rayDir * dist;
        vec3 groundNormal = normalize(groundPos - earthCenter);
        vec3 groundAlbedo = GetResource(bAtmo, uAtmoRenderPushConstant.atmo).data.ground_albedo;
        groundAlpha = 1.0;

        vec3 skyIrradiance;
        vec3 sunIrradiance = GetSunAndSkyIrradiance(
            atmo, 
            GetSampler2D(uAtmoRenderPushConstant.texTransmittance),
            GetSampler2D(uAtmoRenderPushConstant.texIrradiance),
            groundPos - earthCenter,
            groundNormal,
            sunDirection,
            skyIrradiance
        );
        groundRadiance = groundAlbedo * (1.0 / PI) * (sunIrradiance + skyIrradiance);
        vec3 transmittance;
        vec3 inScatter = GetSkyRadianceToPoint(
            atmo,
            GetSampler2D(uAtmoRenderPushConstant.texTransmittance),
            GetSampler3D(uAtmoRenderPushConstant.texScattering),
            GetSampler3D(uAtmoRenderPushConstant.texMieScattering),
            camPosKmRelativeToEarth.xyz,
            groundPos - earthCenter,
            0.0,
            sunDirection,
            transmittance
        );
        groundRadiance = groundRadiance * transmittance + inScatter;
    }
    

    // Get sky radiance
    vec3 transmittance;
    vec3 skyRadiance = GetSkyRadiance(
        atmo,
        GetSampler2D(uAtmoRenderPushConstant.texTransmittance),
        GetSampler3D(uAtmoRenderPushConstant.texScattering),
        GetSampler3D(uAtmoRenderPushConstant.texMieScattering),
        camPosKmRelativeToEarth.xyz,
        rayDir,
        0.0,
        sunDirection,
        transmittance
    );
    if(dot(rayDir,sunDirection) > cosAngRad){
        skyRadiance += transmittance * solarRadiance;
    }
    // Mix
    vec3 radiance = mix(skyRadiance, groundRadiance , groundAlpha);

    vec4 color = vec4(radiance, 1.0) ;
    color = color * 10.0;
    imageStore(GetUAVImage2DRGBA32F(uAtmoRenderPushConstant.outTex), ivec2(tX, tY), vec4(color));
}