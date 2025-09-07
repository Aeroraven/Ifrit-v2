
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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


#extension GL_GOOGLE_include_directive : require


#include "Base.glsl"
#include "Bindless.glsl"
#include "ComputeUtils.glsl"
#include "SamplerUtils.SharedConst.h"
#include "Math.SphericalHarmonics.glsl"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"

layout(
    local_size_x = kAyanamiDbgScrProbeUniformVisKernelSize, 
    local_size_y = kAyanamiDbgScrProbeUniformVisKernelSize, 
    local_size_z = 1 
) in;

layout(push_constant) uniform UPushConst{
    uint m_RTWidth;
    uint m_RTHeight;
    uint m_AlbedoSRV;
    uint m_OutputTextureUAV;
} PushConst;

void main(){
    uvec2 PixelLocation = gl_GlobalInvocationID.xy;
    bool Valid = true;
    if(PixelLocation.x >= PushConst.m_RTWidth || PixelLocation.y >= PushConst.m_RTHeight){
        Valid=false;
    }
    bool IsProbeX = PixelLocation.x % kAyanami_ScreenProbeUniformPlaceTileWidth == 0;
    bool IsProbeY = PixelLocation.y % kAyanami_ScreenProbeUniformPlaceTileWidth == 0;

    if(IsProbeX && IsProbeY){
        imageStore(GetUAVImage2DRGBA32F(PushConst.m_OutputTextureUAV), ivec2(PixelLocation), vec4(1.0, 0.0, 0.0, 1.0));
    }else{
        vec2 UV = (vec2(PixelLocation) + vec2(0.5)) / vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
        vec3 Albedo = SampleTexture2D(PushConst.m_AlbedoSRV, sLinearClamp, UV).rgb;
        imageStore(GetUAVImage2DRGBA32F(PushConst.m_OutputTextureUAV), ivec2(PixelLocation), vec4(Albedo, 1.0));
    }
}