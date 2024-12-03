
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
#extension GL_GOOGLE_include_directive : enable

#include "Base.glsl"
#include "Bindless.glsl"
#include "Deferred.glsl"
#include "Syaro/Syaro.Shared.glsl"
#include "Syaro/Syaro.SharedConst.h"

layout(local_size_x = cEmitGbufThreadGroupSizeX, local_size_y = 1, local_size_z = 1) in;

#include "Syaro/Syaro.EmitGBuffer.Shared.glsl"

void main(){
    uvec2 px = gbcomp_GetPixel();
    if(px.x==~0){
        return;
    }
    uint gBuffer = gbcomp_GetGBufferId();
    uvec2 clusterTri = gbcomp_GetVisBufferData(px);

    gbcomp_TriangleData triData = gbcomp_GetTriangleData(clusterTri,px);

    GBufferPixel pixelData;
    pixelData.albedo = vec3(1.0,1.0,1.0) * 5.0;
    pixelData.normal = vec3(triData.vpNormalVS.xyz) * 0.5 + 0.5;

    gbcomp_WriteGBuffer(gBuffer, px, pixelData);
}