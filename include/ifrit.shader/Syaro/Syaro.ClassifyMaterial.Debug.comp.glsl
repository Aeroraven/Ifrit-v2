
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
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro/Syaro.Shared.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include "Syaro/Syaro.ClassifyMaterial.Shared.glsl"

void main(){
    uint offset = GetResource(bMaterialCounter, uMaterialPassData.materialCounterRef).data[0].offset;
    uint tX = gl_GlobalInvocationID.x;
    uint address = offset + tX;

    uint pixelId = GetResource(bMaterialPixelList, uMaterialPassData.materialPixelListRef).data[address];
    uint pX = pixelId % uMaterialPassPushConstant.renderWidth;
    uint pY = pixelId / uMaterialPassPushConstant.renderWidth;

    imageStore(GetUAVImage2DRGBA32F(uMaterialPassData.debugImageRef), ivec2(pX, pY), vec4(1.0, 0.0, 0.0, 1.0));
}