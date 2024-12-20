
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

// Stockham FFT & IFFT, in single pass

#version 450
#include "Base.glsl"
#include "Bindless.glsl"


layout(push_constant) uniform PushConstFFTConv2d{
    uint srcDownScale;
    uint kernDownScale;
    uint srcRtW;
    uint srcRtH;
    uint kernRtW;
    uint kernRtH;
    uint srcImage;
    uint srcIntermImage;
    uint kernImage;
    uint kernIntermImage;
    uint dstImage;
    uint tempImage;
    uint fftTexSizeWLog;
    uint fftTexSizeHLog;
    uint fftStep;
}pc;

layout(local_size_x=8,local_size_y=8,local_size_z=1) in;

void main(){
    uint px = gl_GlobalInvocationID.x;
    uint py = gl_GlobalInvocationID.y;
    if(px>=pc.srcRtW || py>=pc.srcRtH){
        return;
    }

    uint fftW = 1<<pc.fftTexSizeWLog;
    uint fftH = 1<<pc.fftTexSizeHLog;
    uint downscale = pc.srcDownScale;

    uint padL = fftW/2 - pc.srcRtW/downscale/2;
    uint padR = fftW - pc.srcRtW/downscale - padL;
    uint padT = fftH/2 - pc.srcRtH/downscale/2;
    uint padB = fftH - pc.srcRtH/downscale - padT;

    // Get unpadded src image
    float posXf = float(padL)+float(px)/float(pc.srcRtW)*(fftW-padL-padR);
    float posYf = float(padT)+float(py)/float(pc.srcRtH)*(fftH-padT-padB);

    vec2 srcVal = imageLoad(GetUAVImage2DRGBA32F(pc.kernIntermImage),ivec2(posXf,posYf)).rg;

    // Store to dstimage
    imageStore(GetUAVImage2DRGBA32F(pc.dstImage),ivec2(px,py),vec4(srcVal,0.0,0.0));

}