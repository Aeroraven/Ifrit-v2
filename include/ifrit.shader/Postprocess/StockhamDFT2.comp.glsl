
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

layout(push_constant) uniform PushConstDFT2{
    uint logW;
    uint logH;
    uint rtW;
    uint rtH;
    uint downscaleFactor;
    uint rawSampImg;
    uint srcImg;
    uint dstImg;
    uint orientation; //0-row;1-col
    uint dftMode; //0-dft;1-idft
}pc;

const uint kBlockSize = 256;
layout(local_size_x=kBlockSize,local_size_y=1,local_size_z=1) in;

vec2 loadSrcImg(uint orientation,uint anotherDim,uint pos){
    vec2 ret;
    if(orientation==0){
        ret = imageLoad(GetUAVImage2DRGBA32F(pc.srcImg),ivec2(anotherDim,pos)).rg;
    }else{
        if(pc.dftMode==0){
        //if(pc.curIter==0 && pc.dftMode==0){
            if((pos>=1.0*pc.rtW/pc.downscaleFactor || anotherDim>=1.0*pc.rtH/pc.downscaleFactor)){
                return vec2(0.0);
            }
            vec2 halfPixel = vec2(0.5/float(pc.rtW/pc.downscaleFactor),0.5/float(pc.rtH/pc.downscaleFactor));
            vec2 uv = vec2(float(pos)/float(pc.rtW/pc.downscaleFactor),float(anotherDim)/float(pc.rtH/pc.downscaleFactor));
            uv = uv + halfPixel;
            ret = texture(GetSampler2D(pc.rawSampImg),uv).rg;
            ret.g = 0.0;
        }else{
            ret = imageLoad(GetUAVImage2DRGBA32F(pc.srcImg),ivec2(pos,anotherDim)).rg;
        }
    }

    return ret;
}

void  storeDstImg(uint orientation,uint anotherDim,uint pos,vec2 val){
    if(orientation==0){
        imageStore(GetUAVImage2DRGBA32F(pc.dstImg),ivec2(anotherDim,pos),vec4(val,0.0,0.0));
    }else{
        imageStore(GetUAVImage2DRGBA32F(pc.dstImg),ivec2(pos,anotherDim),vec4(val,0.0,0.0));
    }
}

// I do not know whether this is a good idea to use shared memory in compute shader
// Large shared memory may reduce the occupancy of the GPU. Also, bank conflict may occur.
// However, writing back to global memory is also slow, with many dispatches.
// Profile result:
// 1. Using shared memory + Single dispatch: 240 microseconds
// 2. Using global memory + Mutliple dispatches : 1200 microseconds
shared vec2 sharedMem[kBlockSize*4];

vec2 loadShared(uint iter,uint pos){
    return sharedMem[(iter%2)*(kBlockSize*2)+pos];
}

void storeShared(uint iter,uint pos,vec2 val){
    sharedMem[(iter%2)*(kBlockSize*2)+pos] = val;
}

void stockhamFFT(uint unitId,uint anotherDim,uint iterId,float fftProc,float fftScale){
    uint width = 1<<pc.logW;
    uint height = 1<<pc.logH;
    uint HnWid = width/2;
    uint HnHeight = height/2;

    uint stride = (1<<iterId);
    uint di = unitId/stride;
    uint dj = unitId%stride;
    uint lpos = di * stride;
    uint lpos2 = di * stride * 2;
    vec2 vL = loadShared(iterId,lpos+dj);
    vec2 vR = loadShared(iterId,lpos+dj+(pc.orientation==1?HnWid:HnHeight));

    float vLr = vL.r;
    float vLi = vL.g;
    float vRr = vR.r;
    float vRi = vR.g;

    float c = cos(float(fftProc)*2.0*kPI*dj/(float(stride)*2.0));
    float s = sin(float(fftProc)*2.0*kPI*dj/(float(stride)*2.0));

    float tLr = (vLr + vRr*c - vRi*s)/float(fftScale);
    float tLi = (vLi + vRi*c + vRr*s)/float(fftScale);
    float tRr = (vLr - vRr*c + vRi*s)/float(fftScale);
    float tRi = (vLi - vRi*c - vRr*s)/float(fftScale);

    storeShared(iterId+1,lpos2+dj,vec2(tLr,tLi));
    storeShared(iterId+1,lpos2+dj+stride,vec2(tRr,tRi));
}


void main(){
    uint width = 1<<pc.logW;
    uint height = 1<<pc.logH;
    uint HnWid = width/2;
    uint HnHeight = height/2;
    uint unitId = gl_GlobalInvocationID.x;
    uint anotherDim = gl_GlobalInvocationID.y;
    if(pc.orientation==1&&unitId>=HnWid) return;
    if(pc.orientation==0&&unitId>=HnHeight) return;

    if(pc.orientation==0){
        float fftProc = (pc.dftMode==1)?1.0:-1.0;
        float fftScale = (pc.dftMode==1)?2.0:1.0;
        vec2 pa = loadSrcImg(pc.orientation,anotherDim,unitId);
        vec2 pb = loadSrcImg(pc.orientation,anotherDim,unitId+HnHeight);
        storeShared(0,unitId,pa);
        storeShared(0,unitId+HnHeight,pb);
        barrier();

        for(uint i=0;i<pc.logH;i++){
            // do fft
            stockhamFFT(unitId,anotherDim,i,fftProc,fftScale);
            barrier();
        }
        vec2 da = loadShared(pc.logH,unitId);
        vec2 db = loadShared(pc.logH,unitId+HnHeight);
        storeDstImg(pc.orientation,anotherDim,unitId,da);
        storeDstImg(pc.orientation,anotherDim,unitId+HnHeight,db);
    
    }else if(pc.orientation==1){
        float fftProc = (pc.dftMode==1)?1.0:-1.0;
        float fftScale = (pc.dftMode==1)?2.0:1.0;
        vec2 pa = loadSrcImg(pc.orientation,anotherDim,unitId);
        vec2 pb = loadSrcImg(pc.orientation,anotherDim,unitId+HnWid);
        storeShared(0,unitId,pa);
        storeShared(0,unitId+HnWid,pb);
        barrier();

        for(uint i=0;i<pc.logW;i++){
            stockhamFFT(unitId,anotherDim,i,fftProc,fftScale);
            barrier();
        }
        vec2 da = loadShared(pc.logW,unitId);
        vec2 db = loadShared(pc.logW,unitId+HnWid);
        storeDstImg(pc.orientation,anotherDim,unitId,da);
        storeDstImg(pc.orientation,anotherDim,unitId+HnWid,db);
    }
}
