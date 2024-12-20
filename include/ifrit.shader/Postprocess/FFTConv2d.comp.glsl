
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

const uint kStepDFT1 = 0;
const uint kStepDFT2 = 1;
const uint kStepKernDFT1 = 2;
const uint kStepKernDFT2 = 3;
const uint kStepMultiply = 4;
const uint kStepIDFT2 = 5;
const uint kStepIDFT1 = 6;

const uint kBlockSize = 256;
layout(local_size_x=kBlockSize,local_size_y=1,local_size_z=1) in;

float getSobelKernel(uint x,uint y){
    if(x==0){
        if(y==0){
            return 1.0;
        }else if(y==1){
            return 2.0;
        }else if(y==2){
            return 1.0;
        }
    }else if(x==1){
        if(y==0){
            return 0.0;
        }else if(y==1){
            return 0.0;
        }else if(y==2){
            return 0.0;
        }
    }else if(x==2){
        if(y==0){
            return -1.0;
        }else if(y==1){
            return -2.0;
        }else if(y==2){
            return -1.0;
        }
    }
    return 0.0;
}

shared vec2 fftSharedMem[kBlockSize*4];
vec2 loadFFTShared(uint iter,uint pos){
    return fftSharedMem[(iter%2)*(kBlockSize*2)+pos];
}
void storeFFTShared(uint iter,uint pos,vec2 val){
    fftSharedMem[(iter%2)*(kBlockSize*2)+pos] = val;
}
void stockhamFFTImpl(uint ori,uint logW,uint logH,uint unitId,uint anotherDim,uint iterId,float fftProc,float fftScale){
    uint width = 1<<logW;
    uint height = 1<<logH;
    uint HnWid = width/2;
    uint HnHeight = height/2;

    uint stride = (1<<iterId);
    uint di = unitId/stride;
    uint dj = unitId%stride;
    uint lpos = di * stride;
    uint lpos2 = di * stride * 2;
    vec2 vL = loadFFTShared(iterId,lpos+dj);
    vec2 vR = loadFFTShared(iterId,lpos+dj+(ori==1?HnWid:HnHeight));

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

    storeFFTShared(iterId+1,lpos2+dj,vec2(tLr,tLi));
    storeFFTShared(iterId+1,lpos2+dj+stride,vec2(tRr,tRi));
}

vec4 loadImageWithPaddings(uint imgId,uint downscale,uint rtW,uint rtH,uvec4 pads,uvec2 coord,vec2 shift){
    // LR,TB: pad
    if(imgId==0){
        // loads from sobel kernel
        int fftW = 1<<pc.fftTexSizeWLog;
        int fftH = 1<<pc.fftTexSizeHLog;
        int shiftedX = int(coord.x)-int(pads.x)+fftW/2+fftW;
        int shiftedY = int(coord.y)-int(pads.z)+fftH/2+fftH;
        shiftedX%=fftW;
        shiftedY%=fftH;
        if(shiftedX>=rtW/downscale||shiftedY>=rtH/downscale){
            return vec4(0.0);
        }
        return vec4(getSobelKernel(uint(shiftedX),uint(shiftedY)),vec3(0.0));
    }
    float sampX = float(coord.x-pads.x)/float(rtW/downscale);
    float sampY = float(coord.y-pads.z)/float(rtH/downscale);
    float halfPixelX = 0.5/float(rtW/downscale);
    float halfPixelY = 0.5/float(rtH/downscale);
    vec2 uv = vec2(sampX+halfPixelX,sampY+halfPixelY);
    if(uv.x<0.0||uv.x>1.0||uv.y<0.0||uv.y>1.0){
        return vec4(0.0);
    }
    uv += shift;
    uv = fract(uv);
    vec4 rt = texture(GetSampler2D(imgId),uv);
    rt.g=0;
    return rt;
}

vec4 directImageLoad(uint imgId,uvec2 coord){
    return imageLoad(GetUAVImage2DRGBA32F(imgId),ivec2(coord));
}

void stockhamFFTImplWrapper(uint ori,uint logW,uint logH,uint unitId,uint anotherDim,bool isIdft){
    if(ori==0){
        float fftProc = (isIdft)?1.0:-1.0;
        float fftScale = (isIdft)?2.0:1.0;
        for(uint i=0;i<logH;i++){
            stockhamFFTImpl(ori,logW,logH,unitId,anotherDim,i,fftProc,fftScale);
            barrier();
        }
    }else if(ori==1){
        float fftProc = (isIdft)?1.0:-1.0;
        float fftScale = (isIdft)?2.0:1.0;
        for(uint i=0;i<logW;i++){
            stockhamFFTImpl(ori,logW,logH,unitId,anotherDim,i,fftProc,fftScale);
            barrier();
        }
    }
}

void stockhamFFTFromImageImpl0(uint imgId,uint outImgId, uint downscale,uint texW,uint texH,
                        uint logfftW, uint logfftH, bool isIdft, bool isIfftShift,uint dirload){
    uint fftW = 1<<logfftW;
    uint fftH = 1<<logfftH;
    uint padL = (fftW-texW/downscale)/2;
    uint padR = fftW - texW/downscale - padL;
    uint padT = (fftH-texH/downscale)/2;
    uint padB = fftH - texH/downscale - padT;
    uvec4 pads = uvec4(padL,padR,padT,padB);

    vec2 shift = (isIfftShift)?vec2(0.5,0.5):vec2(0.0,0.0);
    // first orientation = 0
    uint gX = gl_GlobalInvocationID.x;
    uint gY = gl_GlobalInvocationID.y;

    vec4 pa;
    vec4 pb;
    if(dirload==0){
        pa = loadImageWithPaddings(imgId,downscale,texW,texH,pads,uvec2(gY,gX),shift);
        pb = loadImageWithPaddings(imgId,downscale,texW,texH,pads,uvec2(gY,gX+fftH/2),shift);
    }else{
        pa = directImageLoad(imgId,uvec2(gY,gX));
        pb = directImageLoad(imgId,uvec2(gY,gX+fftH/2));
    }
    
    storeFFTShared(0,gX,vec2(pa.r,pa.g));
    storeFFTShared(0,gX+fftH/2,vec2(pb.r,pb.g));

    barrier();
    stockhamFFTImplWrapper(0,logfftW,logfftH,gX,gY,isIdft);
    vec2 da = loadFFTShared(logfftH,gX);
    vec2 db = loadFFTShared(logfftH,gX+fftH/2);
    uvec2 coord1 = uvec2(gY,gX);
    uvec2 coord2 = uvec2(gY,gX+fftH/2);
    imageStore(GetUAVImage2DRGBA32F(outImgId),ivec2(coord1),vec4(da.r,da.g,0.0,0.0));
    imageStore(GetUAVImage2DRGBA32F(outImgId),ivec2(coord2),vec4(db.r,db.g,0.0,0.0));
    barrier();
}

void stockhamFFTFromImageImpl1(uint imgId,uint outImgId, uint downscale,uint texW,uint texH,
                        uint logfftW, uint logfftH, bool isIdft, bool isIfftShift,uint dirload){
    uint fftW = 1<<logfftW;
    uint fftH = 1<<logfftH;
    uint padL = (fftW-texW/downscale)/2;
    uint padR = fftW - texW/downscale - padL;
    uint padT = (fftH-texH/downscale)/2;
    uint padB = fftH - texH/downscale - padT;
    uvec4 pads = uvec4(padL,padR,padT,padB);

    vec2 shift = (isIfftShift)?vec2(0.5,0.5):vec2(0.0,0.0);
    // then orientation = 1
    uint gX = gl_GlobalInvocationID.x;
    uint gY = gl_GlobalInvocationID.y;

    // pa = loadImageWithPaddings(outImgId,downscale,fftH,fftW,pads,uvec2(gX,gY),shift);
    // pb = loadImageWithPaddings(outImgId,downscale,fftH,fftW,pads,uvec2(gX+fftW/2,gY),shift);s
    vec4 pa;
    vec4 pb;
    if(dirload==0){
        pa = loadImageWithPaddings(imgId,downscale,texW,texH,pads,uvec2(gX,gY),shift);
        pb = loadImageWithPaddings(imgId,downscale,texW,texH,pads,uvec2(gX+fftW/2,gY),shift);
    }else{
        pa = directImageLoad(imgId,uvec2(gX,gY));
        pb = directImageLoad(imgId,uvec2(gX+fftW/2,gY));
    }

    storeFFTShared(0,gX,vec2(pa.r,pa.g));
    storeFFTShared(0,gX+fftW/2,vec2(pb.r,pb.g));

    barrier();
    //if(imgId!=0 && !isIdft){
        stockhamFFTImplWrapper(1,logfftW,logfftH,gX,gY,isIdft);
    //}
    vec2 da = loadFFTShared(logfftW,gX);
    vec2 db = loadFFTShared(logfftW,gX+fftW/2);
    uvec2 coord1 = uvec2(gX,gY);
    uvec2 coord2 = uvec2(gX+fftW/2,gY);
    // if(imgId==0 && !isIdft){
    //     imageStore(GetUAVImage2DRGBA32F(outImgId),ivec2(coord1),vec4(pa.r,pa.g,0.0,0.0));
    //     imageStore(GetUAVImage2DRGBA32F(outImgId),ivec2(coord2),vec4(pb.r,pb.g,0.0,0.0));
    //     return;
    // }
    imageStore(GetUAVImage2DRGBA32F(outImgId),ivec2(coord1),vec4(da.r,da.g,0.0,0.0));
    imageStore(GetUAVImage2DRGBA32F(outImgId),ivec2(coord2),vec4(db.r,db.g,0.0,0.0));
    barrier();
}
void stockhamFFTFromImage(uint imgId,uint outImgId,uint tempImgId, uint downscale,uint texW,uint texH,
                        uint logfftW, uint logfftH, bool isIdft, bool isIfftShift, uint ori){
    if(!isIdft){
        if(ori==1){
            stockhamFFTFromImageImpl1(imgId,tempImgId,downscale,texW,texH,logfftW,logfftH,isIdft,isIfftShift,0);
        }else{
            stockhamFFTFromImageImpl0(tempImgId,outImgId,downscale,texW,texH,logfftW,logfftH,isIdft,isIfftShift,1);
        }
    }else{
        if(ori==0){
            stockhamFFTFromImageImpl0(outImgId,tempImgId,downscale,texW,texH,logfftW,logfftH,isIdft,isIfftShift,1);
        }else{
            stockhamFFTFromImageImpl1(tempImgId,imgId,downscale,texW,texH,logfftW,logfftH,isIdft,isIfftShift,1);
        }
    }
}

void freqDomainMultiply(uint imgId1,uint imgId2,uint outImgId,uint fftW,uint fftH){
    uint gX = gl_GlobalInvocationID.x;
    uint gY = gl_GlobalInvocationID.y;
    ivec2 coord1 = ivec2(gX,gY);
    ivec2 coord2 = ivec2(gX+fftW/2,gY);
    vec4 pa = imageLoad(GetUAVImage2DRGBA32F(imgId1),coord1);
    vec4 pb = imageLoad(GetUAVImage2DRGBA32F(imgId2),coord1);
    vec4 pc = imageLoad(GetUAVImage2DRGBA32F(imgId1),coord2);
    vec4 pd = imageLoad(GetUAVImage2DRGBA32F(imgId2),coord2);

    float paR = pa.r;
    float paI = pa.g;
    float pbR = pb.r;
    float pbI = pb.g;
    float pcR = pc.r;
    float pcI = pc.g;
    float pdR = pd.r;
    float pdI = pd.g;

    float raR = paR*pbR - paI*pbI;
    float raI = paR*pbI + paI*pbR;
    float rbR = pcR*pdR - pcI*pdI;
    float rbI = pcR*pdI + pcI*pdR;

    vec4 ra = vec4(raR,raI,0.0,0.0);
    vec4 rb = vec4(rbR,rbI,0.0,0.0);

    imageStore(GetUAVImage2DRGBA32F(outImgId),coord1,ra);
    imageStore(GetUAVImage2DRGBA32F(outImgId),coord2,rb);
}

void main(){
    if(pc.fftStep==kStepDFT1){
        stockhamFFTFromImage(pc.srcImage,pc.srcIntermImage,pc.tempImage,pc.srcDownScale,
            pc.srcRtW,pc.srcRtH,pc.fftTexSizeWLog,pc.fftTexSizeHLog,false,false,1);
    }else if(pc.fftStep==kStepDFT2){
        stockhamFFTFromImage(pc.srcImage,pc.srcIntermImage,pc.tempImage,pc.srcDownScale,
            pc.srcRtW,pc.srcRtH,pc.fftTexSizeWLog,pc.fftTexSizeHLog,false,false,0);
    }else if(pc.fftStep==kStepKernDFT1){
        stockhamFFTFromImage(pc.kernImage,pc.kernIntermImage,pc.tempImage,pc.kernDownScale,
            pc.kernRtW,pc.kernRtH,pc.fftTexSizeWLog,pc.fftTexSizeHLog,false,false,1);
    }else if(pc.fftStep==kStepKernDFT2){
        stockhamFFTFromImage(pc.kernImage,pc.kernIntermImage,pc.tempImage,pc.kernDownScale,
            pc.kernRtW,pc.kernRtH,pc.fftTexSizeWLog,pc.fftTexSizeHLog,false,false,0);
    }else if(pc.fftStep==kStepMultiply){
        uint fftW = 1<<pc.fftTexSizeWLog;
        uint fftH = 1<<pc.fftTexSizeHLog;
        freqDomainMultiply(pc.srcIntermImage,pc.kernIntermImage,pc.srcIntermImage,fftW,fftH);
    }else if(pc.fftStep==kStepIDFT2){
        stockhamFFTFromImage(pc.kernIntermImage,pc.srcIntermImage,pc.tempImage,pc.srcDownScale,
            pc.srcRtW,pc.srcRtH,pc.fftTexSizeWLog,pc.fftTexSizeHLog,true,false,0);
    }else if(pc.fftStep==kStepIDFT1){
        stockhamFFTFromImage(pc.kernIntermImage,pc.srcIntermImage,pc.tempImage,pc.srcDownScale,
            pc.srcRtW,pc.srcRtH,pc.fftTexSizeWLog,pc.fftTexSizeHLog,true,false,1);
    }
}