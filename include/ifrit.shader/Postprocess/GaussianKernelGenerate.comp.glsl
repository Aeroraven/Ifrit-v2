#version 450
#include "Base.glsl"
#include "Bindless.glsl"

layout(push_constant) uniform PushConstGaussionKernelGenerate{
    uint rtW;
    uint rtH;
    uint dstImg;
    float sigma;
}pc;

layout(local_size_x=16,local_size_y=16,local_size_z=1) in;

// This kernel will only be computed once. So performance is not a concern.

float kernelVal(float x,float y){
    float kernelcenterX = float(pc.rtW) / 2.0;
    float kernelcenterY = float(pc.rtH) / 2.0;
    float kernelVal = exp(-(x - kernelcenterX) * (x - kernelcenterX) / (2.0 * pc.sigma * pc.sigma) - (y - kernelcenterY) * (y - kernelcenterY) / (2.0 * pc.sigma * pc.sigma));
    return kernelVal;
}
void main(){
    //Generate a 2D gaussian kernel.
    uint px = gl_GlobalInvocationID.x;
    uint py = gl_GlobalInvocationID.y;
    if(px>=pc.rtW || py>=pc.rtH){
        return;
    }
    float kernVal = kernelVal(float(px),float(py));

    float totalKernVal = 0.0;
    for(uint i = 0;i<pc.rtW;i++){
        for(uint j = 0;j<pc.rtH;j++){
            totalKernVal += kernelVal(float(i),float(j));
        }
    }
    kernVal /= totalKernVal;
    imageStore(GetUAVImage2DRGBA32F(pc.dstImg),ivec2(px,py),vec4(kernVal,0.0,0.0,0.0));
}