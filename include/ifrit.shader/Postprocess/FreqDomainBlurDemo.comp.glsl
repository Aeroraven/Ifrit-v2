#version 450
#include "Base.glsl"
#include "Bindless.glsl"

layout(push_constant) uniform PushConstFFTBlur{
    uint rtW;
    uint rtH;
    uint srcImg;
    uint kernelImg;
    uint dstImg;
}pc;

layout(local_size_x=16,local_size_y=16,local_size_z=1) in;

// Note that fftshift is not implemented in stockham fft.
// So, the convolution kernel should be inverse-fftshifted before the 
// element-wise multiplication with the frequency domain image.

float demoKernel(float x,float y){
    // (x-0.5)^2 + (y-0.5)^2 = 0.2^2
    // Lerp range: radius=0.1=strength=1; radius=0.3=strength=0
    float dist = (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5);
    dist = sqrt(dist);
    float ret = 1.0 - (dist-0.1)/0.2;
    return ret;
}

void main(){
    uint px = gl_GlobalInvocationID.x;
    uint py = gl_GlobalInvocationID.y;
    if(px>=pc.rtW || py>=pc.rtH){
        return;
    }
    vec2 uv = vec2(float(px)/float(pc.rtW),float(py)/float(pc.rtH));
    vec2 shift = vec2(0.5/float(pc.rtW),0.5/float(pc.rtH));
    uv = uv + shift;

    float kernelValX = demoKernel(uv.x,uv.y);
    vec2 kernelVal = vec2(kernelValX);
    //TEST
    imageStore(GetUAVImage2DRGBA32F(pc.dstImg),ivec2(px,py),vec4(kernelVal,0.0,0.0));
}