vec4 ifrit_bnoise2d(uint texId,vec2 uv){
    return SampleTexture2D(texId,sLinearClamp,uv);
}