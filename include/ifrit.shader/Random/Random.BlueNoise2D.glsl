vec4 ifrit_bnoise2d(uint texId,vec2 uv){
    return texture(GetSampler2D(texId),uv);
}