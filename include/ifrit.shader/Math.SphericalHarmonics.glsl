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

struct MTwoBandSH{
    vec4 m_Coef;
};

struct MTwoBandSH_RGB{
    MTwoBandSH m_R;
    MTwoBandSH m_G;
    MTwoBandSH m_B;
}

MTwoBandSH ifrit_SHBasis2Encode(vec3 dir){
    // Follows the Unreal's convention. References:
    // https://www.ppsloan.org/publications/StupidSH36.pdf
    MTwoBandSH sh;
    sh.m_Coef.x = 0.282095f;
    sh.m_Coef.y = -0.488603f * dir.y;
    sh.m_Coef.z = 0.488603f * dir.z;
    sh.m_Coef.w = -0.488603f * dir.x;
    return sh;
}


MTwoBandSH ifrit_SHCosineLobe2Encode(vec3 dir){
    MTwoBandSH sh;
    sh.m_Coef.x = 0.886227f;  
    sh.m_Coef.y = -1.023327f * dir.y;  
    sh.m_Coef.z = 1.023327f * dir.z;   
    sh.m_Coef.w = -1.023327f * dir.x;  
    return sh;
}

MTwoBandSH ifrit_MulSH2(MTwoBandSH sh, float scalar){
    MTwoBandSH result;
    result.m_Coef.x = sh.m_Coef.x * scalar;
    result.m_Coef.y = sh.m_Coef.y * scalar;
    result.m_Coef.z = sh.m_Coef.z * scalar;
    result.m_Coef.w = sh.m_Coef.w * scalar;
    return result;
}

MTwoBandSH ifrit_AddSH2(MTwoBandSH sh1, MTwoBandSH sh2){
    MTwoBandSH result;
    result.m_Coef.x = sh1.m_Coef.x + sh2.m_Coef.x;
    result.m_Coef.y = sh1.m_Coef.y + sh2.m_Coef.y;
    result.m_Coef.z = sh1.m_Coef.z + sh2.m_Coef.z;
    result.m_Coef.w = sh1.m_Coef.w + sh2.m_Coef.w;
    return result;
}

MTwoBandSH_RGB ifrit_MulSH2RGB(MTwoBandSH_RGB sh, float scalar){
    MTwoBandSH_RGB result;
    result.m_R = ifrit_MulSH2(sh.m_R, scalar);
    result.m_G = ifrit_MulSH2(sh.m_G, scalar);
    result.m_B = ifrit_MulSH2(sh.m_B, scalar);
    return result;
}

MTwoBandSH_RGB ifrit_AddSH2RGB(MTwoBandSH_RGB sh1, MTwoBandSH_RGB sh2){
    MTwoBandSH_RGB result;
    result.m_R = ifrit_AddSH2(sh1.m_R, sh2.m_R);
    result.m_G = ifrit_AddSH2(sh1.m_G, sh2.m_G);
    result.m_B = ifrit_AddSH2(sh1.m_B, sh2.m_B);
    return result;
}

float ifrit_DotSH2(MTwoBandSH sh1, MTwoBandSH sh2){
    float ret = dot(sh1.m_Coef, sh2.m_Coef);
    return ret;
}

vec3 ifrit_DotSH2RGB(MTwoBandSH_RGB sh1, MTwoBandSH_RGB sh2){
    vec3 ret = vec3(
        dot(sh1.m_R.m_Coef, sh2.m_R.m_Coef),
        dot(sh1.m_G.m_Coef, sh2.m_G.m_Coef),
        dot(sh1.m_B.m_Coef, sh2.m_B.m_Coef)
    );
    return ret;
}