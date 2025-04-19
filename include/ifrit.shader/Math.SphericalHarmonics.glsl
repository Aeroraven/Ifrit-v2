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

struct MThreeBandSH{
    vec4 m_Coef1;
    vec4 m_Coef2;
    float m_Coef3;
};

struct MTwoBandSH_RGB{
    MTwoBandSH m_R;
    MTwoBandSH m_G;
    MTwoBandSH m_B;
};

struct MThreeBandSH_RGB{
    MThreeBandSH m_R;
    MThreeBandSH m_G;
    MThreeBandSH m_B;
};

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

MThreeBandSH ifrit_SHBasis3Encode(vec3 dir){
    MThreeBandSH sh;
    sh.m_Coef1.x = 0.282095f;
    sh.m_Coef1.y = -0.488603f * dir.y;
    sh.m_Coef1.z = 0.488603f * dir.z;
    sh.m_Coef1.w = -0.488603f * dir.x;
    sh.m_Coef2.x = 1.092548f * dir.x * dir.y;
    sh.m_Coef2.y = -1.092548f * dir.y * dir.z;
    sh.m_Coef2.z = 0.315392f * (3.0f * dir.z * dir.z - 1.0f);
    sh.m_Coef2.w = -1.092548f * dir.x * dir.z;
    sh.m_Coef3 = 0.546274f * (dir.x * dir.x - dir.y * dir.y);
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

MThreeBandSH ifrit_SHCosineLobe3Encode(vec3 dir){
    float PiDiv4 = 0.785398f;
    MThreeBandSH sh;
    sh.m_Coef1.x = 0.886227f;
    sh.m_Coef1.y = -1.023327f * dir.y;
    sh.m_Coef1.z = 1.023327f * dir.z;
    sh.m_Coef1.w = -1.023327f * dir.x;

    sh.m_Coef2.x = 1.092548f * dir.x * dir.y * PiDiv4;
    sh.m_Coef2.y = -1.092548f * dir.y * dir.z * PiDiv4;
    sh.m_Coef2.z = 0.315392f * (3.0f * dir.z * dir.z - 1.0f) * PiDiv4;
    sh.m_Coef2.w = -1.092548f * dir.x * dir.z * PiDiv4;
    sh.m_Coef3 = 0.546274f * (dir.x * dir.x - dir.y * dir.y) * PiDiv4;
    return sh;
}

MTwoBandSH_RGB ifrit_SHBasis2EncodeRGB(vec3 dir){
    MTwoBandSH_RGB sh;
    sh.m_R = ifrit_SHBasis2Encode(dir);
    sh.m_G = ifrit_SHBasis2Encode(dir);
    sh.m_B = ifrit_SHBasis2Encode(dir);
    return sh;
}

MThreeBandSH_RGB ifrit_SHBasis3EncodeRGB(vec3 dir){
    MThreeBandSH_RGB sh;
    sh.m_R = ifrit_SHBasis3Encode(dir);
    sh.m_G = ifrit_SHBasis3Encode(dir);
    sh.m_B = ifrit_SHBasis3Encode(dir);
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

MThreeBandSH ifrit_MulSH3(MThreeBandSH sh, float scalar){
    MThreeBandSH result;
    result.m_Coef1.x = sh.m_Coef1.x * scalar;
    result.m_Coef1.y = sh.m_Coef1.y * scalar;
    result.m_Coef1.z = sh.m_Coef1.z * scalar;
    result.m_Coef1.w = sh.m_Coef1.w * scalar;
    result.m_Coef2.x = sh.m_Coef2.x * scalar;
    result.m_Coef2.y = sh.m_Coef2.y * scalar;
    result.m_Coef2.z = sh.m_Coef2.z * scalar;
    result.m_Coef2.w = sh.m_Coef2.w * scalar;
    result.m_Coef3 = sh.m_Coef3 * scalar;
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

MThreeBandSH ifrit_AddSH3(MThreeBandSH sh1, MThreeBandSH sh2){
    MThreeBandSH result;
    result.m_Coef1.x = sh1.m_Coef1.x + sh2.m_Coef1.x;
    result.m_Coef1.y = sh1.m_Coef1.y + sh2.m_Coef1.y;
    result.m_Coef1.z = sh1.m_Coef1.z + sh2.m_Coef1.z;
    result.m_Coef1.w = sh1.m_Coef1.w + sh2.m_Coef1.w;
    result.m_Coef2.x = sh1.m_Coef2.x + sh2.m_Coef2.x;
    result.m_Coef2.y = sh1.m_Coef2.y + sh2.m_Coef2.y;
    result.m_Coef2.z = sh1.m_Coef2.z + sh2.m_Coef2.z;
    result.m_Coef2.w = sh1.m_Coef2.w + sh2.m_Coef2.w;
    result.m_Coef3 = sh1.m_Coef3 + sh2.m_Coef3;
    return result;
}

MTwoBandSH_RGB ifrit_MulSH2RGB(MTwoBandSH_RGB sh, float scalar){
    MTwoBandSH_RGB result;
    result.m_R = ifrit_MulSH2(sh.m_R, scalar);
    result.m_G = ifrit_MulSH2(sh.m_G, scalar);
    result.m_B = ifrit_MulSH2(sh.m_B, scalar);
    return result;
}

MThreeBandSH_RGB ifrit_MulSH3RGB(MThreeBandSH_RGB sh, float scalar){
    MThreeBandSH_RGB result;
    result.m_R = ifrit_MulSH3(sh.m_R, scalar);
    result.m_G = ifrit_MulSH3(sh.m_G, scalar);
    result.m_B = ifrit_MulSH3(sh.m_B, scalar);
    return result;
}

MTwoBandSH_RGB ifrit_MulSH2RGBColor(MTwoBandSH_RGB sh, vec3 color){
    MTwoBandSH_RGB result;
    result.m_R = ifrit_MulSH2(sh.m_R, color.r);
    result.m_G = ifrit_MulSH2(sh.m_G, color.g);
    result.m_B = ifrit_MulSH2(sh.m_B, color.b);
    return result;
}

MThreeBandSH_RGB ifrit_MulSH3RGBColor(MThreeBandSH_RGB sh, vec3 color){
    MThreeBandSH_RGB result;
    result.m_R = ifrit_MulSH3(sh.m_R, color.r);
    result.m_G = ifrit_MulSH3(sh.m_G, color.g);
    result.m_B = ifrit_MulSH3(sh.m_B, color.b);
    return result;
}

MTwoBandSH_RGB ifrit_AddSH2RGB(MTwoBandSH_RGB sh1, MTwoBandSH_RGB sh2){
    MTwoBandSH_RGB result;
    result.m_R = ifrit_AddSH2(sh1.m_R, sh2.m_R);
    result.m_G = ifrit_AddSH2(sh1.m_G, sh2.m_G);
    result.m_B = ifrit_AddSH2(sh1.m_B, sh2.m_B);
    return result;
}

MThreeBandSH_RGB ifrit_AddSH3RGB(MThreeBandSH_RGB sh1, MThreeBandSH_RGB sh2){
    MThreeBandSH_RGB result;
    result.m_R = ifrit_AddSH3(sh1.m_R, sh2.m_R);
    result.m_G = ifrit_AddSH3(sh1.m_G, sh2.m_G);
    result.m_B = ifrit_AddSH3(sh1.m_B, sh2.m_B);
    return result;
}

float ifrit_DotSH2(MTwoBandSH sh1, MTwoBandSH sh2){
    float ret = dot(sh1.m_Coef, sh2.m_Coef);
    return ret;
}

float ifrit_DotSH3(MThreeBandSH sh1, MThreeBandSH sh2){
    float ret = dot(sh1.m_Coef1, sh2.m_Coef1) + dot(sh1.m_Coef2, sh2.m_Coef2) + sh1.m_Coef3 * sh2.m_Coef3;
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

vec3 ifrit_DotSH3RGB(MThreeBandSH_RGB sh1, MThreeBandSH_RGB sh2){
    vec3 ret = vec3(
        dot(sh1.m_R.m_Coef1, sh2.m_R.m_Coef1) + dot(sh1.m_R.m_Coef2, sh2.m_R.m_Coef2) + sh1.m_R.m_Coef3 * sh2.m_R.m_Coef3,
        dot(sh1.m_G.m_Coef1, sh2.m_G.m_Coef1) + dot(sh1.m_G.m_Coef2, sh2.m_G.m_Coef2) + sh1.m_G.m_Coef3 * sh2.m_G.m_Coef3,
        dot(sh1.m_B.m_Coef1, sh2.m_B.m_Coef1) + dot(sh1.m_B.m_Coef2, sh2.m_B.m_Coef2) + sh1.m_B.m_Coef3 * sh2.m_B.m_Coef3
    );
    return ret;
}

MTwoBandSH_RGB ifrit_ZeroSH2RGB(){
    MTwoBandSH_RGB sh;
    sh.m_R.m_Coef = vec4(0.0);
    sh.m_G.m_Coef = vec4(0.0);
    sh.m_B.m_Coef = vec4(0.0);
    return sh;
}

MThreeBandSH_RGB ifrit_ZeroSH3RGB(){
    MThreeBandSH_RGB sh;
    sh.m_R.m_Coef1 = vec4(0.0);
    sh.m_R.m_Coef2 = vec4(0.0);
    sh.m_R.m_Coef3 = 0.0;
    sh.m_G.m_Coef1 = vec4(0.0);
    sh.m_G.m_Coef2 = vec4(0.0);
    sh.m_G.m_Coef3 = 0.0;
    sh.m_B.m_Coef1 = vec4(0.0);
    sh.m_B.m_Coef2 = vec4(0.0);
    sh.m_B.m_Coef3 = 0.0;
    return sh;
}