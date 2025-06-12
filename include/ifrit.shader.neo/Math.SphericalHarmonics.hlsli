/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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

#pragma once
#include "Common.hlsli"

namespace IfritShader {
namespace Math {

    struct FTwoBandSH
    {
        float4 m_Coef;
    };

    struct FThreeBandSH
    {
        float4 m_Coef1;
        float4 m_Coef2;
        float m_Coef3;
    };

    struct FTwoBandSHRgb
    {
        FTwoBandSH m_Red;
        FTwoBandSH m_Green;
        FTwoBandSH m_Blue;
    };

    struct FThreeBandSHRgb
    {
        FThreeBandSH m_Red;
        FThreeBandSH m_Green;
        FThreeBandSH m_Blue;
    };

    FTwoBandSH SHBasis2Encode(float3 Dir)
    {
        // Follows the Unreal's convention. References:
        // https://www.ppsloan.org/publications/StupidSH36.pdf
        FTwoBandSH sh;
        sh.m_Coef.x = 0.282095f;
        sh.m_Coef.y = -0.488603f * Dir.y;
        sh.m_Coef.z = 0.488603f * Dir.z;
        sh.m_Coef.w = -0.488603f * Dir.x;
        return sh;
    }

    FThreeBandSH SHBasis3Encode(float3 Dir)
    {
        FThreeBandSH sh;
        sh.m_Coef1.x = 0.282095f;
        sh.m_Coef1.y = -0.488603f * Dir.y;
        sh.m_Coef1.z = 0.488603f * Dir.z;
        sh.m_Coef1.w = -0.488603f * Dir.x;
        sh.m_Coef2.x = 1.092548f * Dir.x * Dir.y;
        sh.m_Coef2.y = -1.092548f * Dir.y * Dir.z;
        sh.m_Coef2.z = 0.315392f * (3.0f * Dir.z * Dir.z - 1.0f);
        sh.m_Coef2.w = -1.092548f * Dir.x * Dir.z;
        sh.m_Coef3 = 0.546274f * (Dir.x * Dir.x - Dir.y * Dir.y);
        return sh;
    }

    FTwoBandSH SHCosineLobe2Encode(float3 Dir)
    {
        FTwoBandSH sh;
        sh.m_Coef.x = 0.886227f;  
        sh.m_Coef.y = -1.023327f * Dir.y;  
        sh.m_Coef.z = 1.023327f * Dir.z;   
        sh.m_Coef.w = -1.023327f * Dir.x;  
        return sh;
    }

    FThreeBandSH SHCosineLobe3Encode(float3 Dir)
    {
        float PiDiv4 = 0.785398f; // π/4
        FThreeBandSH sh;
        sh.m_Coef1.x = 0.886227f;
        sh.m_Coef1.y = -1.023327f * Dir.y;
        sh.m_Coef1.z = 1.023327f * Dir.z;
        sh.m_Coef1.w = -1.023327f * Dir.x;

        sh.m_Coef2.x = 1.092548f * Dir.x * Dir.y * PiDiv4;
        sh.m_Coef2.y = -1.092548f * Dir.y * Dir.z * PiDiv4;
        sh.m_Coef2.z = 0.315392f * (3.0f * Dir.z * Dir.z - 1.0f) * PiDiv4;
        sh.m_Coef2.w = -1.092548f * Dir.x * Dir.z * PiDiv4;

        sh.m_Coef3 = 0.546274f * (Dir.x * Dir.x - Dir.y * Dir.y) * PiDiv4;
        return sh;
    }

    FTwoBandSHRgb SHBasis2EncodeRgb(float3 Dir)
    {
        FTwoBandSHRgb sh;
        sh.m_Red = SHBasis2Encode(Dir);
        sh.m_Green = SHBasis2Encode(Dir);
        sh.m_Blue = SHBasis2Encode(Dir);
        return sh;
    }

    FThreeBandSHRgb SHBasis3EncodeRgb(float3 Dir)
    {
        FThreeBandSHRgb sh;
        sh.m_Red = SHBasis3Encode(Dir);
        sh.m_Green = SHBasis3Encode(Dir);
        sh.m_Blue = SHBasis3Encode(Dir);
        return sh;
    }

    FTwoBandSH MulSH2(FTwoBandSH sh, float scalar)
    {
        FTwoBandSH result;
        result.m_Coef = sh.m_Coef * scalar;
        return result;
    }

    FThreeBandSH MulSH3(FThreeBandSH sh, float scalar)
    {
        FThreeBandSH result;
        result.m_Coef1 = sh.m_Coef1 * scalar;
        result.m_Coef2 = sh.m_Coef2 * scalar;
        result.m_Coef3 = sh.m_Coef3 * scalar;
        return result;
    }

    FTwoBandSH AddSH2(FTwoBandSH sh1, FTwoBandSH sh2)
    {
        FTwoBandSH result;
        result.m_Coef = sh1.m_Coef + sh2.m_Coef;
        return result;
    }

    FThreeBandSH AddSH3(FThreeBandSH sh1, FThreeBandSH sh2)
    {
        FThreeBandSH result;
        result.m_Coef1 = sh1.m_Coef1 + sh2.m_Coef1;
        result.m_Coef2 = sh1.m_Coef2 + sh2.m_Coef2;
        result.m_Coef3 = sh1.m_Coef3 + sh2.m_Coef3;
        return result;
    }

    FTwoBandSHRgb MulSH2(FTwoBandSHRgb sh, float scalar)
    {
        FTwoBandSHRgb result;
        result.m_Red = MulSH2(sh.m_Red, scalar);
        result.m_Green = MulSH2(sh.m_Green, scalar);
        result.m_Blue = MulSH2(sh.m_Blue, scalar);
        return result;
    }

    FThreeBandSHRgb MulSH3(FThreeBandSHRgb sh, float scalar)
    {
        FThreeBandSHRgb result;
        result.m_Red = MulSH3(sh.m_Red, scalar);
        result.m_Green = MulSH3(sh.m_Green, scalar);
        result.m_Blue = MulSH3(sh.m_Blue, scalar);
        return result;
    }

    FTwoBandSHRgb MulSH2Color(FTwoBandSHRgb sh, float3 color)
    {
        FTwoBandSHRgb result;
        result.m_Red = MulSH2(sh.m_Red, color.r);
        result.m_Green = MulSH2(sh.m_Green, color.g);
        result.m_Blue = MulSH2(sh.m_Blue, color.b);
        return result;
    }

    FThreeBandSHRgb MulSH3Color(FThreeBandSHRgb sh, float3 color)
    {
        FThreeBandSHRgb result;
        result.m_Red = MulSH3(sh.m_Red, color.r);
        result.m_Green = MulSH3(sh.m_Green, color.g);
        result.m_Blue = MulSH3(sh.m_Blue, color.b);
        return result;
    }

    FTwoBandSHRgb AddSH2(FTwoBandSHRgb sh1, FTwoBandSHRgb sh2)
    {
        FTwoBandSHRgb result;
        result.m_Red = AddSH2(sh1.m_Red, sh2.m_Red);
        result.m_Green = AddSH2(sh1.m_Green, sh2.m_Green);
        result.m_Blue = AddSH2(sh1.m_Blue, sh2.m_Blue);
        return result;
    }

    FThreeBandSHRgb AddSH3(FThreeBandSHRgb sh1, FThreeBandSHRgb sh2)
    {
        FThreeBandSHRgb result;
        result.m_Red = AddSH3(sh1.m_Red, sh2.m_Red);
        result.m_Green = AddSH3(sh1.m_Green, sh2.m_Green);
        result.m_Blue = AddSH3(sh1.m_Blue, sh2.m_Blue);
        return result;
    }

    float DotSH2(FTwoBandSH sh1, FTwoBandSH sh2)
    {
        // Dot product of two SHs
        return dot(sh1.m_Coef, sh2.m_Coef);
    }   

    float DotSH3(FThreeBandSH sh1, FThreeBandSH sh2)
    {
        // Dot product of two SHs
        return dot(sh1.m_Coef1, sh2.m_Coef1) + dot(sh1.m_Coef2, sh2.m_Coef2) + sh1.m_Coef3 * sh2.m_Coef3;
    }

    float3 DotSH2Rgb(FTwoBandSHRgb sh1, FTwoBandSHRgb sh2)
    {
        // Dot product of two RGB SHs
        return float3(
            DotSH2(sh1.m_Red, sh2.m_Red),
            DotSH2(sh1.m_Green, sh2.m_Green),
            DotSH2(sh1.m_Blue, sh2.m_Blue)
        );
    }

    float3 DotSH3Rgb(FThreeBandSHRgb sh1, FThreeBandSHRgb sh2)
    {
        // Dot product of two RGB SHs
        return float3(
            DotSH3(sh1.m_Red, sh2.m_Red),
            DotSH3(sh1.m_Green, sh2.m_Green),
            DotSH3(sh1.m_Blue, sh2.m_Blue)
        );
    }

    FTwoBandSHRgb ZeroSH2Rgb()
    {
        FTwoBandSHRgb sh;
        sh.m_Red.m_Coef = float4(0.0f, 0.0f, 0.0f, 0.0f);
        sh.m_Green.m_Coef = float4(0.0f, 0.0f, 0.0f, 0.0f);
        sh.m_Blue.m_Coef = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return sh;
    }

    FThreeBandSHRgb ZeroSH3Rgb()
    {
        FThreeBandSHRgb sh;
        sh.m_Red.m_Coef1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        sh.m_Red.m_Coef2 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        sh.m_Red.m_Coef3 = 0.0f;
        sh.m_Green.m_Coef1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        sh.m_Green.m_Coef2 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        sh.m_Green.m_Coef3 = 0.0f;
        sh.m_Blue.m_Coef1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        sh.m_Blue.m_Coef2 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        sh.m_Blue.m_Coef3 = 0.0f;
        return sh;
    }

}
}