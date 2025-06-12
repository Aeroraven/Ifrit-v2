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
  

// Bindless.hlsli
#pragma once
#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Samplers.hlsli"

namespace IfritShader
{

#define IFSHADER_BINDLESS_CONSTANT_BUFFER 0
#define IFSHADER_BINDLESS_STRUCTURED_BUFFER 1
#define IFSHADER_BINDLESS_RW_STRUCTURED_BUFFER 2
#define IFSHADER_BINDLESS_COMBINED_IMAGE_SAMPLER 3
#define IFSHADER_BINDLESS_RW_TEXTURE 4
#define IFSHADER_BINDLESS_TEXTURE 5
#define IFSHADER_BINDLESS_SAMPLER 6

#define IFSHADER_BINDLESS_SETID 0

#define _IFSHADER_BINDLESS_NAMING(name) u##name##_bindless
#define _IFSHADER_BINDLESS_TYPE(name) u##name##_bindless_t

#ifndef COMPILER_DXC

    IFSHADER_BINDING(IFSHADER_BINDLESS_CONSTANT_BUFFER, IFSHADER_BINDLESS_SETID)
    __DynamicResource _Ifrit_ResourceHeap_ConstantBuffer[];

    IFSHADER_BINDING(IFSHADER_BINDLESS_STRUCTURED_BUFFER, IFSHADER_BINDLESS_SETID)
    __DynamicResource _Ifrit_ResourceHeap_StructuredBuffer[];

    IFSHADER_BINDING(IFSHADER_BINDLESS_RW_STRUCTURED_BUFFER, IFSHADER_BINDLESS_SETID)
    __DynamicResource _Ifrit_ResourceHeap_RWStructuredBuffer[];

    // We already dropped the support for combined image sampler in prev commits
    // So no need to define it here

    IFSHADER_BINDING(IFSHADER_BINDLESS_RW_TEXTURE, IFSHADER_BINDLESS_SETID)
    __DynamicResource _Ifrit_ResourceHeap_RWTexture[];

    IFSHADER_BINDING(IFSHADER_BINDLESS_TEXTURE, IFSHADER_BINDLESS_SETID)
    __DynamicResource _Ifrit_ResourceHeap_Texture[];

    IFSHADER_BINDING(IFSHADER_BINDLESS_SAMPLER, IFSHADER_BINDLESS_SETID)
    SamplerState _Ifrit_ResourceHeap_Sampler[];

#else
    #error "Bindless support for hlsl is not implemented yet."
#endif

    IFSHADER_TEMPLATE_STRUCT(TRWStructuredBufferHandle,T)
    {
        uint Index;

        T Load(uint Offset)
        {
            RWStructuredBuffer<T> Buffer = _Ifrit_ResourceHeap_RWStructuredBuffer[Index];
            return Buffer[Offset];
        }

        void Store(uint Offset, T Value)
        {
            RWStructuredBuffer<T> Buffer = _Ifrit_ResourceHeap_RWStructuredBuffer[Index];
            Buffer[Offset] = Value;
        }
    };

    IFSHADER_TEMPLATE_STRUCT(TStructuredBufferHandle,T)
    {
        uint Index;

        T Load(uint Offset)
        {
            StructuredBuffer<T> Buffer = _Ifrit_ResourceHeap_StructuredBuffer[Index];
            return Buffer[Offset];
        }
    };

    IFSHADER_TEMPLATE_STRUCT(TConstantBufferHandle,T)
    {
        uint Index;

        ConstantBuffer<T> Load()
        {
            ConstantBuffer<T> Buffer = _Ifrit_ResourceHeap_ConstantBuffer[Index].as<ConstantBuffer<T>>();
            return Buffer;
        }
    };

    IFSHADER_TEMPLATE_STRUCT(TRWTexture2DHandle,T)
    IFSHADER_REQUIRES(T:IFSHADER_TEXELELEMENT_TYPE)
    {
        uint Index;

        T Load(uint2 UV)
        {
            RWTexture2D<T> Texture = _Ifrit_ResourceHeap_RWTexture[Index];
            return Texture[UV];
        }

        void Store(uint2 UV, T Value)
        {
            RWTexture2D<T> Texture = _Ifrit_ResourceHeap_RWTexture[Index];
            Texture[UV] = Value;
        }
    };

    IFSHADER_TEMPLATE_STRUCT(TTexture2DHandle,T)
    IFSHADER_REQUIRES(T:IFSHADER_TEXELELEMENT_TYPE)
    {
        uint Index;

        T Sample(ESamplerType SamplerType, uint2 UV)
        {
            Texture2D<T> Texture = _Ifrit_ResourceHeap_Texture[Index];
            SamplerState Sampler = _Ifrit_ResourceHeap_Sampler[(uint)SamplerType];
            return Texture.Sample(Sampler, UV);
        }

        T Sample(ESamplerType SamplerType, float2 UV)
        {
            Texture2D<T> Texture = _Ifrit_ResourceHeap_Texture[Index];
            SamplerState Sampler = _Ifrit_ResourceHeap_Sampler[(uint)SamplerType];
            return Texture.Sample(Sampler, UV);
        }
    };
    
}