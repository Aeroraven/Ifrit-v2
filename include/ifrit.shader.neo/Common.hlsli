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
#ifndef __cplusplus

   
#define IFSHADER_DEFINE_CONST_UINT32(name, value) static const uint name = value;
#define IFSHADER_DEFINE_CONST_INT32(name, value) static const int name = value;
#define IFSHADER_DEFINE_CONST_FLOAT(name, value) static const float name = value;
#define IFSHADER_DEFINE_CONST_FLOAT2(name, value) static const float2 name = value;
#define IFSHADER_DEFINE_CONST_FLOAT3(name, value) static const float3 name = value;
#define IFSHADER_DEFINE_CONST_FLOAT4(name, value) static const float4 name = value;

#define IFSHADER_VS_ENTRY "vertex"
#define IFSHADER_PS_ENTRY "pixel"
#define IFSHADER_CS_ENTRY "compute"

#ifdef COMPILER_DXC
    #ifdef __HLSL_VERSION 
        #define IFSHADER_TEMPLATE template
        #define IFSHADER_TEMPLATE_STRUCT(x,T) template<typename T> struct x
        #define IFSHADER_REQUIRES(x)
        #define IFSHADER_REQUIRES_2(x,y)
        #define IFSHADER_ARITHMETIC_TYPE 
        #define IFSHADER_INTEGER_TYPE 
        #define IFSHADER_TEXELELEMENT_TYPE
        #define IFSHADER_UNSCOPED_ENUM 
        #define IFSHADER_FLOATVEC_TYPE //float,float2,float3,float4
        #define IFSHADER_ENTRY(x) [[shader(x)]]
        #define IFSHADER_BLOCKSIZE(x,y,z) [[numthreads(x, y, z)]]
        #ifdef IFSHADER_VULKAN
            #define IFSHADER_BINDING(binding, set) [[vk::binding(binding, set)]]
            #define IFSHADER_LOCATION(location) [[vk::location(location)]]
            #define IFSHADER_PUSHCONST [[vk::push_constant]]
        #else
            #define IFSHADER_BINDING(binding, set)
            #define IFSHADER_LOCATION(location)
            #define IFSHADER_PUSHCONST
        #endif
        #define IFSHADER_TYPEALIAS_STRUCT(name, type) struct name : type {};
    #else
        #error "This shader module is only supported in HLSL or Slang."
    #endif
#else
    #define IFSHADER_TEMPLATE __generic
    #define IFSHADER_TEMPLATE_STRUCT(x,T) struct x<T>
    #define IFSHADER_REQUIRES(x) where x
    #define IFSHADER_REQUIRES_2(x,y) where x,y
    #define IFSHADER_ARITHMETIC_TYPE IArithmetic
    #define IFSHADER_INTEGER_TYPE __BuiltinIntegerType
    #define IFSHADER_TEXELELEMENT_TYPE ITexelElement
    #define IFSHADER_UNSCOPED_ENUM [UnscopedEnum]
    #define IFSHADER_FLOATVEC_TYPE __BuiltinFloatingPointType
    #define IFSHADER_ENTRY(x) [shader(x)]
    #define IFSHADER_BLOCKSIZE(x,y,z) [numthreads(x, y, z)]
    #ifdef IFSHADER_VULKAN
        #define IFSHADER_BINDING(x, y) [vk::binding(x, y)]
        #define IFSHADER_LOCATION(x) [vk::location(x)]
        #define IFSHADER_PUSHCONST [vk::push_constant]
    #else
        #define IFSHADER_BINDING(x, y)
        #define IFSHADER_LOCATION(x)
        #define IFSHADER_PUSHCONST
    #endif
    #define IFSHADER_TYPEALIAS_STRUCT(name, type) typealias name = type;
#endif

namespace IfritShader{
    struct PerFramePerViewData 
    {
        float4x4 m_WorldToView;
        float4x4 m_ViewToClip;
        float4x4 m_WorldToClip;
        float4x4 m_ClipToView;
        float4x4 m_ClipToWorld;
        float4x4 m_ViewToWorld;
        float4 m_CameraPosition;
        float4 m_CameraFront;
        float m_RenderWidth;
        float m_RenderHeight;
        float m_CameraNear;
        float m_CameraFar;
        float m_CameraFovX;
        float m_CameraFovY;
        float m_CameraAspect;
        float m_CameraOrthoSize;
        float m_HizLods;
        float m_ViewCameraType;
        float m_CullCamOrthoSizeX;
        float m_CullCamOrthoSizeY;
    };

    struct PerFramePerViewDataRef
    {
        uint m_Ref;
        uint m_Pad0;
        uint m_Pad1;
        uint m_Pad2;
    };

    struct PerObjectDataRaw
    {
        uint m_TransformRef;
        uint m_ObjectDataRef;
        uint m_InstanceDataRef;
        uint m_TransformRefLast;
        uint m_MaterialId;
    };

    struct FInstanceLocalTransform
    {
        float4x4 m_LocalToWorld;
        float4x4 m_WorldToLocal;
        float4 m_MaxScale;
    };

    IFSHADER_TEMPLATE<typename T>
    T DivRoundUp(T Value, T Divisor)
    IFSHADER_REQUIRES(T : IFSHADER_INTEGER_TYPE)
    {
        return (Value + Divisor - T(1)) / Divisor;
    }

    uint GetBlockId(uint3 BlockId, uint3 WorldSize)
    {
        return BlockId.z * WorldSize.x * WorldSize.y + 
                BlockId.y * WorldSize.x + 
                BlockId.x;
    }

    int GetBlockId(int3 BlockId, int3 WorldSize)
    {
        return BlockId.z * WorldSize.x * WorldSize.y + 
                BlockId.y * WorldSize.x + 
                BlockId.x;
    }

    float SignPreserveZero(float v)
    {
        // https://github.com/shacklettbp/madrona/blob/main/src/render/vk/shaders/utils.hlsl
        int i = asint(v);
        return (i < 0) ? -1.0 : 1.0;
    }

    bool IsFirstLane(uint3 GroupThreadId)
    {
        return (GroupThreadId.x == 0 && GroupThreadId.y == 0 && GroupThreadId.z == 0);
    }

    bool IsGlobalFirstThread(uint3 DispatchThreadId)
    {
        return (DispatchThreadId.x == 0 && DispatchThreadId.y == 0 && DispatchThreadId.z == 0);
    }


}

#else
    // constexprs static
    #define IFSHADER_DEFINE_CONST_UINT32(name, value) inline constexpr uint name = value;
    #define IFSHADER_DEFINE_CONST_INT32(name, value) inline constexpr int name = value;
    #define IFSHADER_DEFINE_CONST_FLOAT(name, value) inline constexpr float name = value;
    #define IFSHADER_DEFINE_CONST_FLOAT2(name, value) inline constexpr Vector2f name = value;
    #define IFSHADER_DEFINE_CONST_FLOAT3(name, value) inline constexpr Vector3f name = value;
    #define IFSHADER_DEFINE_CONST_FLOAT4(name, value) inline constexpr Vector4f name = value;

    #define IFSHADER_UNSCOPED_ENUM 

    using uint = unsigned int;
#endif

IFSHADER_DEFINE_CONST_FLOAT(kPI, 3.14159265358979323846f);
IFSHADER_DEFINE_CONST_FLOAT(kPI2, 6.28318530717958647692f); // 2 * kPI
IFSHADER_DEFINE_CONST_FLOAT(kInvPI, 0.31830988618379067154f); // 1 / kPI