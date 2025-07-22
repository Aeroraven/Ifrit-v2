#pragma once

#include "ifrit.shader.neo/Common.hlsli"

#ifndef __cplusplus
    #include "ifrit.shader.neo/Bindless.hlsli"
    #include "ifrit.shader.neo/Math.LinAlg.SpatialTransform.hlsli"
    #include "ifrit.shader.neo/Math.Quaternion.hlsli"

#endif

namespace IfritShader {
namespace Artemis{
namespace Rigid{

    IFSHADER_DEFINE_CONST_UINT32(kRigidTGSizeX, 128);

#ifndef __cplusplus

    struct FRigidColliderEntryRuntimeData
    {
        float4 m_Displacement;
    };

    struct FRigidColliderEntry
    {
        int m_RuntimeId;
        TRWStructuredBufferHandle<FInstanceLocalTransform> m_Transform;
        float m_ColliderRadius;
    };

#ifdef IFSHADER_RIGID_DYNAMICS_3D

    IFSHADER_TYPEALIAS(FSpatialVector, float3);
    IFSHADER_TYPEALIAS(FSpatialMatrix, float3x3);
    IFSHADER_TYPEALIAS(FAngularValue, float3);
    IFSHADER_TYPEALIAS(FAngularRotation, Math::FQuaternion);
    IFSHADER_TYPEALIAS(FAngularMatrix, float3x3);

#else
    IFSHADER_TYPEALIAS(FSpatialVector, float2);
    IFSHADER_TYPEALIAS(FSpatialMatrix, float2x2);
    IFSHADER_TYPEALIAS(FAngularValue, float);
    IFSHADER_TYPEALIAS(FAngularRotation, float);
    IFSHADER_TYPEALIAS(FAngularMatrix, float);

#endif
    struct FRigidColliderDynamicsData
    {
        FSpatialVector m_Position;
        FSpatialVector m_Displacement;
        FAngularRotation m_Rotation;
        FAngularRotation m_RotationLast;
        FAngularValue m_AngularVelocity;
        FAngularMatrix m_InertiaTensor;
        float m_Mass; //TODO: not make this here!
    };

    FSpatialVector ToSpatialVector(float4 v)
    {
        FSpatialVector result;
        result.x = v.x;
        result.y = v.y;
#ifdef IFSHADER_RIGID_DYNAMICS_3D
        result.z = v.z;
#endif
        return result;
    }

    FAngularValue ToAngularValue(float4 v)
    {
        FAngularValue result;
        result = v.x;
#ifdef IFSHADER_RIGID_DYNAMICS_3D
        result = v.y;
        result = v.z;
#endif
        return result;
    }

#endif

}}}
