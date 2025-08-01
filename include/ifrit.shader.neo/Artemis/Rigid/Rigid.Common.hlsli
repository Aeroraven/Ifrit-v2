#pragma once

#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Shared/SharedTypes.h"
#include "ifrit.shader.neo/Shared/Artemis/Rigid.Shared.h"

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

    // NOT USED Now
    struct FRigidColliderEntryRuntimeData
    {
        float4 m_Displacement;
    };

    

#ifdef IFSHADER_RIGID_DYNAMICS_3D

    IFSHADER_TYPEALIAS(FSpatialVector, float3);
    IFSHADER_TYPEALIAS(FSpatialMatrix, float3x3);
    IFSHADER_TYPEALIAS(FAngularValue, float3);
    IFSHADER_TYPEALIAS(FAngularRotation, Math::FQuaternion);
    IFSHADER_TYPEALIAS(FAngularMatrix, float3x3);

    IFSHADER_DEFINE_CONST_INT32(kSizeofColliderDynamicsDataInF32, 3+3+3+4+4+3+9);
    IFSHADER_DEFINE_CONST_INT32(kRotationSectionOffset, 4); //TODO: LAYOUT!!!!!
#else
    IFSHADER_TYPEALIAS(FSpatialVector, float2);
    IFSHADER_TYPEALIAS(FSpatialMatrix, float2x2);
    IFSHADER_TYPEALIAS(FAngularValue, float);
    IFSHADER_TYPEALIAS(FAngularRotation, float);
    IFSHADER_TYPEALIAS(FAngularMatrix, float);

    IFSHADER_DEFINE_CONST_INT32(kSizeofColliderDynamicsDataInF32, 2+2+2+6);
    IFSHADER_DEFINE_CONST_INT32(kRotationSectionOffset, 6);

#endif
    struct FRigidColliderDynamicsData
    {
        FSpatialVector m_Displacement;
        FSpatialVector m_DisplacementOld;
        FSpatialVector m_Position;
        FAngularRotation m_Rotation;
        FAngularRotation m_RotationLast;
        FAngularValue m_AngularVelocity;
        FAngularValue m_AngularVelocityOld;
        FAngularMatrix m_InertiaTensor;
        int m_Pad;
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
