#pragma once

//#define IFSHADER_MPM_3D

#ifdef IFSHADER_MPM_3D
    #ifndef IFSHADER_RIGID_DYNAMICS_3D
        #define IFSHADER_RIGID_DYNAMICS_3D
    #endif
#else
    #ifdef IFSHADER_RIGID_DYNAMICS_3D
        #error "Rigid Dynamics 3D is enabled but MPM is not 3D!"
    #endif
#endif

#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"
#include "ifrit.shader.neo/Shared/Artemis/Rigid.Shared.h"
#include "ifrit.shader.neo/Shared/Artemis/MPMRigidCoupling.Shared.h"
#include "ifrit.shader.neo/Math.Geometry.Intersection.hlsli"
#include "ifrit.shader.neo/Artemis/MPM/RigidCoupling/MPMRigid.Common.hlsli"

#include "ifrit.shader.neo/Math.Quaternion.hlsli"

namespace IfritShader {
namespace Artemis {
namespace MPM { 


    struct FParticleRigidCollisionCheckResult
    {
        Rigid::FSpatialVector m_CollisionLocRigid; // Position @ initial state!
        Rigid::FSpatialVector m_ContactNormal; // Normal @ world space!, surface normal of rigid
        bool m_Collided;
    };

    Rigid::FSpatialVector ConvertRigidPointToWorldSpace(
        Rigid::FSpatialVector RigidPointMS,
        Rigid::FRigidColliderDynamicsData RigidDynamics
    )
    {
        Rigid::FSpatialMatrix RotationMat;
#ifdef IFSHADER_RIGID_DYNAMICS_3D
        RotationMat = Math::QuaternionToRotationMatrix(RigidDynamics.m_Rotation);
#else
        RotationMat = Math::GetRotationMatrix(RigidDynamics.m_Rotation);
#endif
        Rigid::FSpatialVector RigidCenter = RigidDynamics.m_Position + RigidDynamics.m_Displacement;
        Rigid::FSpatialVector WorldSpacePoint = mul(RotationMat, RigidPointMS) + RigidCenter;
        return WorldSpacePoint;
    }

    FParticleRigidCollisionCheckResult DetectParticleRigidCollision(
        Rigid::FSpatialVector ParticlePosition,
        Rigid::FRigidColliderEntry RigidCollider,
        Rigid::FRigidColliderDynamicsData RigidDynamics
    )
    {
        Rigid::FAngularRotation Rotation = RigidDynamics.m_Rotation;
        Rigid::FSpatialMatrix RotationMat;
        Rigid::FSpatialVector RigidCenterPrev = RigidDynamics.m_Position;
        Rigid::FSpatialVector RigidCenterDisplacement = RigidDynamics.m_Displacement;
        Rigid::FSpatialVector RigidCenter = RigidCenterPrev + RigidCenterDisplacement;

#ifdef IFSHADER_RIGID_DYNAMICS_3D
        RotationMat = Math::QuaternionToRotationMatrix(Rotation);
#else
        RotationMat = Math::GetRotationMatrix(Rotation);
#endif
        FParticleRigidCollisionCheckResult Result;
        bool bCollided;
        if(RigidCollider.m_ColliderType == Rigid::ERigidColliderType::Sphere)
        {
            
            float Radius = RigidCollider.m_ColliderRadius;
#ifdef IFSHADER_RIGID_DYNAMICS_3D
            bCollided = Math::SpherePointCollisionInModelSpace3D(ParticlePosition, RigidCenter, 
                Radius, RotationMat, Result.m_CollisionLocRigid, Result.m_ContactNormal);
#else
            bCollided = Math::CirclePointCollisionInModelSpace2D(ParticlePosition,RigidCenter, 
                Radius, RotationMat, Result.m_CollisionLocRigid, Result.m_ContactNormal);
#endif
        }else if(RigidCollider.m_ColliderType == Rigid::ERigidColliderType::Box)
        {
            Rigid::FSpatialVector HalfBoxSize = ToSpatialVector(RigidCollider.m_ColliderCuboidSize) * 0.5f;
            Rigid::FSpatialVector Min = RigidCenter - HalfBoxSize;
            Rigid::FSpatialVector Max = RigidCenter + HalfBoxSize;

#ifdef IFSHADER_RIGID_DYNAMICS_3D
            bCollided = Math::CuboidPointCollisionInModelSpace3D(ParticlePosition,Min, Max, RotationMat,
                Result.m_CollisionLocRigid, Result.m_ContactNormal);
#else
            bCollided = Math::RectanglePointCollisionInModelSpace2D(ParticlePosition, Min, Max, RotationMat,
                Result.m_CollisionLocRigid, Result.m_ContactNormal);
#endif
        }
        Result.m_Collided = bCollided;
        Result.m_ContactNormal = mul(RotationMat, Result.m_ContactNormal);
        return Result;
    }

}}}