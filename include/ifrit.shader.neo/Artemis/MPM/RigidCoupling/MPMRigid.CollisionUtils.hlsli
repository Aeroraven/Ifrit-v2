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
#include "ifrit.shader.neo/Artemis/Contact/SphereContact.hlsli"

namespace IfritShader {
namespace Artemis {
namespace MPM { 


    struct FParticleRigidCollisionCheckResult
    {
        Rigid::FSpatialVector m_CollisionLocRigid; // Position @ initial state!
        Rigid::FSpatialVector m_ContactNormal; // Normal @ world space!, surface normal of rigid
        bool m_Collided;
    };

    struct FRigidRigidCollisionCheckResult
    {
        Rigid::FSpatialVector m_ContactNormal; // Normal @ world space!, surface normal of rigid
        Rigid::FSpatialVector m_ContactPointRigid1; // Contact point in local space of rigid 1
        Rigid::FSpatialVector m_ContactPointRigid2; // Contact point in local space of rigid 2
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

    FRigidRigidCollisionCheckResult DetectRigidRigidCollision(
        Rigid::FRigidColliderEntry RigidCollider1,
        Rigid::FRigidColliderDynamicsData RigidDynamics1,
        Rigid::FRigidColliderEntry RigidCollider2,
        Rigid::FRigidColliderDynamicsData RigidDynamics2
    )
    {
        Rigid::FSpatialVector RigidCenterPrev1 = RigidDynamics1.m_Position;
        Rigid::FSpatialVector RigidCenterDisplacement1 = RigidDynamics1.m_Displacement;
        Rigid::FSpatialVector RigidCenter1 = RigidCenterPrev1 + RigidCenterDisplacement1;
        Rigid::FSpatialVector RigidCenterPrev2 = RigidDynamics2.m_Position;
        Rigid::FSpatialVector RigidCenterDisplacement2 = RigidDynamics2.m_Displacement;
        Rigid::FSpatialVector RigidCenter2 = RigidCenterPrev2 + RigidCenterDisplacement2;

        Rigid::ERigidColliderType ColliderType1 = RigidCollider1.m_ColliderType;
        Rigid::ERigidColliderType ColliderType2 = RigidCollider2.m_ColliderType;

        FRigidRigidCollisionCheckResult Result;

        if(Rigid::ERigidColliderType::Sphere == ColliderType1 && 
           Rigid::ERigidColliderType::Sphere == ColliderType2)
        {
            float Radius1 = RigidCollider1.m_ColliderRadius;
            float Radius2 = RigidCollider2.m_ColliderRadius;
#ifdef IFSHADER_RIGID_DYNAMICS_3D
            Result.m_Collided = false;
            Result.m_ContactNormal = Rigid::FSpatialVector(0.0f);
            Result.m_ContactPointRigid1 = Rigid::FSpatialVector(0.0f);
            Result.m_ContactPointRigid2 = Rigid::FSpatialVector(0.0f);  
#else
            FCircleContactResult CollResult = CircleToCircleContact2D(
                RigidCenter1.xy, Radius1, RigidCenter2.xy, Radius2
            );
            Result.m_Collided = CollResult.Collided;
            Result.m_ContactNormal = CollResult.Normal;
            Result.m_ContactPointRigid1 = CollResult.ContactPoint1;
            Result.m_ContactPointRigid2 = CollResult.ContactPoint2;
#endif
            return Result;
        }

        Result.m_Collided = false;
        Result.m_ContactNormal = Rigid::FSpatialVector(0.0f);
        Result.m_ContactPointRigid1 = Rigid::FSpatialVector(0.0f);
        Result.m_ContactPointRigid2 = Rigid::FSpatialVector(0.0f);  
        return Result;
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
