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
#include "ifrit.shader.neo/Artemis/Contact/SphereCubeContact.hlsli"
#include "ifrit.shader.neo/Artemis/Contact/CubeContact.hlsli"

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
        Rigid::FSpatialVector m_ContactPointRigidWrtCenter1; // Contact point in local space of rigid 1
        Rigid::FSpatialVector m_ContactPointRigidWrtCenter2; // Contact point in local space of rigid 2
        Rigid::FSpatialVector m_ContactPointRigidWrtCenterSec1; // Contact point in local space of rigid 1
        Rigid::FSpatialVector m_ContactPointRigidWrtCenterSec2; // Contact point in local space of rigid 2
        bool m_Collided;
        bool m_HasSecondContact; 
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

    Rigid::FSpatialVector ConvertRigidPointWrtCenterDirToWS(
        Rigid::FSpatialVector RigidDirWS,
        Rigid::FRigidColliderDynamicsData RigidDynamics
    )
    {
        Rigid::FSpatialMatrix RotationMat;
#ifdef IFSHADER_RIGID_DYNAMICS_3D
        RotationMat = Math::QuaternionToRotationMatrix(RigidDynamics.m_Rotation);
#else
        RotationMat = Math::GetRotationMatrix(RigidDynamics.m_Rotation);
#endif
        Rigid::FSpatialVector WorldSpacePoint = mul(RotationMat, RigidDirWS) ;
        return WorldSpacePoint;
    }

    Rigid::FSpatialVector ConvertRigidPointWrtCenterDirToLS(
        Rigid::FSpatialVector RigidDirWS,
        Rigid::FRigidColliderDynamicsData RigidDynamics
    )
    {
        Rigid::FSpatialMatrix RotationMat;
#ifdef IFSHADER_RIGID_DYNAMICS_3D
        RotationMat = Math::Inverse(Math::QuaternionToRotationMatrix(RigidDynamics.m_Rotation));
#else
        RotationMat = Math::Inverse(Math::GetRotationMatrix(RigidDynamics.m_Rotation));
#endif
        Rigid::FSpatialVector WorldSpacePoint = mul(RotationMat, RigidDirWS);
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
        Result.m_HasSecondContact = false;
        Result.m_ContactPointRigidWrtCenterSec1 = Rigid::FSpatialVector(0.0f);
        Result.m_ContactPointRigidWrtCenterSec2 = Rigid::FSpatialVector(0.0f);

        if(Rigid::ERigidColliderType::Sphere == ColliderType1 && 
           Rigid::ERigidColliderType::Sphere == ColliderType2)
        {
            float Radius1 = RigidCollider1.m_ColliderRadius;
            float Radius2 = RigidCollider2.m_ColliderRadius;
#ifdef IFSHADER_RIGID_DYNAMICS_3D
            Result.m_Collided = false;
            Result.m_ContactNormal = Rigid::FSpatialVector(0.0f);
            Result.m_ContactPointRigidWrtCenter1 = Rigid::FSpatialVector(0.0f);
            Result.m_ContactPointRigidWrtCenter2 = Rigid::FSpatialVector(0.0f);  
#else
            FCircleContactResult CollResult = CircleToCircleContact2D(
                RigidCenter1.xy, Radius1, RigidCenter2.xy, Radius2
            );
            Result.m_Collided = CollResult.Collided;
            Result.m_ContactNormal = CollResult.Normal;
            Result.m_ContactPointRigidWrtCenter1 = CollResult.ContactPoint1 - RigidCenter1.xy;
            Result.m_ContactPointRigidWrtCenter2 = CollResult.ContactPoint2 - RigidCenter2.xy;

            // Convert contact points to local space of rigid
            Result.m_ContactPointRigidWrtCenter1 = ConvertRigidPointWrtCenterDirToLS(
                Result.m_ContactPointRigidWrtCenter1, RigidDynamics1
            );
            Result.m_ContactPointRigidWrtCenter2 = ConvertRigidPointWrtCenterDirToLS(
                Result.m_ContactPointRigidWrtCenter2, RigidDynamics2
            );
#endif
            return Result;
        }
        else if(Rigid::ERigidColliderType::Sphere == ColliderType1 && 
                Rigid::ERigidColliderType::Box == ColliderType2)
        {

#ifdef IFSHADER_RIGID_DYNAMICS_3D
            //TODO: Handle 3D sphere-box collision
#else
            float Radius1 = RigidCollider1.m_ColliderRadius;
            float Width2 = RigidCollider2.m_ColliderCuboidSize.x;
            float Height2 = RigidCollider2.m_ColliderCuboidSize.y;
            float Rot2 = RigidDynamics2.m_Rotation;
            FRectVertex Rect2;
            Rect2.Center = RigidCenter2.xy;
            Rect2.HalfSize = float2(Width2 * 0.5f, Height2 * 0.5f);
            FCircleRectContactResult CollResult = CircleRectContact2D(RigidCenter1.xy, Radius1, Rect2, Rot2);
            Result.m_Collided = CollResult.Collided;
            Result.m_ContactNormal = CollResult.Normal;
            Result.m_ContactPointRigidWrtCenter1 = CollResult.ContactPoint1 - RigidCenter1.xy;
            Result.m_ContactPointRigidWrtCenter2 = CollResult.ContactPoint2 - RigidCenter2.xy;
            // Convert contact points to local space of rigid
            Result.m_ContactPointRigidWrtCenter1 = ConvertRigidPointWrtCenterDirToLS(
                Result.m_ContactPointRigidWrtCenter1, RigidDynamics1
            );
            Result.m_ContactPointRigidWrtCenter2 = ConvertRigidPointWrtCenterDirToLS(
                Result.m_ContactPointRigidWrtCenter2, RigidDynamics2
            );
            return Result;
#endif
        } else if(Rigid::ERigidColliderType::Box == ColliderType1 && 
                Rigid::ERigidColliderType::Sphere == ColliderType2)
        {
#ifdef IFSHADER_RIGID_DYNAMICS_3D
            //TODO: Handle 3D sphere-box collision
#else
            float Radius2 = RigidCollider2.m_ColliderRadius;
            float Width1 = RigidCollider1.m_ColliderCuboidSize.x;
            float Height1 = RigidCollider1.m_ColliderCuboidSize.y;
            float Rot1 = RigidDynamics1.m_Rotation;
            FRectVertex Rect1;
            Rect1.Center = RigidCenter1.xy;
            Rect1.HalfSize = float2(Width1 * 0.5f, Height1 * 0.5f);
            FCircleRectContactResult CollResult = RectCircleContact2D(Rect1, Rot1, RigidCenter2.xy, Radius2);
            Result.m_Collided = CollResult.Collided;
            Result.m_ContactNormal = CollResult.Normal;
            Result.m_ContactPointRigidWrtCenter1 = CollResult.ContactPoint1 - RigidCenter1.xy;
            Result.m_ContactPointRigidWrtCenter2 = CollResult.ContactPoint2 - RigidCenter2.xy;
            // Convert contact points to local space of rigid
            Result.m_ContactPointRigidWrtCenter1 = ConvertRigidPointWrtCenterDirToLS(
                Result.m_ContactPointRigidWrtCenter1, RigidDynamics1
            );
            Result.m_ContactPointRigidWrtCenter2 = ConvertRigidPointWrtCenterDirToLS(
                Result.m_ContactPointRigidWrtCenter2, RigidDynamics2
            );
            return Result;
#endif
        }else if(Rigid::ERigidColliderType::Box == ColliderType1 && 
            Rigid::ERigidColliderType::Box == ColliderType2)
        {
#ifdef IFSHADER_RIGID_DYNAMICS_3D   
            //TODO: Handle 3D box-box collision
#else
            FRectVertex Rect1;
            Rect1.Center = RigidCenter1.xy;
            Rect1.HalfSize = RigidCollider1.m_ColliderCuboidSize.xy * 0.5f;
            FRectVertex Rect2;
            Rect2.Center = RigidCenter2.xy;
            Rect2.HalfSize = RigidCollider2.m_ColliderCuboidSize.xy * 0.5f;
            float Rot1 = RigidDynamics1.m_Rotation;
            float Rot2 = RigidDynamics2.m_Rotation;

            FRectContactResult CollResult = RectToRectContact2D(Rect1, Rot1, Rect2, Rot2);
            Result.m_Collided = CollResult.Collided;
            Result.m_ContactNormal = CollResult.Normal;
            Result.m_ContactPointRigidWrtCenter1 = CollResult.ContactPointA1 - RigidCenter1.xy;
            Result.m_ContactPointRigidWrtCenter2 = CollResult.ContactPointB1 - RigidCenter2.xy;

            // Convert contact points to local space of rigid
            Result.m_ContactPointRigidWrtCenter1 = ConvertRigidPointWrtCenterDirToLS(
                Result.m_ContactPointRigidWrtCenter1, RigidDynamics1
            );
            Result.m_ContactPointRigidWrtCenter2 = ConvertRigidPointWrtCenterDirToLS(
                Result.m_ContactPointRigidWrtCenter2, RigidDynamics2
            );
            if(Result.m_HasSecondContact)
            {
                Result.m_HasSecondContact = CollResult.HasSecondContact;
                Result.m_ContactPointRigidWrtCenterSec1 = CollResult.ContactPointA2 - RigidCenter1.xy;
                Result.m_ContactPointRigidWrtCenterSec2 = CollResult.ContactPointB2 - RigidCenter2.xy;
                // Convert second contact points to local space of rigid
                Result.m_ContactPointRigidWrtCenterSec1 = ConvertRigidPointWrtCenterDirToLS(
                    Result.m_ContactPointRigidWrtCenterSec1, RigidDynamics1
                );
                Result.m_ContactPointRigidWrtCenterSec2 = ConvertRigidPointWrtCenterDirToLS(
                    Result.m_ContactPointRigidWrtCenterSec2, RigidDynamics2
                );
            }
            return Result;
#endif
        }


        Result.m_Collided = false;
        Result.m_ContactNormal = Rigid::FSpatialVector(0.0f);
        Result.m_ContactPointRigidWrtCenter1 = Rigid::FSpatialVector(0.0f);
        Result.m_ContactPointRigidWrtCenter2 = Rigid::FSpatialVector(0.0f);  
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
        Result.m_ContactNormal = -mul(RotationMat, Result.m_ContactNormal);
        return Result;
    }

}}}
