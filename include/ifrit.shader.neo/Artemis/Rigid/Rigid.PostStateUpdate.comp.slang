#pragma ifrit.multi_compile IFSHADER_RIGID_DYNAMICS_3D
//#define IFSHADER_RIGID_DYNAMICS_3D
#include "ifrit.shader.neo/Artemis/Rigid/Rigid.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace Rigid{

    struct PushConst_T
    {
        float4 m_Gravity;
        float4 m_ExtMoment; // tau_ext in original xpbd-rigid-body paper
        uint m_NumRigidBodies;
        float m_DeltaTime;
        TStructuredBufferHandle<int> m_IndirectCounter;
        TStructuredBufferHandle<FRigidColliderEntry> m_ColliderEntries;
        TRWStructuredBufferHandle<FRigidColliderDynamicsData> m_ColliderDynamics;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kRigidTGSizeX, 1, 1)
    void RigidPostStateUpdateCS(uint3 ThreadId:SV_DispatchThreadID,
                                uint3 GroupId:SV_GroupID,
                                uint3 GroupThreadId:SV_GroupThreadID)
    {
        uint NumRigidBodies = PushConst.m_NumRigidBodies;
        if(NumRigidBodies == ~0u)
        {
            NumRigidBodies = PushConst.m_IndirectCounter.Load(0);
        }
        if(ThreadId.x >= NumRigidBodies)
        {
            return;
        }

        FRigidColliderEntry ColliderEntry = PushConst.m_ColliderEntries.Load(ThreadId.x);
        FRigidColliderDynamicsData DynamicsData = PushConst.m_ColliderDynamics.Load(ColliderEntry.m_RuntimeId);
        
        DynamicsData.m_Position += DynamicsData.m_Displacement;
        
#ifndef IFSHADER_RIGID_DYNAMICS_3D
        DynamicsData.m_AngularVelocity = (DynamicsData.m_Rotation - DynamicsData.m_RotationLast) / PushConst.m_DeltaTime;
#else
        //TODO: Add 3D rotation handling
#endif
        // v<-v+h*fext/m;
        // dx<-dx+h*h*fext/(2*m);
        FSpatialVector Gravity = ToSpatialVector(PushConst.m_Gravity);
        FSpatialVector Accel = Gravity;
        FSpatialVector Displacement = DynamicsData.m_Displacement;
        Displacement += Accel * PushConst.m_DeltaTime * PushConst.m_DeltaTime * 0.5f;   

        // q_prev<-q
        //DynamicsData.m_RotationLast = DynamicsData.m_Rotation;

        // w<-w+h*inv(I)@(tau_ext-w.cross(I@w))
        FAngularValue TauExt = ToAngularValue(PushConst.m_ExtMoment);
        FAngularMatrix Inertia =  DynamicsData.m_InertiaTensor;
        FAngularValue AngularVelocity = DynamicsData.m_AngularVelocity;
        TauExt = TauExt - Math::Cross(AngularVelocity, Math::Mul(Inertia, AngularVelocity));

        FAngularMatrix InertiaInv = Math::Inverse(Inertia);
        FAngularValue DeltaW = Math::Mul(InertiaInv, TauExt) * PushConst.m_DeltaTime;
        AngularVelocity += DeltaW;

        // (3D) q<-q+h*0.5*QuatMul(<w.xyz,0>,q); && q<-norm(q)
        // (2D) q<-q+h*w
        FAngularRotation NewQ;
#ifdef IFSHADER_RIGID_DYNAMICS_3D
        Math::FQuaternion Wq = Math::ToQuaternion(float4(AngularVelocity.xyz, 0.0f));
        Math::FQuaternion NewQTemp = Math::Mul(Wq, DynamicsData.m_Rotation);
        NewQ = Math::Add(DynamicsData.m_Rotation,Math::Mul(PushConst.m_DeltaTime * 0.5f, NewQTemp));
        NewQ = Math::Normalize(NewQ);
#else   
        NewQ = DynamicsData.m_Rotation + AngularVelocity * PushConst.m_DeltaTime;
#endif
    
        //DynamicsData.m_Rotation = NewQ;
        //DynamicsData.m_AngularVelocity = AngularVelocity;
        DynamicsData.m_Displacement = Displacement;

        if(abs(DynamicsData.m_Position.x) < 0.01f && abs(DynamicsData.m_Position.y) < 0.01f)
        {
            DynamicsData.m_Position.xy = float2(1.0f, 1.0f);
            while(true){
                DynamicsData.m_Position+=1.0f;
            }
        }


        
        PushConst.m_ColliderDynamics.Store(DynamicsData, ColliderEntry.m_RuntimeId);
    }

}}}
