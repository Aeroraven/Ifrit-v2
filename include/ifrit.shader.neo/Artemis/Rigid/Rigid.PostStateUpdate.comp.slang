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
        if(length(DynamicsData.m_Displacement)<1e-6f)
        {
            //DynamicsData.m_Displacement = FSpatialVector(0.0f);
        }
        DynamicsData.m_Position += DynamicsData.m_Displacement;
        
#ifndef IFSHADER_RIGID_DYNAMICS_3D
        DynamicsData.m_AngularVelocity = (DynamicsData.m_Rotation - DynamicsData.m_RotationLast) / PushConst.m_DeltaTime;
#else
        //TODO: Add 3D rotation handling
#endif
        PushConst.m_ColliderDynamics.Store(DynamicsData, ColliderEntry.m_RuntimeId);
    }

}}}
