#pragma ifrit.multi_compile IFSHADER_RIGID_DYNAMICS_3D
//#define IFSHADER_RIGID_DYNAMICS_3D
#include "ifrit.shader.neo/Artemis/Rigid/Rigid.Common.hlsli"
#include "ifrit.shader.neo/Math.LinAlg.SpatialTransform.hlsli"

namespace IfritShader{
namespace Artemis{
namespace Rigid{

    struct PushConst_T
    {
        uint m_NumRigidBodies;
        TStructuredBufferHandle<int> m_IndirectCounter;
        TStructuredBufferHandle<FRigidColliderEntry> m_ColliderEntries;
        TRWStructuredBufferHandle<FRigidColliderDynamicsData> m_ColliderDynamics;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kRigidTGSizeX, 1, 1)
    void RigidLoadTransformCS(uint3 ThreadId:SV_DispatchThreadID,
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
        FInstanceLocalTransform InstanceTransform = ColliderEntry.m_Transform.Load();


#ifdef IFSHADER_RIGID_DYNAMICS_3D
        DynamicsData.m_Position = InstanceTransform.m_Position.xyz;
        DynamicsData.m_Rotation = Math::IdentityQ();
        DynamicsData.m_RotationLast = Math::IdentityQ();
        DynamicsData.m_AngularVelocity = FSpatialVector(0.0f, 0.0f, 0.0f);
        DynamicsData.m_InertiaTensor = 1.0f;
#else
        DynamicsData.m_Position = InstanceTransform.m_Position.xy;
        DynamicsData.m_Rotation = InstanceTransform.m_Rotation.z;
        DynamicsData.m_InertiaTensor = ColliderEntry.m_Inertia2D;
#endif

        PushConst.m_ColliderDynamics.Store(DynamicsData, ColliderEntry.m_RuntimeId);
    }

}}}
