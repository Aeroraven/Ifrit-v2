#include "ifrit.shader.neo/Artemis/Rigid/Rigid.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace Rigid{

    struct PushConst_T
    {
        float4 m_Gravity;
        float m_TimeStep;
        int m_NumColliders;
        TStructuredBufferHandle<FRigidColliderEntry> m_ColliderEntries;
        TRWStructuredBufferHandle<FRigidColliderEntryRuntimeData> m_ColliderEntriesRuntime;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kRigidTGSizeX, 1, 1)
    void RigidMotionTestCS(uint3 ThreadId:SV_DispatchThreadID,
                           uint3 GroupId:SV_GroupID,
                           uint3 GroupThreadId:SV_GroupThreadID)
    {
        if(ThreadId.x >= PushConst.m_NumColliders)
            return;

        FRigidColliderEntry Collider = PushConst.m_ColliderEntries.Load(ThreadId.x);
        FRigidColliderEntryRuntimeData ColliderRuntime = PushConst.m_ColliderEntriesRuntime.Load(Collider.m_RuntimeId);
        FInstanceLocalTransform InstanceTransform = Collider.m_Transform.Load();

        float3 Displacement = ColliderRuntime.m_Displacement.xyz + PushConst.m_Gravity.xyz * PushConst.m_TimeStep * PushConst.m_TimeStep;

        float3 Position = InstanceTransform.m_Position.xyz;
        Position = Position + Displacement;
        InstanceTransform.m_Position.xyz = Position;

        InstanceTransform.m_LocalToWorld = Math::GetModelToWorldMatrix(InstanceTransform.m_Position.xyz,
                                                                 InstanceTransform.m_Rotation.xyz,
                                                                 InstanceTransform.m_MaxScale.xyz);
        InstanceTransform.m_WorldToLocal = Math::Inverse(InstanceTransform.m_LocalToWorld);
        Collider.m_Transform.Store(InstanceTransform);
        ColliderRuntime.m_Displacement.xyz = Displacement;

        PushConst.m_ColliderEntriesRuntime.Store(ColliderRuntime, ThreadId.x);
    }

}}} 
