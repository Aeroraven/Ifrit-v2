#include "ifrit.shader.neo/Artemis/Rigid/Rigid.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace Rigid{

    struct PushConst_T
    {
        int m_NumColliders;
        TRWStructuredBufferHandle<FRigidColliderEntry> m_ColliderEntries;
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
        FInstanceLocalTransform InstanceTransform = Collider.m_Transform.Load();

        float3 Position = InstanceTransform.m_Position.xyz;
        Position = Position + float3(0.0f, -0.01f, 0.0f);
        InstanceTransform.m_Position.xyz = Position;

        InstanceTransform.m_LocalToWorld = Math::GetModelToWorldMatrix(InstanceTransform.m_Position.xyz,
                                                                 InstanceTransform.m_Rotation.xyz,
                                                                 InstanceTransform.m_MaxScale.xyz);
        InstanceTransform.m_WorldToLocal = Math::Inverse(InstanceTransform.m_LocalToWorld);
        Collider.m_Transform.Store(InstanceTransform);
    }

}}} 