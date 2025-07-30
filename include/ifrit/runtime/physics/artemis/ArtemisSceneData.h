#pragma once
#include "ifrit/runtime/common/Pch.h"
#include "ifrit.shader.neo/Shared/Artemis/Rigid.Shared.h"

namespace Ifrit::Runtime::Artemis
{

    struct ArtemisColliderElementRuntimeData
    {
        u32 m_DataSection[36] = { 0 };
    };

    struct ArtemisSceneData
    {
        u32                                       m_FrameId                      = 0;
        u32                                       m_NumGpuColliders              = 0;
        u32                                       m_AllocatedRuntimeIds          = 0;
        RHI::RhiBufferRef                         m_GpuColliderDataBuffer[2]     = { nullptr, nullptr };
        RHI::RhiBufferRef                         m_GpuColliderDataBufferRuntime = nullptr;
        Vec<Shader::Artemis::FRigidColliderEntry> m_ColliderData;
        Vec<ArtemisColliderElementRuntimeData>    m_ColliderDataRuntime;
        Vec<GUID>                                 m_ColliderIDs;
    };

} // namespace Ifrit::Runtime::Artemis
