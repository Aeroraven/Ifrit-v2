#pragma once
#include "ifrit/runtime/common/Pch.h"

namespace Ifrit::Runtime::Artemis
{

    struct ArtemisColliderElement
    {
        u32 m_RuntimeId    = 0;
        u32 m_TransformRef = 0;
        f32 m_Radius       = 1.0f;
    };

    struct ArtemisColliderElementRuntimeData
    {
        Vector4f m_Displacement = Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
    };

    struct ArtemisSceneData
    {
        u32                                    m_NumGpuColliders              = 0;
        u32                                    m_AllocatedRuntimeIds          = 0;
        RHI::RhiBufferRef                      m_GpuColliderDataBuffer        = nullptr;
        RHI::RhiBufferRef                      m_GpuColliderDataBufferRuntime = nullptr;
        Vec<ArtemisColliderElement>            m_ColliderData;
        Vec<ArtemisColliderElementRuntimeData> m_ColliderDataRuntime;
        Vec<GUID>                              m_ColliderIDs;
    };

} // namespace Ifrit::Runtime::Artemis
