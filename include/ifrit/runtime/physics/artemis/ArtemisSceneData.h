#pragma once
#include "ifrit/runtime/common/Pch.h"

namespace Ifrit::Runtime::Artemis
{

    struct ArtemisColliderElement
    {
        Vector4f m_Displacement = Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
        u32      m_TransformRef = 0;
        f32      m_Radius       = 1.0f;
    };

    struct ArtemisSceneData
    {
        u32                         m_NumGpuColliders       = 0;
        RHI::RhiBufferRef           m_GpuColliderDataBuffer = nullptr;
        Vec<ArtemisColliderElement> m_ColliderData;
        Vec<GUID>                   m_ColliderIDs;
    };

} // namespace Ifrit::Runtime::Artemis