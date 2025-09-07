#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/forwarding/FwdComponent.h"
#include "ifrit/rhi/common/RhiForwardingTypes.h"
namespace Ifrit::Runtime
{
    struct TransformAllocData
    {
        bool m_Changed;
        u32  m_TransformRef;
        u32  m_TransformRefLast;
    };

    IFRIT_RUNTIME_API TransformAllocData UpdateTransformGPUData(Transform* transform, RHI::RhiBackend* rhi);

} // namespace Ifrit::Runtime