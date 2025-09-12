#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/math/VectorDefs.h"

namespace Ifrit::Runtime::Artemis
{
    struct RigidBaseConfig
    {
        Vector3f m_Gravity = Vector3f(0.0f, 0.0f, 0.0f);
    };

} // namespace Ifrit::Runtime::Artemis