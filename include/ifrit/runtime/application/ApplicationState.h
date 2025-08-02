#pragma once
#include "ifrit/runtime/common/Pch.h"

namespace Ifrit::Runtime
{
    struct ApplicationState
    {
        bool m_EditorMode              = false;
        bool m_EnableRenderingPipeline = true;
    };

} // namespace Ifrit::Runtime
