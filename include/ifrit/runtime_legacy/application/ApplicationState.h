#pragma once
#include "ifrit/runtime/common/Pch.h"

namespace Ifrit::Runtime
{
    struct ApplicationState
    {
        bool m_EditorMode              = false;
        bool m_EnableRenderingPipeline = true;

        // Profiler states
        bool mProfilerRequestFrameCapture = false;

        // Editor states
        f32  mEditorViewportX      = 0.0f;
        f32  mEditorViewportY      = 0.0f;
        f32  mEditorViewportWidth  = 0.0f;
        f32  mEditorViewportHeight = 0.0f;
    };

} // namespace Ifrit::Runtime
