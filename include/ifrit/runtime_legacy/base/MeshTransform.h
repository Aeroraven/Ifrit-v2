#pragma once
#include "ifrit/runtime/common/Pch.h"

namespace Ifrit::Runtime
{

    struct MeshInstanceTransform
    {
        Matrix4x4f model;
        Matrix4x4f invModel;
        Vector4f   maxScale;
        Vector4f   m_Position;
        Vector4f   m_Rotation;
    };

} // namespace Ifrit::Runtime