
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "Ayanami/Ayanami.Shared.glsl"
#include "Ayanami/Ayanami.SharedConst.h"

layout(location = 0) in vec3 meshColor;

layout(push_constant) uniform UPushConstant{
    vec4 m_WorldBoundMin;
    vec4 m_WorldBoundMax;
    uint m_PerFrame;
    uint m_VoxelsPerWidth;
    uint m_ObjectGridId;
} PushConst;

layout(location = 0) out vec4 outColor;

//TODO: MSAA for better quality
void main(){
    outColor = vec4(meshColor, 1.0);
}
