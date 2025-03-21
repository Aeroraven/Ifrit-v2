
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

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

// Simple Temporary Anti-Aliasing
// Implementation might be incorrect, but it works for now.
// 
// Reference: https://zhuanlan.zhihu.com/p/425233743

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

#include "Bindless.glsl"

void main(){
    outColor = vec4(texCoord, 0.0, 1.0);
}