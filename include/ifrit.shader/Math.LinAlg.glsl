
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

mat4 ifrit_CubeSpaceRemap(vec3 srcMin,vec3 srcMax,vec3 dstMin,vec3 dstMax){
    float scaleX = (dstMax.x - dstMin.x) / (srcMax.x - srcMin.x);
    float scaleY = (dstMax.y - dstMin.y) / (srcMax.y - srcMin.y);
    float scaleZ = (dstMax.z - dstMin.z) / (srcMax.z - srcMin.z);
    float offsetX = dstMin.x - srcMin.x * scaleX;
    float offsetY = dstMin.y - srcMin.y * scaleY;
    float offsetZ = dstMin.z - srcMin.z * scaleZ;

    vec4 col1 = vec4(scaleX, 0.0, 0.0, 0.0);
    vec4 col2 = vec4(0.0, scaleY, 0.0, 0.0);
    vec4 col3 = vec4(0.0, 0.0, scaleZ, 0.0);
    vec4 col4 = vec4(offsetX, offsetY, offsetZ, 1.0);
    mat4 ret = mat4(col1, col2, col3, col4);
    return ret;
}