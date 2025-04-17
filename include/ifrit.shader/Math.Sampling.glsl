
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


vec4 ifrit_SampleCosineHemisphereWithPDF(vec2 uv){
    float phi = uv.x * 2.0 * kPI;
    float cosTheta = sqrt(uv.y);
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    vec4 ret;
    ret.x = cos(phi) * sinTheta;
    ret.y = sin(phi) * sinTheta;
    ret.z = cosTheta;
    ret.w = cosTheta / kPI; // PDF
    return ret;
}

vec4 ifrit_SampleUniformSphereWithPDF(vec2 uv){
    float phi = uv.x * 2.0 * kPI;
    float cosTheta = 1.0 - 2.0 * uv.y;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    vec4 ret;
    ret.x = cos(phi) * sinTheta;
    ret.y = sin(phi) * sinTheta;
    ret.z = cosTheta;
    ret.w = 1.0 / (4.0 * kPI); // PDF
    return ret;
}

