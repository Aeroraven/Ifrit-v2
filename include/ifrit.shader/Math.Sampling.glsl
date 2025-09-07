
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


vec4 ifrit_SampleCosineHemisphereWithPDF(vec2 uv, vec3 Normal){
    vec3 SampleH = ifrit_SampleUniformSphereWithPDF(uv).xyz;
    vec3 SampleH2 = normalize(Normal+SampleH);
    float PDF = dot(SampleH2, Normal) / kPI;
    vec4 ret;
    ret.xyz = SampleH2;
    ret.w = PDF;
    return ret;
}



// Ray Tracing Gems 16.5.4.2
// Better description in Clarberg's
// "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD"


float ifrit_SignPreserveZero(float v)
{
    return (v<0.0) ? -1.0:1.0;
}

vec3 ifrit_ConcentricOctahedralTransform(vec2 u)
{
    // https://zhuanlan.zhihu.com/p/408898601
    // https://fileadmin.cs.lth.se/graphics/research/papers/2008/simdmapping/clarberg_simdmapping08_preprint.pdf
    // Port from ifrit.core.math

    // This implementation is based on shacklettbp/madrona
    // https://github.com/shacklettbp/madrona/blob/main/src/render/vk/shaders/utils.hlsl

    const float PI = 3.14159265358979323846;
    u = u * 2.0 - 1.0;

    // Compute radius r (branchless)
    float d = 1.0 - (abs(u.x) + abs(u.y));
    float r = 1.0 - abs(d);

    // Compute phi in the first quadrant (branchless, except for the
    // division-by-zero test), using sign(u) to map the result to the
    // correct quadrant below
    float phi = (r == 0.0) ? 0.0 :
        (PI * ((abs(u.y) - abs(u.x)) / r + 1.0));

    float f = r * sqrt(2.0 - r * r);

    // abs() around f * cos/sin(phi) is necessary because they can return
    // negative 0 due to floating precision
    float x = ifrit_SignPreserveZero(u.x) * abs(f * cos(phi));
    float y = ifrit_SignPreserveZero(u.y) * abs(f * sin(phi));
    float z = ifrit_SignPreserveZero(d) * (1.0 - r * r);

    return normalize(vec3(x, y, z));
}
