
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

vec3 ifrit_ConcentricOctahedralTransform(vec2 UV){
    // https://zhuanlan.zhihu.com/p/408898601
    // https://fileadmin.cs.lth.se/graphics/research/papers/2008/simdmapping/clarberg_simdmapping08_preprint.pdf
    // Port from ifrit.core.math

    const float PI = 3.14159265358979323846;

    vec2 sampleOffset = UV * 2.0 - vec2(1.0);

    float u = sampleOffset.x;
    float v = sampleOffset.y;
    float d = 1.0 - abs(u) - abs(v);
    float r = 1.0 - abs(d);

    float z = (d > 0.0 ? 1.0 : -1.0) * (1.0 - r * r);
    float theta = PI / 4.0 * ((abs(v) - abs(u)) / (r + 1.0));
    float sinT = sin(theta) * (v >= 0.0 ? 1.0 : -1.0);
    float cosT = cos(theta) * (u >= 0.0 ? 1.0 : -1.0);
    float x = cosT * r * sqrt(2.0 - z * z);
    float y = sinT * r * sqrt(2.0 - z * z);
    return vec3(x, y, z);
}