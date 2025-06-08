
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

// === Screen Probe ===

const bool kVisTracingHierarchy = false;
const bool kSkipScreenTrace = true; // Skip screen trace for debugging purposes
const bool kEnableOctMapBorderFix = true;

vec3 AyaShared_GetScreenProbeTraceCoord(uvec2 TraceRayCoord, vec2 Jitter){
    vec2 UV = (vec2(TraceRayCoord)+Jitter + vec2(0.5)) / vec2(kAyanami_ScreenProbeProbeHemiRes);
    //return ifrit_ConcentricOctahedralTransform(UV);
    return ifrit_ConcentricOctahedralTransform(UV).xyz;
}