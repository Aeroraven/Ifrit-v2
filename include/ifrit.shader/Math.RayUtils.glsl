
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


vec2 ifrit_RayIntersectWithUnitRect2D(vec2 Origin, vec2 Dir){
    float tx1 = (0.0 - Origin.x) / Dir.x;
    float tx2 = (1.0 - Origin.x) / Dir.x;
    float ty1 = (0.0 - Origin.y) / Dir.y;
    float ty2 = (1.0 - Origin.y) / Dir.y;
    float tmin = max(min(tx1, tx2), min(ty1, ty2));
    float tmax = min(max(tx1, tx2), max(ty1, ty2));
    if(tmax < 0.0 || tmin > tmax) return vec2(-1.0, -1.0);
    return vec2(tmin, tmax);
}