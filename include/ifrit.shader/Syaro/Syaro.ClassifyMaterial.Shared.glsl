
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



struct MaterialCounter{
    uint counter;
    uint offset;
};

struct MaterialPassIndirectCommand{
    uint x;
    uint y;
    uint z;
};

RegisterStorage(bMaterialCounter,{
    uint totalCounter;
    uint pad;
    MaterialCounter data[];
});

RegisterStorage(bMaterialPixelList,{
    uint data[];
});
RegisterStorage(bPerPixelCounterOffset,{
    uint data[];
});
RegisterStorage(bMaterialPassIndirectCommand,{
    MaterialPassIndirectCommand data[];
});

layout(binding = 0, set = 1) uniform MaterialPassData{
    uint materialDepthRef;
    uint materialCounterRef;
    uint materialPixelListRef;
    uint perPixelCounterOffsetRef;
    uint indirectCommandRef;
    uint debugImageRef;
} uMaterialPassData;

layout(push_constant) uniform MaterialPassPushConstant{
    uint renderWidth;
    uint renderHeight;
    uint totalMaterials;
} uMaterialPassPushConstant;