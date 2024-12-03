
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



struct Meshlet {
    uint vertex_offset;
    uint triangle_offset;
    uint vertex_count;
    uint triangle_count;
    vec4 normalCone;
    vec4 normalConeApex;
    vec4 boundSphere;
};

RegisterUniform(bLocalTransform,{
    mat4 m_localToWorld;
});
RegisterStorage(bMeshDataRef,{
    vec4 boundingSphere;
    uint vertexBuffer;
    uint normalBufferId;
    uint tangentBufferId;
    uint uvBufferId;
    uint meshletBuffer;
    uint meshletVertexBuffer;
    uint meshletIndexBuffer;
    uint meshletCullBuffer;
    uint bvhNodeBuffer;
    uint clusterGroupBuffer;
    uint meshletInClusterBuffer;
    uint cpCounterBuffer;
    uint pad0;
    uint pad2;
    uint pad3;
});
RegisterStorage(bInstanceDataRef,{
    uint cpQueueBuffer;
    uint cpCounterBuffer;
    uint filteredMeshletsBuffer;
    uint materialId;
});
RegisterUniform(bPerframeView,{
    PerFramePerViewData data;
});
RegisterStorage(bPerObjectRef,{
    PerObjectData data[];
});