
struct Meshlet {
  uint vertex_offset;
  uint triangle_offset;
  uint vertex_count;
  uint triangle_count;
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
