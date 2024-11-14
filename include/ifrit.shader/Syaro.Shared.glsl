

RegisterUniform(bLocalTransform,{
    mat4 m_localToWorld;
});
RegisterStorage(bMeshDataRef,{
    uint vertexBuffer;
    uint meshletBuffer;
    uint meshletVertexBuffer;
    uint meshletIndexBuffer;
    uint meshletCullBuffer;
    uint bvhNodeBuffer;
    uint clusterGroupBuffer;
    uint meshletInClusterBuffer;
    uint cpCounterBuffer;
    uint pad1;
    uint pad2;
    uint pad3;
    vec4 boundingSphere;
});
RegisterStorage(bInstanceDataRef,{
    uint cpQueueBuffer;
    uint cpCounterBuffer;
    uint filteredMeshletsBuffer;
    uint pad;
});
RegisterUniform(bPerframeView,{
    PerFramePerViewData data;
});
RegisterStorage(bPerObjectRef,{
    PerObjectData data[];
});