
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