# Ifrit-v2

GPU/CPU-Parallelized tile-based software rasterizer.



## MT-CPU Renderer Abstraction

```C++
class TileRasterRenderer{
public:
    void bindFrameBuffer(FrameBuffer& frameBuffer);
    void bindVertexBuffer(const VertexBuffer& vertexBuffer);
    void bindIndexBuffer(const std::vector<int>& indexBuffer);
    void bindVertexShader(VertexShader& vertexShader);
    void bindFragmentShader(FragmentShader& fragmentShader);
    void intializeRenderContext();
    void setBlendFunc(IfritColorAttachmentBlendState state);
    void setDepthFunc(IfritCompareOp depthFunc);

    void optsetDepthTestEnable(bool opt);

    void render(bool clearFramebuffer);
    void clear();
    void init();
}
```



## CUDA Renderer Abstraction

``` C++
class TileRasterRendererCuda{
 public:
		void init();
		void bindFrameBuffer(FrameBuffer& frameBuffer, bool useDoubleBuffer = true);
		void bindVertexBuffer(const VertexBuffer& vertexBuffer);
		void bindIndexBuffer(const std::vector<int>& indexBuffer);
		void bindVertexShader(VertexShader* vertexShader, VaryingDescriptor& varyingDescriptor);
		void bindFragmentShader(FragmentShader* fragmentShader);
		void bindGeometryShader(GeometryShader* geometryShader);
		void bindMeshShader(MeshShader* meshShader, VaryingDescriptor& varyingDescriptor, iint3 localSize);
		void bindTaskShader(TaskShader* taskShader, VaryingDescriptor& varyingDescriptor);

		void createTexture(int slotId, const IfritImageCreateInfo& createInfo);
		void createSampler(int slotId, const IfritSamplerT& samplerState);
		void generateMipmap(int slotId, IfritFilter filter);
		void blitImage(int srcSlotId, int dstSlotId, const IfritImageBlit& region, IfritFilter filter);
		void copyHostBufferToImage(void* srcBuffer, int dstSlot, const std::vector<IfritBufferImageCopy>& regions);

		void createBuffer(int slotId, int bufSize);
		void copyHostBufferToBuffer(const void* srcBuffer, int dstSlot, int size);

		void setScissors(const std::vector<ifloat4>& scissors);
		void setScissorTestEnable(bool option);

		void setMsaaSamples(IfritSampleCountFlagBits msaaSamples);

		void setRasterizerPolygonMode(IfritPolygonMode mode);
		void setBlendFunc(IfritColorAttachmentBlendState state);
		void setDepthFunc(IfritCompareOp depthFunc);
		void setDepthTestEnable(bool option);
		void setCullMode(IfritCullMode cullMode);
		void setClearValues(const std::vector<ifloat4>& clearColors, float clearDepth);

		void clear();
		void drawElements();
		void drawMeshTasks(int numWorkGroups, int firstWorkGroup);
}
```

