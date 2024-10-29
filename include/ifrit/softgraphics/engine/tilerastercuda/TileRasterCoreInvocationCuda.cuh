#pragma once
#ifdef IFRIT_FEATURE_CUDA
#include "ifrit/softgraphics/core/cuda/CudaUtils.cuh"
#include "ifrit/softgraphics/core/definition/CoreDefs.h"
#include "ifrit/softgraphics/engine/base/TypeDescriptor.h"


#include "ifrit/softgraphics/engine/base/Constants.h"
#include "ifrit/softgraphics/engine/base/Shaders.h"
#include "ifrit/softgraphics/engine/base/Structures.h"
#include "ifrit/softgraphics/engine/tileraster/TileRasterCommon.h"
#include "ifrit/softgraphics/engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"


namespace Ifrit::SoftRenderer::TileRaster::CUDA::Invocation {
enum GeometryGenerationPipelineType {
  IFCUINVO_GEOMETRY_GENERATION_UNDEFINED = 0,
  IFCUINVO_GEOMETRY_GENERATION_CONVENTIONAL = 1,
  IFCUINVO_GEOMETRY_GENERATION_MESHSHADER = 2
};

struct RenderingInvocationArgumentSet {
  char *dVertexBuffer;
  TypeDescriptorEnum *dVertexTypeDescriptor;
  int *dIndexBuffer;
  VertexShader *dVertexShader;
  FragmentShader *dFragmentShader;
  GeometryShader *dGeometryShader;
  ifloat4 **dColorBuffer;
  ifloat4 **dHostColorBuffer;
  ifloat4 **hColorBuffer;
  uint32_t dHostColorBufferSize;
  float *dDepthBuffer;
  ifloat4 *dPositionBuffer;
  TileRasterDeviceContext *deviceContext;
  int totalIndices;
  bool doubleBuffering;
  ifloat4 **dLastColorBuffer;
  IfritPolygonMode polygonMode = IF_POLYGON_MODE_FILL;
  ifloat4 *hClearColors;
  float hClearDepth;

  MeshShader *dMeshShader;
  TaskShader *dTaskShader;
  iint3 gMeshShaderLocalSize;
  int gMeshShaderNumWorkGroups;
  GeometryGenerationPipelineType gGeometryPipelineType;
  int gMeshShaderAttributes;
};
void invokeCudaRenderingClear(const RenderingInvocationArgumentSet &args)
    IFRIT_AP_NOTHROW;
void invokeCudaRendering(const RenderingInvocationArgumentSet &args)
    IFRIT_AP_NOTHROW;

void invokeFragmentShaderUpdate(FragmentShader *dFragmentShader)
    IFRIT_AP_NOTHROW;
void updateFrameBufferConstants(uint32_t width, uint32_t height);
void updateScissorTestData(const ifloat4 *scissorAreas, int numScissors,
                           bool scissorEnable);
void initCudaRendering();
void updateVertexLayout(TypeDescriptorEnum *dVertexTypeDescriptor,
                        int attrCounts);

int *getIndexBufferDeviceAddr(const int *hIndexBuffer, uint32_t indexBufferSize,
                              int *dOldIndexBuffer);
char *getVertexBufferDeviceAddr(const char *hVertexBuffer, uint32_t bufferSize,
                                char *dOldBuffer);
TypeDescriptorEnum *
getTypeDescriptorDeviceAddr(const TypeDescriptorEnum *hBuffer,
                            uint32_t bufferSize,
                            TypeDescriptorEnum *dOldBuffer);
float *getDepthBufferDeviceAddr(uint32_t bufferSize, float *dOldBuffer);
ifloat4 *getPositionBufferDeviceAddr(uint32_t bufferSize, ifloat4 *dOldBuffer);
void getColorBufferDeviceAddr(const std::vector<ifloat4 *> &hColorBuffer,
                              std::vector<ifloat4 *> &dhColorBuffer,
                              ifloat4 **&dColorBuffer, uint32_t bufferSize,
                              std::vector<ifloat4 *> &dhOldColorBuffer,
                              ifloat4 **dOldBuffer);
void updateAttributes(uint32_t attributeCounts);
void updateVarying(uint32_t varyingCounts);
void updateVertexCount(uint32_t vertexCount);

char *deviceMalloc(uint32_t size);
void deviceFree(char *ptr);
void createTexture(uint32_t texId, const IfritImageCreateInfo &createInfo,
                   float *data);
void createSampler(uint32_t slotId, const IfritSamplerT &samplerState);
void createDeviceBuffer(uint32_t slotId, int bufferSize);
void copyHostBufferToBuffer(const void *srcBuffer, int dstSlot, int size);

void setBlendFunc(IfritColorAttachmentBlendState blendState);
void setDepthFunc(IfritCompareOp depthFunc);
void setCullMode(IfritCullMode cullMode);
void setMsaaSampleBits(IfritSampleCountFlagBits sampleBits);
} // namespace Ifrit::SoftRenderer::TileRaster::CUDA::Invocation
#endif