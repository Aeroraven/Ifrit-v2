#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/base/Mesh.h"

namespace Ifrit::Runtime::Geometry
{
    IFRIT_RUNTIME_API void AllocateMeshGPUResources(RHI::RhiBackend* rhi, Mesh* mesh, u32 maxVertices, u32 maxIndices);
    IFRIT_RUNTIME_API void ForceMeshObjectBufferSync(RHI::RhiBackend* rhi, Mesh* mesh);

} // namespace Ifrit::Runtime::Geometry