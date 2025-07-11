
#pragma once
#include "ifrit/runtime/material/ShaderRegistry.h"
#include "ifrit/runtime/base/Base.h"

namespace Ifrit::Runtime::Internal
{

#define DECLARE_VS(name) name "/VS"
#define DECLARE_FS(name) name "/PS"
#define DECLARE_CS(name) name "/CS"
#define DECLARE_MS(name) name "/MS"

#define SDEF IF_CONSTEXPR static const char*
    IFRIT_RUNTIME_API void RegisterRuntimeInternalShadersGeometry(ShaderRegistry* shaderRegistry);

    static struct InternalShaderTableGeometry
    {

        SDEF SurfReconGridBuildCS       = DECLARE_CS("Geometry/SurfRecon.GridRebuild");
        SDEF SurfReconFilterCellsCS     = DECLARE_CS("Geometry/SurfRecon.FilterCells");
        SDEF SurfReconFilterBlocksCS    = DECLARE_CS("Geometry/SurfRecon.FilterBlocks");
        SDEF SurfReconVoxelMeshingCS    = DECLARE_CS("Geometry/SurfRecon.VoxelMeshing");
        SDEF SurfReconPrepareDispArgsCS = DECLARE_CS("Geometry/SurfRecon.PrepareDispArgs");
        SDEF SurfReconPrepareDrawArgsCS = DECLARE_CS("Geometry/SurfRecon.PrepareDrawArgs");

    } kIntShaderTableGeometry;

#undef SDEF

#undef DECLARE_VS
#undef DECLARE_FS
#undef DECLARE_CS
#undef DECLARE_MS

} // namespace Ifrit::Runtime::Internal