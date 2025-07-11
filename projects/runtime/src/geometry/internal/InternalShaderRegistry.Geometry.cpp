#include "ifrit/runtime/geometry/internal/InternalShaderRegistry.Geometry.h"
#include "ifrit/runtime/material/internal/ShaderRegistryMacros.h"

namespace Ifrit::Runtime::Internal
{
    IFRIT_APIDECL void RegisterRuntimeInternalShadersGeometry(ShaderRegistry* shaderRegistry)
    {
        const auto& ISTGeo = kIntShaderTableGeometry;

        REG_COMPUTE_NEO(
            ISTGeo.SurfReconGridBuildCS, "Meshing/SurfaceRecon/SurfRecon.GridBuild", "SurfReconGridBuildCS");
        REG_COMPUTE_NEO(
            ISTGeo.SurfReconFilterCellsCS, "Meshing/SurfaceRecon/SurfRecon.FilterCells", "SurfReconFilterCellsCS");
        REG_COMPUTE_NEO(
            ISTGeo.SurfReconFilterBlocksCS, "Meshing/SurfaceRecon/SurfRecon.FilterBlocks", "SurfReconFilterBlocksCS");
        REG_COMPUTE_NEO(
            ISTGeo.SurfReconVoxelMeshingCS, "Meshing/SurfaceRecon/SurfRecon.VoxelMeshing", "SurfReconVoxelMeshingCS");
        REG_COMPUTE_NEO(ISTGeo.SurfReconPrepareDispArgsCS, "Meshing/SurfaceRecon/SurfRecon.PrepareDispArgs",
            "SurfReconPrepareDispArgsCS");
        REG_COMPUTE_NEO(ISTGeo.SurfReconPrepareDrawArgsCS, "Meshing/SurfaceRecon/SurfRecon.PrepareDrawArgs",
            "SurfReconPrepareDrawArgsCS");
    }
} // namespace Ifrit::Runtime::Internal