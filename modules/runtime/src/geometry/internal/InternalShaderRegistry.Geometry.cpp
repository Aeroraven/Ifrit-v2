#include "ifrit/runtime/geometry/internal/InternalShaderRegistry.Geometry.h"
#include "ifrit/runtime/rendercore/shadercore/internal/ShaderRegistryMacros.h"

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

        REG_COMPUTE_NEO(ISTGeo.SurfReconVertexCompactCS, "Meshing/SurfaceRecon/SurfRecon.VertexCompact",
            "SurfReconVertexCompactCS");
        REG_COMPUTE_NEO(ISTGeo.SurfReconVertexNormalBuildCS, "Meshing/SurfaceRecon/SurfRecon.VertexNormalBuild",
            "SurfReconVertexNormalBuildCS");
        REG_COMPUTE_NEO(ISTGeo.SurfReconVertexDensityCS, "Meshing/SurfaceRecon/SurfRecon.VertexDensity",
            "SurfReconVertexDensityCS");
        REG_COMPUTE_NEO(ISTGeo.SurfReconComputeDispArgsCS, "Meshing/SurfaceRecon/SurfRecon.ComputeDispArgs",
            "SurfReconComputeDispArgsCS");

        REG_COMPUTE_NEO(ISTGeo.SurfReconDebugVertexVisualizeCS, "Meshing/SurfaceRecon/SurfRecon.Debug.VertexVisualize",
            "SurfReconDebugVertexVisualizeCS");
    }
} // namespace Ifrit::Runtime::Internal
