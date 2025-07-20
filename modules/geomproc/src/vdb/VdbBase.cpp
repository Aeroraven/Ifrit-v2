#include "ifrit/geomproc/vdb/VdbBase.h"
#include <sstream>
#include "openvdb/openvdb/openvdb.h"
#include "openvdb/io/Stream.h"
#include "ifrit/core/logging/Logging.h"
#include "openvdb/tools/Interpolation.h"

namespace Ifrit::GeometryProc::VDB
{
    static void OpenVDBInit()
    {
        static bool isInitialized = false;
        if (isInitialized)
            return;
        openvdb::initialize();
        isInitialized = true;
    }

    IFRIT_APIDECL VDBDescriptor LoadVdbFromString(String s)
    {
        OpenVDBInit();

        std::stringstream      ss(s);
        openvdb::io::Stream    vdbStream(static_cast<std::istream&>(ss));
        openvdb::GridPtrVecPtr grids = vdbStream.getGrids();

        // find the first float grid
        for (const auto& grid : *grids)
        {
            if (grid->isType<openvdb::FloatGrid>())
            {
                VDBDescriptor desc;
                desc.m_VdbData = grid->copyGrid();
                return desc;
                break;
            }
        }
    }

    IFRIT_APIDECL void PrintVdbMeta(const VDBDescriptor& p)
    {
        auto               gridPtr   = std::any_cast<openvdb::GridBase::Ptr>(p.m_VdbData);
        auto               floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(gridPtr);

        openvdb::CoordBBox indexBBox = floatGrid->evalActiveVoxelBoundingBox();

        // Convert to world space
        openvdb::Vec3f     minWorld =
            floatGrid->indexToWorld(openvdb::Vec3f(indexBBox.min().x(), indexBBox.min().y(), indexBBox.min().z()));
        openvdb::Vec3f maxWorld =
            floatGrid->indexToWorld(openvdb::Vec3f(indexBBox.max().x(), indexBBox.max().y(), indexBBox.max().z()));
        auto minIndex         = indexBBox.min();
        auto maxIndex         = indexBBox.max();
        auto voxelSize        = floatGrid->voxelSize();
        auto activeVoxelCount = floatGrid->activeVoxelCount();

        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*floatGrid);

        IF_LOG_INFO("VDBUtil", "VDB Meta:");
        IF_LOG_INFO("VDBUtil", "  Grid Type: {}", floatGrid->type());
        IF_LOG_INFO("VDBUtil", "  Value Type: {}", floatGrid->valueType());
        IF_LOG_INFO("VDBUtil", "  Active Voxel Count: {}", activeVoxelCount);
        IF_LOG_INFO("VDBUtil", "  Index BBox: min({},{},{}) max({},{},{})", minIndex.x(), minIndex.y(), minIndex.z(),
            maxIndex.x(), maxIndex.y(), maxIndex.z());
        IF_LOG_INFO("VDBUtil", "  World BBox: min({},{},{}) max({},{},{})", minWorld.x(), minWorld.y(), minWorld.z(),
            maxWorld.x(), maxWorld.y(), maxWorld.z());

        //
        openvdb::Vec3f worldPos(0, 12, 0);
        auto           pv = sampler.wsSample(worldPos);
        IF_LOG_INFO("VDBUtil", "  Sampled Value at World Position (0,0,0): {}", pv);
    }

} // namespace Ifrit::GeometryProc::VDB