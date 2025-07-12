#include "ifrit/geomproc/vdb/VdbSampler.h"
#include "ifrit/geomproc/sampler/PoissonSampler.h"
#include "openvdb/openvdb.h"
#include "openvdb/io/Stream.h"
#include "openvdb/tools/Interpolation.h"

namespace Ifrit::GeometryProc::VDB
{
    IFRIT_APIDECL Vec<Vector3f> PoissonSampleVdbZpcReference(const VDBDescriptor& vdbDesc, f32 dx, u32 ppc)
    {
        auto               gridPtr   = std::any_cast<openvdb::GridBase::Ptr>(vdbDesc.m_VdbData);
        auto               floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(gridPtr);

        openvdb::CoordBBox indexBBox = floatGrid->evalActiveVoxelBoundingBox();
        openvdb::Vec3f     minWorld =
            floatGrid->indexToWorld(openvdb::Vec3f(indexBBox.min().x(), indexBBox.min().y(), indexBBox.min().z()));
        openvdb::Vec3f maxWorld =
            floatGrid->indexToWorld(openvdb::Vec3f(indexBBox.max().x(), indexBBox.max().y(), indexBBox.max().z()));

        Vector3f                            minWorldVec(minWorld.x(), minWorld.y(), minWorld.z());
        Vector3f                            maxWorldVec(maxWorld.x(), maxWorld.y(), maxWorld.z());

        Sampler::PoissonSamplerArgs<f32, 3> args;
        args.m_CellDx          = dx;
        args.m_ParticlePerCell = ppc;
        args.m_MinBound        = minWorldVec;
        args.m_MaxBound        = maxWorldVec;

        auto refs = Sampler::LoadZpcPoissonSamplerReferences();

        auto vdbSampler = openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler>(*floatGrid);
        auto predicate  = [&](const Vector3f& pos) -> bool {
            openvdb::Vec3d indexPos = floatGrid->worldToIndex(openvdb::Vec3d(pos.x, pos.y, pos.z));
            f32            value    = vdbSampler.isSample(indexPos);
            return value < 0.0f;
        };
        return Sampler::PoissonSample<f32, 3>(args, refs, predicate);
    }
} // namespace Ifrit::GeometryProc::VDB