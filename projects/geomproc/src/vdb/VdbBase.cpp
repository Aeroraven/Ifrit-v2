#include "ifrit/geomproc/vdb/VdbBase.h"
#include <sstream>
#include "openvdb/openvdb/openvdb.h"
#include "openvdb/io/Stream.h"

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

        VDBDescriptor          desc;
        desc.m_VdbData = grids;
        return desc;
    }

} // namespace Ifrit::GeometryProc::VDB