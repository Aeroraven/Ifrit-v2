#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/rhi/common/RhiForwardingTypes.h"
#include "ifrit/core/logging/Logging.h"

#ifndef IF_MODULE_RHI
    #ifndef __INTELLISENSE__
        #error "Invalid inclusion"
    #endif
#endif

namespace Ifrit::RHI
{
    IF_FORCEINLINE u32 GetNumVerticesFromPrimitive(RhiRasterizerTopology topo, u32 numPrims)
    {
        switch (topo)
        {
            case RhiRasterizerTopology::TriangleList:
                return numPrims * 3;
            case RhiRasterizerTopology::Line:
                return numPrims * 2;
            case RhiRasterizerTopology::Point:
                return numPrims;
            default:
                IF_UNLIKELY
                {
                    IF_LOG_CRITICAL("RhiCmdTranslationHelper", "Invalid topology type");
                    return 0;
                }
        }
    }
} // namespace Ifrit::RHI