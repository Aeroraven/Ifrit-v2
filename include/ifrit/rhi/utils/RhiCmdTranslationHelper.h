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
    IF_FORCEINLINE u32 GetNumVerticesFromPrimitive(ERhiRasterizerTopology topo, u32 numPrims)
    {
        switch (topo)
        {
            case ERhiRasterizerTopology::TriangleList:
                return numPrims * 3;
            case ERhiRasterizerTopology::Line:
                return numPrims * 2;
            case ERhiRasterizerTopology::Point:
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