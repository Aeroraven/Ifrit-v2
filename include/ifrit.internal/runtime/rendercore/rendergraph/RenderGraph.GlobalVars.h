#pragma once
#include "ifrit/runtime/rendercore/rendergraph/RenderGraph.h"
#include "ifrit/core/console/ConsoleObject.h"

namespace Ifrit::Runtime::RenderCore::RDG
{

    enum ERDGMemoryHeapType
    {
        DeviceLocal = 0,
        HostShared  = 1,
        Count       = 2,
    };

    struct RDGSubresourcePair
    {
        u32                                mMipLevel   = 0;
        u32                                mArrayLayer = 0;

        IF_NODISCARD constexpr inline bool operator==(const RDGSubresourcePair& other) const noexcept
        {
            return mMipLevel == other.mMipLevel && mArrayLayer == other.mArrayLayer;
        }
    };

    static TConsoleVariable<u32> cvRDGEnableAsyncCompute("cv.RDG.AsyncCompute", 1,
        "Enable Async Compute Queue in Render Graph: 0 - Disabled, 1 - Manual, 2 - Forced", CVF_ReadOnly);
    static TConsoleVariable<u32> cvRDGEnableAsyncTransfer("cv.RDG.AsyncTransfer", 1,
        "Enable Async Transfer Queue in Render Graph: 0 - Disabled, 1 - Manual, 2 - Forced", CVF_ReadOnly);
    static TConsoleVariable<u32> cvRDGResourceReusingStrategy("cv.RDG.ResourceReusingStrategy", 2,
        "Resource Reusing Strategy: 0 - No Reuse, 1 - Resource Reusing, 2 - Memory Aliasing", CVF_ReadOnly);

} // namespace Ifrit::Runtime::RenderCore::RDG

namespace std
{
    template <> struct hash<Ifrit::Runtime::RenderCore::RDG::RDGSubresourcePair>
    {
        size_t operator()(const Ifrit::Runtime::RenderCore::RDG::RDGSubresourcePair& pair) const noexcept
        {
            size_t h1 = std::hash<Ifrit::u32>{}(pair.mMipLevel);
            size_t h2 = std::hash<Ifrit::u32>{}(pair.mArrayLayer);
            return h1 ^ (h2 << 1); // or use boost::hash_combine
        }
    };
} // namespace std
