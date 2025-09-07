#pragma once
#include "RhiApi.h"
#include "RhiBaseTypes.h"
#include "ifrit/rhi/common/RhiCommandList.h"
#include "ifrit/rhi/common/RhiDevice.h"

namespace Ifrit::RHI
{

    class IFRIT_RHI_API RhiDeviceProcs : public RhiDeviceChild
    {
    public:
        virtual Owner<RhiCommandListContext> AcquireCommandListContext(
            ERhiCommandListPipelineType type, RhiCommandListBase* immed)                                        = 0;
        virtual void ReleaseCommandListContext(Owner<RhiCommandListContext> context, RhiCommandListBase* immed) = 0;
        virtual Ref<RhiTaskSubmission> SubmitCommandListContext(
            RhiCommandListContext* context, RhiCommandListBase* immed, Vec<Ref<RhiTaskSubmission>> toWait) = 0;
    };
} // namespace Ifrit::RHI