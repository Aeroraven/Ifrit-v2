#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/rhi/common/RhiBaseTypes.h"
#include "ifrit/runtime/forwarding/FwdScene.h"

namespace Ifrit::Runtime
{
    class IApplication;
    class IFRIT_APIDECL ISubsystem
    {
    public:
        virtual ~ISubsystem()                                                                         = default;
        virtual void                          OnInitialize(IApplication* app)                         = 0;
        virtual void                          OnShutdown()                                            = 0;
        virtual void                          OnFrameBegin()                                          = 0;
        virtual void                          OnFrameEnd()                                            = 0;
        virtual Owner<RHI::RhiTaskSubmission> OnPreRendering(RHI::RhiTaskSubmission* prevSubmission)  = 0;
        virtual Owner<RHI::RhiTaskSubmission> OnPostRendering(RHI::RhiTaskSubmission* prevSubmission) = 0;
        virtual void                          OnUpdate(Scene* scene)                                  = 0;
    };
} // namespace Ifrit::Runtime
