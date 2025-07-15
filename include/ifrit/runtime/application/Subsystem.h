#pragma once

#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/base/Base.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL ISubsystem
    {
    public:
        virtual void OnInitialize(IApplication* app) = 0;
        virtual void OnShutdown()                    = 0;
        virtual void OnFrameBegin()                  = 0;
        virtual void OnFrameEnd()                    = 0;
    };
} // namespace Ifrit::Runtime