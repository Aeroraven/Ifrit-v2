#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/common/Pch.h"

namespace Ifrit::Runtime
{
    class ProfileStatScopeData;
    class IFRIT_RUNTIME_API ProfileStatScope
    {
    public:
        ProfileStatScope(const RHI::RhiCommandList* cmdList, const String& eventName);
        ~ProfileStatScope();

    private:
        ProfileStatScopeData* mData = nullptr;
    };

    class IFRIT_RUNTIME_API HostProfileStatScope
    {
    public:
        HostProfileStatScope(const String& eventName);
        ~HostProfileStatScope();

    private:
        ProfileStatScopeData* mData = nullptr;
    };

    IFRIT_RUNTIME_API Owner<HostProfileStatScope> CreateHostProfileStatScope(const String& eventName);

// Macros for host profiling
#define IFRIT_STAT_HOST_SCOPE(eventName)                                      \
    Owner<Ifrit::Runtime::HostProfileStatScope> _profileStatScope##__LINE__ = \
        Ifrit::Runtime::CreateHostProfileStatScope(eventName);

} // namespace Ifrit::Runtime