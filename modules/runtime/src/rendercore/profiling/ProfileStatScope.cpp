#include "ifrit/runtime/rendercore/profiling/ProfileStatScope.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/rendercore/profiling/ProfileDataManager.h"

namespace Ifrit::Runtime
{
    struct ProfileStatScopeData
    {
        const RHI::RhiCommandList* mCmdList;
        String                     mEventName;

        ProfileStatScopeData(const RHI::RhiCommandList* cmdList, const String& eventName)
            : mCmdList(cmdList), mEventName(eventName)
        {
        }
    };
    IFRIT_APIDECL ProfileStatScope::ProfileStatScope(const RHI::RhiCommandList* cmdList, const String& eventName)
    {
        mData                   = new ProfileStatScopeData(cmdList, eventName);
        auto profileDataManager = GetActiveApplication()->GetProfileDataManager();
        profileDataManager->ReportBeginEvent(cmdList, eventName);
    }

    IFRIT_APIDECL ProfileStatScope::~ProfileStatScope()
    {
        auto profileDataManager = GetActiveApplication()->GetProfileDataManager();
        profileDataManager->ReportEndEvent(mData->mCmdList, mData->mEventName);
        delete mData;
        mData = nullptr;
    }

    IFRIT_APIDECL HostProfileStatScope::HostProfileStatScope(const String& eventName)
    {
        mData                   = new ProfileStatScopeData(nullptr, eventName);
        auto profileDataManager = GetActiveApplication()->GetProfileDataManager();
        profileDataManager->ReportHostBeginEvent(eventName);
    }
    IFRIT_APIDECL HostProfileStatScope::~HostProfileStatScope()
    {
        auto profileDataManager = GetActiveApplication()->GetProfileDataManager();
        profileDataManager->ReportHostEndEvent(mData->mEventName);
        delete mData;
        mData = nullptr;
    }
    IFRIT_APIDECL Owner<HostProfileStatScope> CreateHostProfileStatScope(const String& eventName)
    {
        return MakeOwner<HostProfileStatScope>(eventName);
    }
} // namespace Ifrit::Runtime
