#pragma once
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/base/Base.h"

namespace Ifrit::Runtime
{

    struct ProfileBriefReport
    {
        String mEventName;
        f32    mAvgDurationMs;
        f32    mMaxDurationMs;
        f32    mMinDurationMs;
    };

    struct ProfileDataManagerInternal;
    class IFRIT_RUNTIME_API ProfileDataManager
    {
    public:
        ProfileDataManager();
        ~ProfileDataManager();

        void                    FrameProceed();
        void                    ReportAccumulateEvent(const String& eventName, f32 durationMs);
        Vec<ProfileBriefReport> GetBriefReport() const;

        void                    SetMaxFramesToKeep(u32 maxFrames);
        u32                     GetMaxFramesToKeep() const;

    private:
        struct ProfileDataManagerInternal* mInternalData;
    };
} // namespace Ifrit::Runtime