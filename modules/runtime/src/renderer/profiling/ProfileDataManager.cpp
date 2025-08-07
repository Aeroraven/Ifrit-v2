#include "ifrit/runtime/renderer/profiling/ProfileDataManager.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include <deque>
#include <chrono>

namespace Ifrit::Runtime
{
    struct ProfileDataManagerInternal
    {
        enum class ETimerStatus
        {
            Idle,
            Running,
            Stopped
        };

        enum class EProfileDevice
        {
            CPU,
            GPU
        };

        struct ProfileEventEntry
        {
            f32 mDurationMs;
            u32 mFrameIndex;
        };

        struct ProfileEventData
        {
            ETimerStatus                  mStatus = ETimerStatus::Idle;
            std::deque<ProfileEventEntry> mRecentEntries;
            f32                           mTotalDurationMs = 0.0f;
            f32                           mMaxDurationMs   = 0.0f;
            f32                           mMinDurationMs   = std::numeric_limits<f32>::max();
            u32                           mCount           = 0;

            void                          AddEntry(f32 duration, u32 frameIndex, u32 maxFrames)
            {

                // if last entry has the same frame index, update it
                if (!mRecentEntries.empty() && mRecentEntries.back().mFrameIndex == frameIndex)
                {
                    mRecentEntries.back().mDurationMs += duration;
                    return;
                }

                // Add new entry
                mRecentEntries.push_back({ duration, frameIndex });

                // Remove old entries beyond the frame window
                while (!mRecentEntries.empty() && (frameIndex - mRecentEntries.front().mFrameIndex) >= maxFrames)
                {
                    mRecentEntries.pop_front();
                }

                // Recalculate statistics
                RecalculateStats();
            }

            void RecalculateStats()
            {
                if (mRecentEntries.empty())
                {
                    mTotalDurationMs = 0.0f;
                    mMaxDurationMs   = 0.0f;
                    mMinDurationMs   = std::numeric_limits<f32>::max();
                    mCount           = 0;
                    return;
                }

                mTotalDurationMs = 0.0f;
                mMaxDurationMs   = 0.0f;
                mMinDurationMs   = std::numeric_limits<f32>::max();
                mCount           = static_cast<u32>(mRecentEntries.size());

                for (const auto& entry : mRecentEntries)
                {
                    if (entry.mDurationMs > 1145000000.0f)
                    {
                        continue;
                    }
                    mTotalDurationMs += entry.mDurationMs;
                    mMaxDurationMs = std::max(mMaxDurationMs, entry.mDurationMs);
                    mMinDurationMs = std::min(mMinDurationMs, entry.mDurationMs);
                }
            }
        };

        HashMap<String, ProfileEventData>              mEventDataMap;
        HashMap<String, Vec<Ref<RHI::RhiDeviceTimer>>> mEventTimers;
        HashMap<String, u32>                           mAvailableTimerId;
        HashMap<String, EProfileDevice>                mEventDeviceType;
        u32                                            mCurrentFrameIndex = 0;
        u32                                            mMaxFramesToKeep   = 60; // Default: keep last 60 frames
        u64                                            mHostTimestamp     = 0;
    };

    ProfileDataManager::ProfileDataManager() { mInternalData = new ProfileDataManagerInternal(); }

    ProfileDataManager::~ProfileDataManager() { delete mInternalData; }

    void ProfileDataManager::FrameProceed()
    {
        mInternalData->mCurrentFrameIndex++;

        // Optional: Clean up old entries for all events
        for (auto& [eventName, eventData] : mInternalData->mEventDataMap)
        {
            if (eventData.mStatus == ProfileDataManagerInternal::ETimerStatus::Idle)
            {
                eventData.mStatus = ProfileDataManagerInternal::ETimerStatus::Idle;
                for (auto i = 0; i < mInternalData->mAvailableTimerId[eventName]; ++i)
                {
                    ReportAccumulateEvent(eventName, mInternalData->mEventTimers[eventName][i]->GetElapsedMs());
                }
                mInternalData->mAvailableTimerId[eventName] = 0;
            }
            // Remove entries older than the frame window
            while (!eventData.mRecentEntries.empty()
                && (mInternalData->mCurrentFrameIndex - eventData.mRecentEntries.front().mFrameIndex)
                    >= mInternalData->mMaxFramesToKeep)
            {
                eventData.mRecentEntries.pop_front();
            }
            eventData.RecalculateStats();
        }
    }

    void ProfileDataManager::ReportAccumulateEvent(const String& eventName, f32 durationMs)
    {
        auto& eventData = mInternalData->mEventDataMap[eventName];
        eventData.AddEntry(durationMs, mInternalData->mCurrentFrameIndex, mInternalData->mMaxFramesToKeep);
    }
    void ProfileDataManager::ReportBeginEvent(const RHI::RhiCommandList* cmdList, const String& eventName)
    {
        auto curTimerId                            = mInternalData->mAvailableTimerId[eventName];
        mInternalData->mEventDeviceType[eventName] = ProfileDataManagerInternal::EProfileDevice::GPU;
        if (curTimerId >= mInternalData->mEventTimers[eventName].size())
        {
            mInternalData->mEventTimers[eventName].resize(curTimerId + 1);
            mInternalData->mEventTimers[eventName][curTimerId] = GetActiveApplication()->GetRhi()->CreateDeviceTimer();
        }
        auto& eventTimer = mInternalData->mEventTimers[eventName][curTimerId];
        eventTimer->Start(cmdList);
        auto& eventData = mInternalData->mEventDataMap[eventName];
        IF_LOG_ASSERTION("ProfileDataManager", eventData.mStatus == ProfileDataManagerInternal::ETimerStatus::Idle,
            "Event '{}' is already running.", eventName);
        if (eventData.mStatus == ProfileDataManagerInternal::ETimerStatus::Idle)
        {
            eventData.mStatus = ProfileDataManagerInternal::ETimerStatus::Running;
        }
    }
    void ProfileDataManager::ReportEndEvent(const RHI::RhiCommandList* cmdList, const String& eventName)
    {
        auto  curTimerId = mInternalData->mAvailableTimerId[eventName];
        auto& eventTimer = mInternalData->mEventTimers[eventName][curTimerId];
        IF_LOG_ASSERTION(
            "ProfileDataManager", eventTimer != nullptr, "Event timer for '{}' is not initialized.", eventName);
        if (eventTimer)
        {
            eventTimer->Stop(cmdList);
        }
        auto& eventData = mInternalData->mEventDataMap[eventName];
        IF_LOG_ASSERTION("ProfileDataManager", eventData.mStatus == ProfileDataManagerInternal::ETimerStatus::Running,
            "Event '{}' is not running.", eventName);
        if (eventData.mStatus == ProfileDataManagerInternal::ETimerStatus::Running)
        {
            eventData.mStatus = ProfileDataManagerInternal::ETimerStatus::Idle;
        }
        mInternalData->mAvailableTimerId[eventName]++;
    }

    void ProfileDataManager::ReportHostBeginEvent(const String& eventName)
    {
        auto& eventData                            = mInternalData->mEventDataMap[eventName];
        mInternalData->mEventDeviceType[eventName] = ProfileDataManagerInternal::EProfileDevice::CPU;
        IF_LOG_ASSERTION("ProfileDataManager", eventData.mStatus == ProfileDataManagerInternal::ETimerStatus::Idle,
            "Event '{}' is already running.", eventName);
        if (eventData.mStatus == ProfileDataManagerInternal::ETimerStatus::Idle)
        {
            eventData.mStatus             = ProfileDataManagerInternal::ETimerStatus::Running;
            mInternalData->mHostTimestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        }
    }

    void ProfileDataManager::ReportHostEndEvent(const String& eventName)
    {
        auto& eventData = mInternalData->mEventDataMap[eventName];
        IF_LOG_ASSERTION("ProfileDataManager", eventData.mStatus == ProfileDataManagerInternal::ETimerStatus::Running,
            "Event '{}' is not running.", eventName);
        if (eventData.mStatus == ProfileDataManagerInternal::ETimerStatus::Running)
        {
            auto endTimestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            f32  durationMs   = static_cast<f32>((endTimestamp - mInternalData->mHostTimestamp) * 1e-6);
            ReportAccumulateEvent(eventName, durationMs);
            eventData.mStatus = ProfileDataManagerInternal::ETimerStatus::Idle;
        }
    }

    Vec<ProfileBriefReport> ProfileDataManager::GetBriefReport() const
    {
        Vec<ProfileBriefReport> reports;
        reports.reserve(mInternalData->mEventDataMap.size());

        for (const auto& [eventName, data] : mInternalData->mEventDataMap)
        {
            if (data.mCount == 0)
                continue; // Skip empty events

            ProfileBriefReport report;
            auto               deviceType = mInternalData->mEventDeviceType.at(eventName);
            if (deviceType == ProfileDataManagerInternal::EProfileDevice::GPU)
                report.mEventName = "[Device] " + eventName;
            else if (deviceType == ProfileDataManagerInternal::EProfileDevice::CPU)
                report.mEventName = "[Host] " + eventName;
            report.mAvgDurationMs = data.mTotalDurationMs / static_cast<f32>(data.mCount);
            report.mMaxDurationMs = data.mMaxDurationMs;
            report.mMinDurationMs = data.mMinDurationMs;
            reports.emplace_back(std::move(report));
        }
        return reports;
    }

    void ProfileDataManager::SetMaxFramesToKeep(u32 maxFrames)
    {
        mInternalData->mMaxFramesToKeep = maxFrames;

        // Clean up existing data to match new limit
        for (auto& [eventName, eventData] : mInternalData->mEventDataMap)
        {
            while (!eventData.mRecentEntries.empty()
                && (mInternalData->mCurrentFrameIndex - eventData.mRecentEntries.front().mFrameIndex) >= maxFrames)
            {
                eventData.mRecentEntries.pop_front();
            }
            eventData.RecalculateStats();
        }
    }

    u32 ProfileDataManager::GetMaxFramesToKeep() const { return mInternalData->mMaxFramesToKeep; }

} // namespace Ifrit::Runtime