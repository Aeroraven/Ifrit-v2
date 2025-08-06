#include "ifrit/runtime/renderer/profiling/ProfileDataManager.h"
#include <deque>

namespace Ifrit::Runtime
{
    struct ProfileDataManagerInternal
    {
        struct ProfileEventEntry
        {
            f32 mDurationMs;
            u32 mFrameIndex;
        };

        struct ProfileEventData
        {
            std::deque<ProfileEventEntry> mRecentEntries;
            f32                           mTotalDurationMs = 0.0f;
            f32                           mMaxDurationMs   = 0.0f;
            f32                           mMinDurationMs   = std::numeric_limits<f32>::max();
            u32                           mCount           = 0;

            void                          AddEntry(f32 duration, u32 frameIndex, u32 maxFrames)
            {
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
                    mTotalDurationMs += entry.mDurationMs;
                    mMaxDurationMs = std::max(mMaxDurationMs, entry.mDurationMs);
                    mMinDurationMs = std::min(mMinDurationMs, entry.mDurationMs);
                }
            }
        };

        HashMap<String, ProfileEventData> mEventDataMap;
        u32                               mCurrentFrameIndex = 0;
        u32                               mMaxFramesToKeep   = 60; // Default: keep last 60 frames
    };

    ProfileDataManager::ProfileDataManager() { mInternalData = new ProfileDataManagerInternal(); }

    ProfileDataManager::~ProfileDataManager() { delete mInternalData; }

    void ProfileDataManager::FrameProceed()
    {
        mInternalData->mCurrentFrameIndex++;

        // Optional: Clean up old entries for all events
        for (auto& [eventName, eventData] : mInternalData->mEventDataMap)
        {
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

    Vec<ProfileBriefReport> ProfileDataManager::GetBriefReport() const
    {
        Vec<ProfileBriefReport> reports;
        reports.reserve(mInternalData->mEventDataMap.size());

        for (const auto& [eventName, data] : mInternalData->mEventDataMap)
        {
            if (data.mCount == 0)
                continue; // Skip empty events

            ProfileBriefReport report;
            report.mEventName     = eventName;
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