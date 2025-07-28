
#pragma once
#include "ifrit/runtime/base/ActorBehavior.h"

using namespace Ifrit::Runtime;

namespace Ifrit
{
    inline constexpr f32 kDefaultTimestep = 100.0f;
    static f32           sTimestep        = 1.0f / kDefaultTimestep;

    class IF_CLASS() MPMTiming : public ActorBehavior
    {
        using ActorBehavior::ActorBehavior;

    public:
        IF_PROPERTY()
        f32 mInvTimestep = kDefaultTimestep;

    private:
        typedef ActorBehavior Super;
        f32                   mTimestep = 1.0f / mInvTimestep;

    public:
        void SetupProperties() override
        {
            AddProperty<f32, EPropertyEditorType::Range>("Time Interval", mInvTimestep, 60.0f, 2000.0f, 0.001f);
        }
        void OnUpdate() override { sTimestep = 1.0f / mInvTimestep; }
    };
} // namespace Ifrit