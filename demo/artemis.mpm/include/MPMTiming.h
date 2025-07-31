
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
        IF_PROPERTY(Editable, UISlider = (min = 60.0, max = 1000.0))
        f32 mInvTimestep = kDefaultTimestep;

    private:
        typedef ActorBehavior Super;
        f32                   mTimestep = 1.0f / mInvTimestep;

    public:
        void OnUpdate() override { sTimestep = 1.0f / mInvTimestep; }
    };
} // namespace Ifrit
