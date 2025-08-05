
#pragma once
#include "ifrit/runtime/base/ActorBehavior.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/physics/artemis/ArtemisController.h"
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

        IF_PROPERTY(Editable, UISlider = (min = 0.0, max = 1000.0))
        f32 mSleep = 0.0f;

    private:
        typedef ActorBehavior Super;
        f32                   mTimestep = 1.0f / mInvTimestep;

    public:
        void OnUpdate() override
        {
            sTimestep              = 1.0f / mInvTimestep;
            auto artemisController = GetActiveApplication()->GetSubsystem<Artemis::ArtemisController>();
            if (artemisController)
            {
                artemisController->SetTimestep(sTimestep);
            }
            if (mSleep > 0.0f)
            {
                Sleep(mSleep);
            }
        }
    };
} // namespace Ifrit
