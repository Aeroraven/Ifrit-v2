
#pragma once
#include "ifrit/runtime/base/ActorBehavior.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/input/InputSystem.h"
#include "ifrit/runtime/physics/artemis/rigid/GPURigidCollider.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMSimulator.h"
#include "ifrit/runtime/physics/artemis/ArtemisController.h"
#include "ifrit/runtime/geometry/preset/Circle2D.h"
#include "ifrit/runtime/material/SyaroDefaultGBufEmitter.h"
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/base/MeshComponent.h"
#include "ifrit/runtime/base/Transform.h"
#include "ifrit/runtime/scene/SceneManager.h"
using namespace Ifrit::Runtime;

namespace Ifrit
{
    enum class EInteractionMode : u8
    {
        Push,
        Pull,
        Drain
    };

    class IF_CLASS() MPMMouseInteractor : public ActorBehavior
    {
        using ActorBehavior::ActorBehavior;

    public:
        IF_PROPERTY(Editable, UISelect)
        EInteractionMode mInteractionMode = EInteractionMode::Pull;

        IF_PROPERTY(Editable, UISlider = (min = 0.001f, max = 0.5f))
        float mRadius = 0.1f;

        IF_PROPERTY(Editable, UISlider = (min = 0.0001f, max = 0.005f))
        float mActivation = 0.001f;

    private:
        typedef ActorBehavior Super;

        float                 mLastMouseX    = 0.0f;
        float                 mLastMouseY    = 0.0f;
        bool                  mLastMouseDown = false;

    public:
        void OnUpdate() override
        {

            auto inputSystem   = GetActiveApplication()->GetSubsystem<InputSystem>();
            auto assetRegistry = GetActiveApplication()->GetAssetRegistry();
            auto activeScene   = GetActiveApplication()->GetSceneManager()->GetActiveScene();
            auto aspect        = 1.0f;
            auto mainCamera    = activeScene->GetMainCamera();
            if (mainCamera)
            {
                aspect = mainCamera->GetAspect();
            }
            if (inputSystem)
            {
                if (inputSystem->IsMouseButtonHold(EInputMouseButton::Left))
                {
                    auto mouseX = inputSystem->GetMouseX() * 2.0f - 1.0f;
                    mouseX      = mouseX * aspect;
                    mouseX      = mouseX * 0.5f + 0.5f;
                    auto mouseY = inputSystem->GetMouseY();
                    if (mouseX < 0.0f || mouseX > 1.0f || mouseY < 0.0f || mouseY > 1.0f)
                    {
                        return;
                    }
                    auto deltaX = mouseX - mLastMouseX;
                    auto deltaY = mouseY - mLastMouseY;
                    mLastMouseX = mouseX;
                    mLastMouseY = mouseY;
                    if (!mLastMouseDown)
                    {
                        deltaX         = 0.0f;
                        deltaY         = 0.0f;
                        mLastMouseDown = true;
                    }

                    auto artemisController = GetActiveApplication()->GetSubsystem<Artemis::ArtemisController>();
                    if (artemisController)
                    {
                        auto mpmSimulator = ForcedCheckedCast<Artemis::MPMSimulator>(
                            artemisController->GetPresetSolver(Artemis::EPresetArtemisSimulator::MPM));
                        if (mpmSimulator)
                        {
                            mpmSimulator->SetMousePosition(mouseX, 1.0f - mouseY);
                            mpmSimulator->SetMouseVelocity(deltaX, -deltaY);
                            mpmSimulator->SetMousePushMode(mInteractionMode == EInteractionMode::Push);
                            mpmSimulator->SetMouseRadAct(mActivation, mRadius);
                        }
                    }
                }
                else
                {
                    mLastMouseDown         = false;
                    auto artemisController = GetActiveApplication()->GetSubsystem<Artemis::ArtemisController>();
                    if (artemisController)
                    {
                        auto mpmSimulator = ForcedCheckedCast<Artemis::MPMSimulator>(
                            artemisController->GetPresetSolver(Artemis::EPresetArtemisSimulator::MPM));
                        if (mpmSimulator)
                        {
                            mpmSimulator->SetMousePosition(1145.0f, 1919.0f);
                            mpmSimulator->SetMouseVelocity(0.0f, 0.0f);
                            mpmSimulator->SetMousePushMode(false);
                            mpmSimulator->SetMouseRadAct(mActivation, mRadius);
                        }
                    }
                }
            }
        }
    };
} // namespace Ifrit
