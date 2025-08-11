
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
        typedef ActorBehavior Super;

    public:
        IF_PROPERTY(Editable, UISelect)
        EInteractionMode mInteractionMode = EInteractionMode::Pull;

        IF_PROPERTY(Editable, UISlider = (min = 0.001f, max = 0.5f))
        float mRadius = 0.1f;

        IF_PROPERTY(Editable, UISlider = (min = 0.0001f, max = 0.005f))
        float mActivation = 0.001f;

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
                    auto rawX   = inputSystem->GetMouseX();
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

                        mLastMouseDirWS = GetMouseDirWS(mainCamera, rawX, mouseY);
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
                            mpmSimulator->SetMouseDirection(mLastMouseDirWS, mainCamera->GetCameraPosition());
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
                            mpmSimulator->SetMouseDirection(mLastMouseDirWS, Vector3f(114.0f, 514.0f, 0.0f));
                        }
                    }
                }
            }
        }

    private:
        Vector3f mLastMouseDirWS = Vector3f(0.0f, 0.0f, 1.0f);
        float    mLastMouseX     = 0.0f;
        float    mLastMouseY     = 0.0f;
        bool     mLastMouseDown  = false;

    private:
        Vector3f GetMouseDirWS(Camera* camera, float mouseX01, float mouseY01)
        {
            using namespace Ifrit::Math;
            if (!camera)
                return Vector3f(0.0f, 0.0f, 1.0f);
            float    xNDC = mouseX01 * 2.0f - 1.0f;
            float    yNDC = (mouseY01) * 2.0f - 1.0f;
            auto     V    = camera->GetWorldToCameraMatrix();
            auto     P    = camera->GetProjectionMatrix();
            auto     Inv  = Math::Inverse(Math::MatMul(P, V));

            // For left-handed z-front (DirectX style) near=0, far=1 in clip/NDC.
            Vector4f nearH(xNDC, yNDC, 0.1f, 1.0f);
            Vector4f farH(xNDC, yNDC, 0.9f, 1.0f);

            nearH = Math::MatMul(Inv, nearH);
            farH  = Math::MatMul(Inv, farH);

            // Perspective divide
            if (nearH.w != 0.0f)
                nearH = nearH * (1.0f / nearH.w);
            if (farH.w != 0.0f)
                farH = farH * (1.0f / farH.w);

            Vector3f nearWS(nearH.x, nearH.y, nearH.z);
            Vector3f farWS(farH.x, farH.y, farH.z);

            Vector3f dir = Math::Normalize(farWS - nearWS);

            // Optional log:
            IF_LOG_INFO("MPMMouseInteractor", "Ray dir: {}, {}, {}", dir.x, dir.y, dir.z);

            return dir;
        }
    };
} // namespace Ifrit
