
#pragma once
#include "ifrit/runtime/base/ActorBehavior.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/input/InputSystem.h"
#include "ifrit/runtime/physics/artemis/rigid/GPURigidCollider.h"
#include "ifrit/runtime/geometry/preset/Circle2D.h"
#include "ifrit/runtime/geometry/preset/Square2D.h"
#include "ifrit/runtime/material/SyaroDefaultGBufEmitter.h"
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/base/MeshComponent.h"
#include "ifrit/runtime/base/Transform.h"
#include "ifrit/runtime/scene/SceneManager.h"
using namespace Ifrit::Runtime;

namespace Ifrit
{
    enum class ERigidEmitShape : u8
    {
        Sphere,
        Cuboid,
    };

    class IF_CLASS() RigidEmitter : public ActorBehavior
    {
        using ActorBehavior::ActorBehavior;

    public:
        IF_PROPERTY(Editable, UISlider = (min = 0.01, max = 100.0))
        f32 mMass = 1.145f;

        IF_PROPERTY(Editable, UISelect)
        ERigidEmitShape mShape = ERigidEmitShape::Cuboid;

    private:
        typedef ActorBehavior Super;
        int                   mEmitCount = 0;

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
                if (inputSystem->IsMouseButtonPressed(EInputMouseButton::Left))
                {

                    auto mouseX = inputSystem->GetMouseX() * 2.0f - 1.0f;
                    mouseX      = mouseX * aspect;
                    mouseX      = mouseX * 0.5f + 0.5f;
                    auto mouseY = inputSystem->GetMouseY();
                    if (mouseX < 0.0f || mouseX > 1.0f || mouseY < 0.0f || mouseY > 1.0f)
                    {
                        return;
                    }

                    auto circleMesh = assetRegistry->GetAssetByName<Geometry::Circle2DAsset>("Circle2DAsset");
                    if (!circleMesh)
                    {
                        circleMesh = assetRegistry->CreateAsset<Geometry::Circle2DAsset>("Circle2DAsset", 0.05f, 32);
                    }
                    auto boxMesh = assetRegistry->GetAssetByName<Geometry::Square2DAsset>("Box2DAsset");
                    if (!boxMesh)
                    {
                        boxMesh = assetRegistry->CreateAsset<Geometry::Square2DAsset>("Box2DAsset", 0.1f, 0.1f);
                    }

                    auto material = assetRegistry->GetAssetByName<DefaultMaterialAsset>("DefaultMaterialAsset");
                    if (!material)
                    {
                        material = assetRegistry->CreateAsset<DefaultMaterialAsset>("DefaultMaterialAsset");
                    }
                    auto node      = activeScene->GetRootNode()->GetChildren()[0];
                    auto rigid     = node->AddGameObject("RigidCollider" + std::to_string(mEmitCount++));
                    auto rigidMesh = rigid->AddComponent<MeshFilter>();
                    if (mShape == ERigidEmitShape::Cuboid)
                    {
                        rigidMesh->SetMeshSource(boxMesh);
                    }
                    else
                    {
                        rigidMesh->SetMeshSource(circleMesh);
                    }
                    auto rigidRenderer = rigid->AddComponent<MeshRenderer>();
                    rigidRenderer->SetMaterialSource(material);
                    auto rigidTransform = rigid->GetComponent<Transform>();
                    rigidTransform->SetPosition({ mouseX, 1.0f - mouseY, 0.0f });
                    rigidTransform->SetDevice(TransformUpdateDevice::GPU);
                    auto rigidCollider = rigid->AddComponent<Artemis::GPURigidCollider>();
                    rigidCollider->SetRadius(0.05f);
                    rigidCollider->mRigidMass = mMass;
                    if (mShape == ERigidEmitShape::Sphere)
                    {
                        rigidCollider->SetColliderType(Artemis::GPURigidColliderType::Sphere);
                    }
                    else if (mShape == ERigidEmitShape::Cuboid)
                    {
                        rigidCollider->SetColliderType(Artemis::GPURigidColliderType::Box);
                    }

                    rigidCollider->SetEnable(true);
                }
            }
        }
    };
} // namespace Ifrit
