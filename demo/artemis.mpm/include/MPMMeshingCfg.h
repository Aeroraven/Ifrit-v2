
#pragma once
#include "ifrit/runtime/base/ActorBehavior.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/physics/artemis/mpm/visualization/ProceduralMPMMesh.h"
#include "ifrit/runtime/base/MeshComponent.h"
using namespace Ifrit::Runtime;

namespace Ifrit
{

    class IF_CLASS() MPMMeshingCfg : public ActorBehavior
    {
        using ActorBehavior::ActorBehavior;

    public:
        IF_PROPERTY(Editable, UISlider = (min = 0.0f, max = 100.0f))
        f32 mIsoValue = 10.0f;

    private:
        typedef ActorBehavior Super;

    public:
        void OnUpdate() override
        {
            auto meshFilter = GetParent()->GetComponent<MeshFilter>();
            auto mesh       = meshFilter->GetMesh();
            if (mesh)
            {
                auto proc = ForcedCheckedCast<Artemis::ProceduralMPMMesh>(mesh);
                if (proc)
                {
                    proc->SetIsoValue(mIsoValue);
                }
            }
        }
    };
} // namespace Ifrit
