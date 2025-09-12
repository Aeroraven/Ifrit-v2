#pragma once

#include "ifrit/runtime/base/Base.h"
#include "ifrit/rhi/common/RhiForwardingTypes.h"
#include "ifrit/runtime/physics/artemis/ArtemisSceneData.h"
#include "ifrit/runtime/forwarding/FwdScene.h"

namespace Ifrit::Runtime::Artemis
{
    IFRIT_RUNTIME_API void CollectPhysicsSceneData(Scene* scene, RHI::RhiBackend* rhi, u32 frameId);
}
