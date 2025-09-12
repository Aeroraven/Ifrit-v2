#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/runtime/base/Base.h"
#include "ifrit/core/reflection/ReflAttrs.h"
#include "ifrit/runtime/base/Component.h"
namespace Ifrit::Runtime
{
    struct IF_CLASS() TempPrefabSerializationData
    {
    public:
        IF_PROPERTY()
        Owner<GameObject> mGameObject;

        IF_PROPERTY()
        Vec<Owner<Component>> mComponents;
    };

} // namespace Ifrit::Runtime