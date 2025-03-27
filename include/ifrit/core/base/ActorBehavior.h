
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "Component.h"

namespace Ifrit::Core
{
    struct ActorBehaviorAttribute
    {
        u32 m_placeHolder;

        IFRIT_STRUCT_SERIALIZE(m_placeHolder);
    };

    class IFRIT_APIDECL ActorBehavior : public Component, public AttributeOwner<ActorBehaviorAttribute>
    {
    private:
    public:
        ActorBehavior(){};
        ActorBehavior(Ref<SceneObject> parent)
            : Component(parent), AttributeOwner<ActorBehaviorAttribute>() {}

        String Serialize() override { return SerializeAttribute(); }
        void   Deserialize() override { DeserializeAttribute(); }

        IFRIT_COMPONENT_SERIALIZE(m_attributes);
    };
} // namespace Ifrit::Core

IFRIT_COMPONENT_REGISTER(Ifrit::Core::ActorBehavior);