
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
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/math/VectorDefs.h"
#include "ifrit/core/serialization/MathTypeSerialization.h"
#include "ifrit/core/serialization/SerialInterface.h"
#include "ifrit/core/typing/Util.h"

namespace Ifrit::Runtime
{

    enum class LightType
    {
        Directional
    };
    struct LightData
    {
        LightType m_type                = LightType::Directional;
        bool      m_affectPbrSky        = false;
        bool      m_shadowMap           = false;
        u32       m_shadowMapResolution = 512;
        IFRIT_STRUCT_SERIALIZE(m_type, m_affectPbrSky, m_shadowMap);
    };

    class IFRIT_APIDECL Light : public Component, public AttributeOwner<LightData>
    {
    public:
        Light(){};
        Light(Ref<GameObject> owner) : Component(owner), AttributeOwner() {}
        virtual ~Light() = default;
        inline String    Serialize() override { return SerializeAttribute(); }
        inline void      Deserialize() override { DeserializeAttribute(); }

        // getters
        inline LightType GetType() const { return m_attributes.m_type; }
        inline bool      GetAffectPbrSky() const { return m_attributes.m_affectPbrSky; }
        inline bool      GetShadowMap() const { return m_attributes.m_shadowMap; }
        inline u32       GetShadowMapResolution() const { return m_attributes.m_shadowMapResolution; }

        // setters
        inline void      SetType(const LightType& type) { m_attributes.m_type = type; }
        inline void      SetAffectPbrSky(bool affectPbrSky) { m_attributes.m_affectPbrSky = affectPbrSky; }
        inline void      SetShadowMap(bool shadowMap) { m_attributes.m_shadowMap = shadowMap; }
        inline void      SetShadowMapResolution(u32 resolution) { m_attributes.m_shadowMapResolution = resolution; }
    };
} // namespace Ifrit::Runtime

IFRIT_COMPONENT_REGISTER(Ifrit::Runtime::Light)
IFRIT_ENUMCLASS_SERIALIZE(Ifrit::Runtime::LightType)