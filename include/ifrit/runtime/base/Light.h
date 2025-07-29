
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
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/core/serialization/SerialEnumDefine.h"
#include "ifrit/core/reflection/ReflAttrs.h"

namespace Ifrit::Runtime
{

    enum class LightType
    {
        Directional
    };

    class IFRIT_APIDECL IF_CLASS() Light : public Component
    {
    public:
        IF_PROPERTY()
        LightType mType = LightType::Directional;

        IF_PROPERTY()
        bool AffectPbrSky = false;

        IF_PROPERTY()
        bool ShadowMap = false;

        IF_PROPERTY()
        u32 ShadowMapResolution = 512;

    public:
        using Component::Component;
        virtual ~Light() = default;

        // getters

        inline LightType GetType() const { return mType; }
        inline bool      GetAffectPbrSky() const { return AffectPbrSky; }
        inline bool      GetShadowMap() const { return ShadowMap; }
        inline u32       GetShadowMapResolution() const { return ShadowMapResolution; }

        // setters

        inline void      SetType(const LightType& type) { mType = type; }
        inline void      SetAffectPbrSky(bool affectPbrSky) { AffectPbrSky = affectPbrSky; }
        inline void      SetShadowMap(bool shadowMap) { ShadowMap = shadowMap; }
        inline void      SetShadowMapResolution(u32 resolution) { ShadowMapResolution = resolution; }
    };
} // namespace Ifrit::Runtime
