
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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
#include "ifrit/runtime/base/Component.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Runtime::Ayanami
{
    // This holds some properties for the renderer
    class IFRIT_APIDECL AyanamiMeshMarker : public Component
    {
    private:
        u32 m_PlaceHolder;
        u32 m_TrivialMeshCardIndex = ~0u;

    public:
        AyanamiMeshMarker() {}
        AyanamiMeshMarker(Ref<GameObject> owner) : Component(owner) {}
        virtual ~AyanamiMeshMarker() = default;

        inline String Serialize() override { return ""; }
        inline void   Deserialize() override {}

    public:
        IF_FORCEINLINE u32  GetTrivialMeshCardIndex() const { return m_TrivialMeshCardIndex; }
        IF_FORCEINLINE void SetTrivialMeshCardIndex(u32 index) { m_TrivialMeshCardIndex = index; }

        IFRIT_COMPONENT_SERIALIZE(m_PlaceHolder);
    };

} // namespace Ifrit::Runtime::Ayanami

IFRIT_COMPONENT_REGISTER(Ifrit::Runtime::Ayanami::AyanamiMeshMarker)
