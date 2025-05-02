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
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/forwarding/FwdBase.h"

namespace Ifrit::Runtime
{

    struct ShaderVariantDesc
    {
        String      m_Name;
        Vec<String> m_Defines;

        ShaderVariantDesc() = default;
        ShaderVariantDesc(const String& name, const Vec<String>& defines) : m_Name(name), m_Defines(defines) {}
        ShaderVariantDesc(const String& name) : m_Name(name) {}

        bool operator==(const ShaderVariantDesc& other) const
        {
            return m_Name == other.m_Name && m_Defines == other.m_Defines;
        }
    };

} // namespace Ifrit::Runtime