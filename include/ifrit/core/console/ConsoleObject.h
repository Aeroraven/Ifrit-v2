
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
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"

#include "ifrit/core/console/ConsoleObjectManager.h"

namespace Ifrit
{
    template <typename T> class ConsoleVariable
    {
    private:
        ConsoleVariableRegistryEntry<T>* m_Entry;

    public:
        ConsoleVariable(const char* name, T value, const char* description, u8 flags = CVF_Default)
        {
            Uref<IConsoleVariableRegistryEntry> entry =
                std::make_unique<ConsoleVariableRegistryEntry<T>>(value, description, flags);
            m_Entry = static_cast<ConsoleVariableRegistryEntry<T>*>(entry.get());
            GetConsoleVariableRegistry()->RegisterVariable(name, entry);
        }

        T    GetValue() const { return m_Entry->GetValue(); }
        void SetValue(T value) { m_Entry->SetValue(value); }
    };
} // namespace Ifrit