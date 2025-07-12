
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

#include "ifrit/core/console/ConsoleObjectManager.h"
#include "ifrit/core/logging/Logging.h"

namespace Ifrit
{
    struct FConsoleVariableRegistryData
    {
        HashMap<String, Owner<IFConsoleVariableRegistryEntry>> m_CVars;
    };

    IFRIT_APIDECL FConsoleVariableRegistry::~FConsoleVariableRegistry() { delete m_Data; }

    IFRIT_APIDECL FConsoleVariableRegistry::FConsoleVariableRegistry() : m_Data(new FConsoleVariableRegistryData()) {}

    IFRIT_APIDECL void FConsoleVariableRegistry::RegisterVariable(
        const char* name, Owner<IFConsoleVariableRegistryEntry>& ptr)
    {
        m_Data->m_CVars[name] = std::move(ptr);
        iDebug("ConsoleVariableRegistry: Registered console variable: {}", name);
    }

    IFRIT_APIDECL void FConsoleVariableRegistry::UnregisterVariable(const char* name) { m_Data->m_CVars.erase(name); }

    IFRIT_APIDECL IFConsoleVariableRegistryEntry* FConsoleVariableRegistry::FindVariableGeneric(const char* name) const
    {
        auto it = m_Data->m_CVars.find(name);
        if (it != m_Data->m_CVars.end())
        {
            return it->second.get();
        }
        return nullptr;
    }

    IFRIT_APIDECL FConsoleVariableRegistry* GetFConsoleVariableRegistry()
    {
        static FConsoleVariableRegistry registry;
        return &registry;
    }
} // namespace Ifrit