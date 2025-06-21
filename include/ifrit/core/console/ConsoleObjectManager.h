
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
#include "ifrit/core/typing/Util.h"
#include "ifrit/core/typing/Traits.h"

namespace Ifrit
{
    enum ConsoleVariableFlag : u8
    {
        CVF_Default          = 0x00,
        CVF_RenderThreadSafe = 0x01, // not used now
    };

    class IFRIT_CORE_API IConsoleVariableRegistryEntry
    {
    };

    template <typename T IF_REQUIRES(TypeIsAnyOf_v<T, i32, u32, f32, String>)>
    class IFRIT_CORE_API ConsoleVariableRegistryEntry : public IConsoleVariableRegistryEntry
    {
    public:
        ConsoleVariableRegistryEntry(T value, const char* description, u8 flags = CVF_Default)
            : Value(value), Description(description), Flags(flags)
        {
        }

        T    GetValue() const { return Value; }
        T*   GetValuePtr() { return &Value; }
        void SetValue(T value) { Value = value; }

    private:
        T      Value;
        String Description;
        u8     Flags;
    };

    template class IFRIT_CORE_API ConsoleVariableRegistryEntry<i32>;
    template class IFRIT_CORE_API ConsoleVariableRegistryEntry<u32>;
    template class IFRIT_CORE_API ConsoleVariableRegistryEntry<f32>;
    template class IFRIT_CORE_API ConsoleVariableRegistryEntry<String>;

    struct ConsoleVariableRegistryData;
    class IFRIT_CORE_API ConsoleVariableRegistry
    {
    private:
        ConsoleVariableRegistryData* m_Data;

    public:
        ~ConsoleVariableRegistry();
        ConsoleVariableRegistry();

        void                           RegisterVariable(const char* name, Owner<IConsoleVariableRegistryEntry>& ptr);
        void                           UnregisterVariable(const char* name);
        IConsoleVariableRegistryEntry* FindVariableGeneric(const char* name) const;

        template <typename T> ConsoleVariableRegistryEntry<T>* FindVariable(const char* name) const
        {
            return CheckedCast<ConsoleVariableRegistryEntry<T>>(FindVariableGeneric(name));
        }
    };

    IFRIT_CORE_API ConsoleVariableRegistry* GetConsoleVariableRegistry();

} // namespace Ifrit