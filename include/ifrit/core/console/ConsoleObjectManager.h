
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
    enum EConsoleVariableFlag : u8
    {
        CVF_Default          = 0x00,
        CVF_RenderThreadSafe = 0x01, // not used now
        CVF_ReadOnly         = 0x02,
    };

    class IFRIT_CORE_API IFConsoleVariableRegistryEntry{};

    template <typename T>
        requires IConceptIsAnyOf<T, TTypeSet<i32, u32, f32, i64, u64, bool, String>>
    class IFRIT_CORE_API FConsoleVariableRegistryEntry : public IFConsoleVariableRegistryEntry
    {
    public:
        FConsoleVariableRegistryEntry(T value, const char* description, u8 flags = CVF_Default)
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

    template class IFRIT_CORE_API FConsoleVariableRegistryEntry<i32>;
    template class IFRIT_CORE_API FConsoleVariableRegistryEntry<u32>;
    template class IFRIT_CORE_API FConsoleVariableRegistryEntry<f32>;
    template class IFRIT_CORE_API FConsoleVariableRegistryEntry<i64>;
    template class IFRIT_CORE_API FConsoleVariableRegistryEntry<u64>;
    template class IFRIT_CORE_API FConsoleVariableRegistryEntry<String>;
    template class IFRIT_CORE_API FConsoleVariableRegistryEntry<bool>;

    struct FConsoleVariableRegistryData;
    class IFRIT_CORE_API FConsoleVariableRegistry
    {
    private:
        FConsoleVariableRegistryData* m_Data;

    public:
        ~FConsoleVariableRegistry();
        FConsoleVariableRegistry();

        void                            RegisterVariable(const char* name, Owner<IFConsoleVariableRegistryEntry>& ptr);
        void                            UnregisterVariable(const char* name);
        IFConsoleVariableRegistryEntry* FindVariableGeneric(const char* name) const;

        template <typename T> FConsoleVariableRegistryEntry<T>* FindVariable(const char* name) const
        {
            return CheckedCast<FConsoleVariableRegistryEntry<T>>(FindVariableGeneric(name));
        }
    };

    IFRIT_CORE_API FConsoleVariableRegistry* GetFConsoleVariableRegistry();

} // namespace Ifrit