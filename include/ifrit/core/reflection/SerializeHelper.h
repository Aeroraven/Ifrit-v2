
#pragma once
#include "ifrit/core/reflection/Serializer.h"
#include "ifrit/core/reflection/archive/JSONArchive.h"

namespace Ifrit::Reflection
{
    template <typename T> inline void DeserializeFromJSON(T& obj, const String& jsonData)
    {
        JSONArchive archive;
        archive.LoadFromString(jsonData);
        archive.SetState(ESerializationState::Reading);
        InvokeDeserialize(obj, &archive);
    }

    template <typename T> inline String SerializeToJSON(const T& obj)
    {
        JSONArchive archive;
        archive.SetState(ESerializationState::Writing);
        InvokeSerialize(const_cast<T&>(obj), &archive);
        return archive.GetResult();
    }
} // namespace Ifrit::Reflection