#pragma once
#include "ifrit/core/typing/EnumReflection.h"

// https://stackoverflow.com/questions/39969621/how-to-use-cereal-to-serialize-enum-types
#define IFRIT_ENUMCLASS_SERIALIZE(enumClass)                                                                       \
    namespace cereal                                                                                               \
    {                                                                                                              \
        template <class Archive> inline Ifrit::String save_minimal(Archive const&, enumClass x)                    \
        {                                                                                                          \
            return Ifrit::GetEnumName(x);                                                                          \
        }                                                                                                          \
                                                                                                                   \
        template <class Archive> inline void load_minimal(Archive const&, enumClass& x, Ifrit::String const& name) \
        {                                                                                                          \
            x = Ifrit::GetEnumFromName<enumClass>(name);                                                           \
        }                                                                                                          \
    }