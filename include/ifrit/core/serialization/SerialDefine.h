#pragma once
#define IFRIT_STRUCT_SERIALIZE(...) \
    template <class Archive> void serialize(Archive& ar) { ar(__VA_ARGS__); }

#define IFRIT_STRUCT_SERIALIZE_COND(cond, ...)           \
    template <class Archive> void serialize(Archive& ar) \
    {                                                    \
        ar(cond);                                        \
        if (cond)                                        \
        {                                                \
            ar(__VA_ARGS__);                             \
        }                                                \
    }
