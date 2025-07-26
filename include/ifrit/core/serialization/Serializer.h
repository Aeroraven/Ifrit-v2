
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
// #include <cereal/archives/binary.hpp>
// #include <cereal/archives/json.hpp>
// #include <cereal/types/queue.hpp>
// #include <cereal/cereal.hpp>
// #include <cereal/types/map.hpp>
// #include <cereal/types/polymorphic.hpp>
// #include <cereal/types/string.hpp>
// #include <cereal/types/unordered_map.hpp>
// #include <cereal/types/vector.hpp>
// #include <cereal/types/array.hpp>
// #include <sstream>
#include "ifrit/core/serialization/SerialDefine.h"
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/logging/Logging.h"

namespace Ifrit::Serialization
{

    IFRIT_CORE_API void SerializationErrorReport(const String& str);

    enum class ESerializationFormat : u8
    {
        Binary,
        Json,
    };

    template <ESerializationFormat Fmt> struct TSerializerTraits;
    template <> struct TSerializerTraits<ESerializationFormat::Binary>
    {
        using OutputArchive                     = std::void_t<int>;
        using InputArchive                      = std::void_t<int>;
        static constexpr const char* FormatName = "Binary";
    };
    template <> struct TSerializerTraits<ESerializationFormat::Json>
    {

        using OutputArchive                     = std::void_t<int>;
        using InputArchive                      = std::void_t<int>;
        static constexpr const char* FormatName = "Json";
    };

    template <ESerializationFormat Fmt, class T> String Serialize(const T& src)
    {
        //std::ostringstream oss;
        IF_LOG_REMOVED_FEATURE("Serialization", "Static serialization (cereal)");
        return "";
    }

    template <ESerializationFormat Fmt, class T> void Deserialize(const String& src, T& dst)
    {

        std::istringstream iss(src);
        IF_LOG_REMOVED_FEATURE("Serialization", "Static serialization (cereal)");
    }

} // namespace Ifrit::Serialization