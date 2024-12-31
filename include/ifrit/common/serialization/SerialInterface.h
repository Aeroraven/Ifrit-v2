
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
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <sstream>
#include <string>

#define IFRIT_STRUCT_SERIALIZE(...)                                            \
  template <class Archive> void serialize(Archive &ar) { ar(__VA_ARGS__); }

#define IFRIT_STRUCT_SERIALIZE_COND(cond, ...)                                 \
  template <class Archive> void serialize(Archive &ar) {                       \
    ar(cond);                                                                  \
    if (cond) {                                                                \
      ar(__VA_ARGS__);                                                         \
    }                                                                          \
  }

#define IFRIT_DERIVED_REGISTER(x) CEREAL_REGISTER_TYPE(x)
#define IFRIT_INHERIT_REGISTER(base, derived)                                  \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(base, derived)

namespace Ifrit::Common::Serialization {
template <class T> void serialize(T &src, std::string &dst) {
  std::ostringstream oss;
  {
    try {
      cereal::JSONOutputArchive ar(oss);
      ar(src);
    } catch (const std::exception &e) {
      printf("Error: %s\n", e.what());
      std::abort();
    }
  }
  dst = oss.str();
}

template <class T> void serializeBinary(T &src, std::string &dst) {
  std::ostringstream oss;
  {
    try {
      cereal::BinaryOutputArchive ar(oss);
      ar(src);
    } catch (const std::exception &e) {
      printf("Error: %s\n", e.what());
      std::abort();
    }
  }
  dst = oss.str();
}

template <class T> void deserialize(const std::string &src, T &dst) {
  std::istringstream iss(src);
  {
    cereal::JSONInputArchive ar(iss);
    ar(dst);
  }
}

template <class T> void deserializeBinary(const std::string &src, T &dst) {
  std::istringstream iss(src);
  {
    cereal::BinaryInputArchive ar(iss);
    ar(dst);
  }
}

#define IFRIT_ENUMCLASS_SERIALIZE(enumClass)                                   \
  template <class Archive> void serialize(Archive &ar, enumClass &x) {         \
    ar(cereal::make_nvp(#enumClass, x));                                       \
  }

} // namespace Ifrit::Common::Serialization