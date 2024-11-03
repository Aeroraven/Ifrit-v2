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

template <class T> void deserialize(const std::string &src, T &dst) {
  std::istringstream iss(src);
  {
    cereal::JSONInputArchive ar(iss);
    ar(dst);
  }
}

} // namespace Ifrit::Common::Serialization