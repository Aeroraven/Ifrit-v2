#pragma once
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/unordered_map.hpp>
#include <sstream>
#include <string>

#define IFRIT_STRUCT_SERIALIZE(...)                                            \
  template <class Archive> void serialize(Archive &ar) { ar(__VA_ARGS__); }

namespace Ifrit::Common::Serialization {
template <class T> void serialize(T &src, std::string &dst) {
  std::ostringstream oss;
  {
    cereal::JSONOutputArchive ar(oss);
    ar(src);
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