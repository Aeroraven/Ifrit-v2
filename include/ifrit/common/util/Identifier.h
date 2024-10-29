#pragma once
#define UUID_SYSTEM_GENERATOR
#include "stduuid/stduuid.h"

namespace Ifrit::Common::Utility {

inline void generateUuid(std::string &id) {
  uuids::uuid idx = uuids::uuid_system_generator{}();
  id = uuids::to_string(idx);
}
} // namespace Ifrit::Common::Utility