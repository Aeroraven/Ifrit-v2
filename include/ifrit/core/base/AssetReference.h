#pragma once
#include "ifrit/common/serialization/SerialInterface.h"
#include <string>

namespace Ifrit::Core {
struct AssetReference {
  std::string m_fileId;
  std::string m_uuid;
  std::string m_name;

  IFRIT_STRUCT_SERIALIZE(m_fileId, m_uuid, m_name)
};
} // namespace Ifrit::Core