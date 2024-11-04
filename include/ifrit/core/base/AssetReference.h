#pragma once
#include "ifrit/common/serialization/SerialInterface.h"
#include "ifrit/common/util/ApiConv.h"
#include <memory>
#include <string>


namespace Ifrit::Core {
struct AssetReference {
  std::string m_fileId;
  std::string m_uuid;
  std::string m_name;
  bool m_usingAsset = false;
  IFRIT_STRUCT_SERIALIZE(m_fileId, m_uuid, m_name, m_usingAsset)
};

class IFRIT_APIDECL IAssetCompatible {
public:
  virtual void _polyHolderAsset() {}
};

class AssetReferenceContainer {
public:
  AssetReference m_assetReference;
  bool m_usingAsset = false;
  std::weak_ptr<IAssetCompatible> m_asset;

  IFRIT_STRUCT_SERIALIZE(m_assetReference, m_usingAsset)
};

} // namespace Ifrit::Core