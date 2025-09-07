#include "ifrit/core/reflection/RttiIdentifier.h"
#include <string>
namespace Ifrit::Reflection
{
    IFRIT_CORE_API u64 GetTypeIDHash(const std::type_info& typeInfo)
    {
        String            typeName = typeInfo.name();
        std::hash<String> hasher;
        u64               hash = hasher(typeName);
        return hash;
    }
} // namespace Ifrit::Reflection