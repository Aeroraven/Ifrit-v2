#include "ifrit/core/algo/Guid.h"

#define UUID_SYSTEM_GENERATOR
#include "stduuid/stduuid.h"
#include "ifrit/core/typing/EnumReflection.h"

namespace Ifrit
{
    IFRIT_APIDECL void GUID::GenerateString()
    {
        if (m_CvtString.empty())
        {
            m_CvtString = uuids::to_string(uuids::uuid{ m_Data });
        }
    }

    IFRIT_APIDECL String GUID::ToString() const { return m_CvtString; }

    IFRIT_APIDECL GUID   GUID::Generate()
    {
        uuids::uuid   idx    = uuids::uuid_system_generator{}();
        auto          uuidU8 = idx.as_bytes();
        Array<u8, 16> data;
        for (auto i = 0; i < 16; i++)
        {
            data[i] = GetEnumUnderlyingValue(uuidU8[i]);
        }
        GUID guid(data);
        guid.m_Generated = true;
        guid.GenerateString();
        return guid;
    }

} // namespace Ifrit