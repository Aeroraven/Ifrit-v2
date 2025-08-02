#include "ifrit/core/algo/Guid.h"
#include "ifrit/core/reflection/Archive.h"
namespace Ifrit
{
    void GUID::DoSerialize(Reflection::Archive* archive) const
    {
        archive->BeginObject("GUID");
        archive->BeginArray("Data");
        for (const auto& byte : m_Data)
        {
            archive->Serialize(byte);
        }
        archive->EndArray();
        archive->BeginObject("CvtString");
        archive->Serialize(m_CvtString);
        archive->EndObject();
        archive->BeginObject("Generated");
        archive->Serialize(m_Generated);
        archive->EndObject();
        archive->EndObject();
    }
    void GUID::DoDeserialize(Reflection::Archive* archive)
    {
        archive->BeginObject("GUID");
        archive->BeginArray("Data");
        for (auto& byte : m_Data)
        {
            archive->Serialize(byte);
            archive->NextArrayElement();
        }
        archive->EndArray();
        archive->BeginObject("CvtString");
        archive->Serialize(m_CvtString);
        archive->EndObject();
        archive->BeginObject("Generated");
        archive->Serialize(m_Generated);
        archive->EndObject();
        archive->EndObject();
    }
} // namespace Ifrit
