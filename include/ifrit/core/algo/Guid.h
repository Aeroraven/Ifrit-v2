

#pragma once

#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/reflection/Fwd.h"

namespace Ifrit
{
    class IFRIT_CORE_API GUID
    {
    private:
        Array<u8, 16> m_Data;
        String        m_CvtString = "";
        bool          m_Generated = false;

    public:
        GUID() noexcept = default;
        GUID(const Array<u8, 16>& data) noexcept : m_Data(data), m_Generated(true) { GenerateString(); }
        GUID(const GUID& other) : m_Data(other.m_Data), m_CvtString(other.m_CvtString), m_Generated(other.m_Generated)
        {
        }
        GUID(GUID&& other) noexcept
            : m_Data(std::move(other.m_Data)), m_CvtString(std::move(other.m_CvtString)), m_Generated(other.m_Generated)
        {
            other.m_Generated = false;
        }

        inline GUID& operator=(const GUID& other)
        {
            if (this != &other)
            {
                m_Data      = other.m_Data;
                m_CvtString = other.m_CvtString;
                m_Generated = other.m_Generated;
            }
            return *this;
        }

        inline bool operator==(const GUID& other) const { return m_Data == other.m_Data; }

    private:
        void GenerateString();

    public:
        String      ToString() const;
        static GUID Generate();

        void        DoSerialize(Reflection::Archive* archive) const;
        void        DoDeserialize(Reflection::Archive* archive);
    };

} // namespace Ifrit

namespace std
{
    template <> struct hash<Ifrit::GUID>
    {
        size_t operator()(const Ifrit::GUID& guid) const noexcept { return std::hash<std::string>()(guid.ToString()); }
    };
} // namespace std