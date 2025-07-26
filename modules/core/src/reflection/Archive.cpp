#include "ifrit/core/reflection/Archive.h"
#include "ifrit/core/logging/Logging.h"
#include <iostream>
#include "json/json.h"
#include <stack>
namespace Ifrit::Reflection
{
    enum class ETopLevelType : u8
    {
        Object,
        Array,
    };

    struct TrivialArchivePrivate
    {
        nlohmann::json*             JsonData;
        std::stack<nlohmann::json*> ObjectStack;
        std::stack<ETopLevelType>   TopLevelStack;
    };

    class TrivialArchiveHelper
    {
    public:
        static void AppendObjectToJson(TrivialArchivePrivate* privateData, const String& name)
        {
            // if toplevel is object, create a new object
            if (privateData->TopLevelStack.top() == ETopLevelType::Object)
            {
                (*privateData->ObjectStack.top())[name] = nlohmann::json::object();
                privateData->ObjectStack.push(&(*privateData->ObjectStack.top())[name]);
                privateData->TopLevelStack.push(ETopLevelType::Object);
            }
            else if (privateData->TopLevelStack.top() == ETopLevelType::Array)
            {
                (*privateData->ObjectStack.top()).push_back(nlohmann::json::object());
                privateData->ObjectStack.push(&(*privateData->ObjectStack.top()).back());
                privateData->TopLevelStack.push(ETopLevelType::Object);
            }
        }

        static void AppendArrayToJson(TrivialArchivePrivate* privateData, const String& name)
        {
            // if toplevel is object, create a new array
            if (privateData->TopLevelStack.top() == ETopLevelType::Object)
            {
                (*privateData->ObjectStack.top())[name] = nlohmann::json::array();
                privateData->ObjectStack.push(&(*privateData->ObjectStack.top())[name]);
                privateData->TopLevelStack.push(ETopLevelType::Array);
            }
            else if (privateData->TopLevelStack.top() == ETopLevelType::Array)
            {
                (*privateData->ObjectStack.top()).push_back(nlohmann::json::array());
                privateData->ObjectStack.push(&(*privateData->ObjectStack.top()).back());
                privateData->TopLevelStack.push(ETopLevelType::Array);
            }
        }

        template <typename T>
        static void AppendValueToJson(TrivialArchivePrivate* privateData, const T& value, String name = "")
        {
            if (privateData->TopLevelStack.top() == ETopLevelType::Object)
            {
                if (!name.empty())
                {
                    (*privateData->ObjectStack.top())[name] = value;
                }
                else
                {
                    (*privateData->ObjectStack.top()) = value;
                }
            }
            else if (privateData->TopLevelStack.top() == ETopLevelType::Array)
            {
                if (!name.empty())
                {
                    (*privateData->ObjectStack.top())[name].push_back(value);
                }
                else
                {
                    (*privateData->ObjectStack.top()).push_back(value);
                }
            }
        }
    };

    TrivialArchive::TrivialArchive() : PrivateData(new TrivialArchivePrivate())
    {
        PrivateData->JsonData = new nlohmann::json();
        PrivateData->ObjectStack.push(PrivateData->JsonData);
        PrivateData->TopLevelStack.push(ETopLevelType::Object);
    }
    TrivialArchive::~TrivialArchive()
    {
        delete PrivateData->JsonData;
        delete PrivateData;
    }

    void TrivialArchive::BeginObject(const String& name)
    {
        TrivialArchiveHelper::AppendObjectToJson(PrivateData, name);
    }
    void TrivialArchive::EndObject()
    {
        if (PrivateData->ObjectStack.size() > 1)
        {
            PrivateData->ObjectStack.pop();
            PrivateData->TopLevelStack.pop();
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "EndObject called without matching BeginObject");
        }
    }
    void TrivialArchive::BeginArray(const String& name) { TrivialArchiveHelper::AppendArrayToJson(PrivateData, name); }

    void TrivialArchive::EndArray()
    {
        if (PrivateData->ObjectStack.size() > 1)
        {
            PrivateData->ObjectStack.pop();
            PrivateData->TopLevelStack.pop();
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "EndArray called without matching BeginArray");
        }
    }

    void   TrivialArchive::Serialize(u8& value) { TrivialArchiveHelper::AppendValueToJson(PrivateData, value); }
    void   TrivialArchive::Serialize(u16& value) { TrivialArchiveHelper::AppendValueToJson(PrivateData, value); }
    void   TrivialArchive::Serialize(u32& value) { TrivialArchiveHelper::AppendValueToJson(PrivateData, value); }
    void   TrivialArchive::Serialize(u64& value) { TrivialArchiveHelper::AppendValueToJson(PrivateData, value); }
    void   TrivialArchive::Serialize(i8& value) { TrivialArchiveHelper::AppendValueToJson(PrivateData, value); }
    void   TrivialArchive::Serialize(i16& value) { TrivialArchiveHelper::AppendValueToJson(PrivateData, value); }
    void   TrivialArchive::Serialize(i32& value) { TrivialArchiveHelper::AppendValueToJson(PrivateData, value); }
    void   TrivialArchive::Serialize(i64& value) { TrivialArchiveHelper::AppendValueToJson(PrivateData, value); }
    void   TrivialArchive::Serialize(f32& value) { TrivialArchiveHelper::AppendValueToJson(PrivateData, value); }
    void   TrivialArchive::Serialize(f64& value) { TrivialArchiveHelper::AppendValueToJson(PrivateData, value); }
    void   TrivialArchive::Serialize(String& value) { TrivialArchiveHelper::AppendValueToJson(PrivateData, value); }

    String TrivialArchive::GetResult() const
    {
        return PrivateData->JsonData->dump(4, ' ', false, nlohmann::json::error_handler_t::replace);
    }

} // namespace Ifrit::Reflection