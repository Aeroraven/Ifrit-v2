#include "ifrit/core/reflection/archive/JSONArchive.h"
#include "ifrit/core/logging/Logging.h"
#include "json/json.h"
#include <stack>
namespace Ifrit::Reflection
{
    enum class ETopLevelType : u8
    {
        Object,
        Array,
    };

    struct JSONArchivePrivate
    {
        nlohmann::json*                      JsonData;
        std::stack<nlohmann::json*>          ObjectStack;
        std::stack<ETopLevelType>            TopLevelStack;

        std::stack<nlohmann::json::iterator> IteratorStack;
        std::stack<nlohmann::json::iterator> EndIteratorStack;

        std::stack<i32>                      ObjectStackValue;
    };

    class JSONArchiveHelper
    {
    public:
        template <typename T> static void ReadValueFromJson(JSONArchivePrivate* privateData, T& value)
        {
            auto* current = privateData->ObjectStack.top();
            if (privateData->TopLevelStack.top() == ETopLevelType::Array)
            {
                if (privateData->IteratorStack.top() != privateData->EndIteratorStack.top())
                {
                    value = privateData->IteratorStack.top()->get<T>();
                }
            }
            else
            {
                value = current->get<T>();
            }
        }

        static bool HasObjectInJson(JSONArchivePrivate* privateData, const String& name)
        {
            auto* current = privateData->ObjectStack.top();
            auto  ret     = current->contains(name); // && (*current)[name].is_object();
            if (!ret)
            {
                // print available keys
                IF_LOG_ERROR("Reflector",
                    "The deserialization requires the object with name {}, but the archive does not contain this key. Available keys are:",
                    name);
                for (auto it = current->begin(); it != current->end(); ++it)
                {
                    IF_LOG_ERROR("Reflector", "  - '{}' (type: {}), {}", it.key(), it.value().type_name(),
                        (*current)[name].is_object());
                }
                IF_LOG_CRITICAL("Reflector", " Serialization failed, aborting the application");
            }
            return ret;
        }

        static bool HasArrayInJson(JSONArchivePrivate* privateData, const String& name)
        {
            auto* current = privateData->ObjectStack.top();
            bool  ret     = current->contains(name) && (*current)[name].is_array();
            if (!ret)
            {
                // print available keys
                IF_LOG_ERROR("Reflector",
                    "The deserialization requires the array with name {}, but the archive does not contain this key. Available keys are:",
                    name);
                for (auto it = current->begin(); it != current->end(); ++it)
                {
                    IF_LOG_ERROR("Reflector", "  - '{}' (type: {}), {}", it.key(), it.value().type_name(),
                        (*current)[name].is_object());
                }
                IF_LOG_CRITICAL("Reflector", " Serialization failed, aborting the application");
            }
            return ret;
        }

        static void BeginObjectRead(JSONArchivePrivate* privateData, const String& name)
        {
            auto* current = privateData->ObjectStack.top();
            if (privateData->TopLevelStack.top() == ETopLevelType::Object)
            {
                privateData->ObjectStack.push(&(*current)[name]);
                privateData->TopLevelStack.push(ETopLevelType::Object);
            }
            else if (privateData->TopLevelStack.top() == ETopLevelType::Array)
            {
                privateData->ObjectStack.push(&(*privateData->IteratorStack.top()));
                privateData->TopLevelStack.push(ETopLevelType::Object);
            }
        }

        static void BeginArrayRead(JSONArchivePrivate* privateData, const String& name)
        {
            auto* current = privateData->ObjectStack.top();
            if (privateData->TopLevelStack.top() == ETopLevelType::Object)
            {
                privateData->ObjectStack.push(&(*current)[name]);
                privateData->TopLevelStack.push(ETopLevelType::Array);

                auto& arr = *privateData->ObjectStack.top();
                privateData->IteratorStack.push(arr.begin());
                privateData->EndIteratorStack.push(arr.end());
            }
        }

        static void AppendObjectToJson(JSONArchivePrivate* privateData, const String& name)
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

        static void AppendArrayToJson(JSONArchivePrivate* privateData, const String& name)
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
        static void AppendValueToJson(JSONArchivePrivate* privateData, const T& value, String name = "")
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

        template <typename T>
        static void DoSerialize(JSONArchivePrivate* privateData, ESerializationState State, T& value)
        {
            if (State == ESerializationState::Reading)
            {
                ReadValueFromJson(privateData, value);
            }
            else
            {
                AppendValueToJson(privateData, value);
            }
        }
    };

    JSONArchive::JSONArchive() : PrivateData(new JSONArchivePrivate())
    {
        PrivateData->JsonData = new nlohmann::json();
        PrivateData->ObjectStack.push(PrivateData->JsonData);
        PrivateData->TopLevelStack.push(ETopLevelType::Object);
    }
    JSONArchive::~JSONArchive()
    {
        delete PrivateData->JsonData;
        delete PrivateData;
    }

    void JSONArchive::EndObject()
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

    void JSONArchive::EndArray()
    {
        if (PrivateData->ObjectStack.size() > 1)
        {
            PrivateData->ObjectStack.pop();
            PrivateData->TopLevelStack.pop();
            if (State == ESerializationState::Reading)
            {
                PrivateData->IteratorStack.pop();
                PrivateData->EndIteratorStack.pop();
            }
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "EndArray called without matching BeginArray");
        }
    }

    void JSONArchive::Serialize(u8& value) { JSONArchiveHelper::DoSerialize(PrivateData, GetState(), value); }
    void JSONArchive::Serialize(u16& value) { JSONArchiveHelper::DoSerialize(PrivateData, GetState(), value); }
    void JSONArchive::Serialize(u32& value) { JSONArchiveHelper::DoSerialize(PrivateData, GetState(), value); }
    void JSONArchive::Serialize(u64& value) { JSONArchiveHelper::DoSerialize(PrivateData, GetState(), value); }
    void JSONArchive::Serialize(i8& value) { JSONArchiveHelper::DoSerialize(PrivateData, GetState(), value); }
    void JSONArchive::Serialize(i16& value) { JSONArchiveHelper::DoSerialize(PrivateData, GetState(), value); }
    void JSONArchive::Serialize(i32& value) { JSONArchiveHelper::DoSerialize(PrivateData, GetState(), value); }
    void JSONArchive::Serialize(i64& value) { JSONArchiveHelper::DoSerialize(PrivateData, GetState(), value); }
    void JSONArchive::Serialize(f32& value) { JSONArchiveHelper::DoSerialize(PrivateData, GetState(), value); }
    void JSONArchive::Serialize(f64& value) { JSONArchiveHelper::DoSerialize(PrivateData, GetState(), value); }
    void JSONArchive::Serialize(String& value) { JSONArchiveHelper::DoSerialize(PrivateData, GetState(), value); }
    void JSONArchive::Serialize(bool& value) { JSONArchiveHelper::DoSerialize(PrivateData, GetState(), value); }

    bool JSONArchive::HasObject(const String& name) { return JSONArchiveHelper::HasObjectInJson(PrivateData, name); }

    bool JSONArchive::HasArray(const String& name) { return JSONArchiveHelper::HasArrayInJson(PrivateData, name); }

    void JSONArchive::BeginObject(const String& name)
    {
        if (State == ESerializationState::Writing)
        {
            JSONArchiveHelper::AppendObjectToJson(PrivateData, name);
        }
        else
        {
            JSONArchiveHelper::BeginObjectRead(PrivateData, name);
        }
    }

    void JSONArchive::BeginArray(const String& name)
    {
        if (State == ESerializationState::Writing)
        {
            JSONArchiveHelper::AppendArrayToJson(PrivateData, name);
        }
        else
        {
            JSONArchiveHelper::BeginArrayRead(PrivateData, name);
        }
    }

    size_t JSONArchive::GetArraySize()
    {
        if (State == ESerializationState::Reading)
        {
            return PrivateData->ObjectStack.top()->size();
        }
        return 0;
    }

    void JSONArchive::BeginArrayIteration() {}

    bool JSONArchive::HasNextArrayElement()
    {
        if (State == ESerializationState::Reading)
        {
            return PrivateData->IteratorStack.top() != PrivateData->EndIteratorStack.top();
        }
        return false;
    }

    void JSONArchive::NextArrayElement()
    {
        if (State == ESerializationState::Reading)
        {
            ++PrivateData->IteratorStack.top();
        }
    }

    void JSONArchive::LoadFromString(const String& data)
    {
        State                  = ESerializationState::Reading;
        *PrivateData->JsonData = nlohmann::json::parse(data);

        // Clear stacks and reinitialize for reading
        while (!PrivateData->ObjectStack.empty())
            PrivateData->ObjectStack.pop();
        while (!PrivateData->TopLevelStack.empty())
            PrivateData->TopLevelStack.pop();

        PrivateData->ObjectStack.push(PrivateData->JsonData);
        PrivateData->TopLevelStack.push(ETopLevelType::Object);
    }
    String JSONArchive::GetResult() const
    {
        return PrivateData->JsonData->dump(4, ' ', false, nlohmann::json::error_handler_t::replace);
    }

    void JSONArchive::PushObjectVerificationReq()
    {
        PrivateData->ObjectStackValue.push(PrivateData->ObjectStack.size());
    }
    bool JSONArchive::PopObjectVerificationReq()
    {
        auto topValue = PrivateData->ObjectStackValue.top();
        PrivateData->ObjectStackValue.pop();
        if (PrivateData->ObjectStack.size() != topValue)
        {
            IF_LOG_ERROR("Reflector", "Object verification failed, expected stack size {}, but got {}", topValue,
                PrivateData->ObjectStack.size());
            return false;
        }
        return true;
    }

} // namespace Ifrit::Reflection