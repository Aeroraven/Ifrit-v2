#pragma once

#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/math/VectorDefs.h"
namespace Ifrit::Reflection
{
    enum class ESerializationState : u8
    {
        Reading,
        Writing,
    };

    class Archive
    {
    protected:
        ESerializationState State = ESerializationState::Writing;

    public:
        void                SetState(ESerializationState state) { State = state; }
        ESerializationState GetState() const { return State; }

        virtual ~Archive()                             = default;
        virtual void   BeginObject(const String& name) = 0;
        virtual void   EndObject()                     = 0;
        virtual void   BeginArray(const String& name)  = 0;
        virtual void   EndArray()                      = 0;

        // Reading operations
        virtual bool   HasObject(const String& name) = 0;
        virtual bool   HasArray(const String& name)  = 0;
        virtual size_t GetArraySize()                = 0;
        virtual void   BeginArrayIteration()         = 0;
        virtual bool   HasNextArrayElement()         = 0;
        virtual void   NextArrayElement()            = 0;

        virtual void   Serialize(u8& value)               = 0;
        virtual void   Serialize(u16& value)              = 0;
        virtual void   Serialize(u32& value)              = 0;
        virtual void   Serialize(u64& value)              = 0;
        virtual void   Serialize(i8& value)               = 0;
        virtual void   Serialize(i16& value)              = 0;
        virtual void   Serialize(i32& value)              = 0;
        virtual void   Serialize(i64& value)              = 0;
        virtual void   Serialize(f32& value)              = 0;
        virtual void   Serialize(f64& value)              = 0;
        virtual void   Serialize(String& value)           = 0;
        virtual void   Serialize(bool& value)             = 0;
        virtual void   LoadFromString(const String& data) = 0;

        template <typename T>
            requires std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<String, T>
        void Serialize(const T& value)
        {
            if (State == ESerializationState::Writing)
            {
                Serialize(const_cast<T&>(value));
            }
            else
            {
                IF_LOG_CRITICAL("Archive", "Cannot serialize value in reading state: {}", typeid(T).name());
            }
        }

        template <typename T> void Serialize(CoreVec2<T>& value)
        {
            BeginObject("__ifrit_vector2");
            BeginObject("x");
            Serialize(value.x);
            EndObject();
            BeginObject("y");
            Serialize(value.y);
            EndObject();
            EndObject();
        }

        template <typename T> void Serialize(CoreVec3<T>& value)
        {
            BeginObject("__ifrit_vector3");
            BeginObject("x");
            Serialize(value.x);
            EndObject();
            BeginObject("y");
            Serialize(value.y);
            EndObject();
            BeginObject("z");
            Serialize(value.z);
            EndObject();
            EndObject();
        }

        template <typename T> void Serialize(CoreVec4<T>& value)
        {
            BeginObject("__ifrit_vector4");
            BeginObject("x");
            Serialize(value.x);
            EndObject();
            BeginObject("y");
            Serialize(value.y);
            EndObject();
            BeginObject("z");
            Serialize(value.z);
            EndObject();
            BeginObject("w");
            Serialize(value.w);
            EndObject();
            EndObject();
        }

        virtual String GetResult() const = 0;
    };

} // namespace Ifrit::Reflection