#pragma once

#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/logging/Logging.h"

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
        virtual ~Archive()                           = default;
        virtual void BeginObject(const String& name) = 0;
        virtual void EndObject()                     = 0;
        virtual void BeginArray(const String& name)  = 0;
        virtual void EndArray()                      = 0;

        virtual void Serialize(u8& value)     = 0;
        virtual void Serialize(u16& value)    = 0;
        virtual void Serialize(u32& value)    = 0;
        virtual void Serialize(u64& value)    = 0;
        virtual void Serialize(i8& value)     = 0;
        virtual void Serialize(i16& value)    = 0;
        virtual void Serialize(i32& value)    = 0;
        virtual void Serialize(i64& value)    = 0;
        virtual void Serialize(f32& value)    = 0;
        virtual void Serialize(f64& value)    = 0;
        virtual void Serialize(String& value) = 0;

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

        virtual String GetResult() const = 0;
    };

    struct TrivialArchivePrivate;
    class TrivialArchiveHelper;

    class IFRIT_APIDECL TrivialArchive : public Archive
    {
    private:
        TrivialArchivePrivate* PrivateData;

    public:
        TrivialArchive();
        ~TrivialArchive() override;

        void   BeginObject(const String& name) override;
        void   EndObject() override;
        void   BeginArray(const String& name) override;
        void   EndArray() override;

        void   Serialize(u8& value) override;
        void   Serialize(u16& value) override;
        void   Serialize(u32& value) override;
        void   Serialize(u64& value) override;
        void   Serialize(i8& value) override;
        void   Serialize(i16& value) override;
        void   Serialize(i32& value) override;
        void   Serialize(i64& value) override;
        void   Serialize(f32& value) override;
        void   Serialize(f64& value) override;
        void   Serialize(String& value) override;

        String GetResult() const override;

        friend class TrivialArchiveHelper;
    };

} // namespace Ifrit::Reflection