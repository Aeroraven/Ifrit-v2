#pragma once

#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"

namespace Ifrit::Reflection
{
    class Archive
    {
    public:
        virtual void BeginObject(const String& name) = 0;
        virtual void EndObject()                     = 0;
        virtual void BeginArray(const String& name)  = 0;
        virtual void EndArray()                      = 0;

        virtual void Serialize(u8& value)  = 0;
        virtual void Serialize(u16& value) = 0;
        virtual void Serialize(u32& value) = 0;
        virtual void Serialize(u64& value) = 0;
        virtual void Serialize(i8& value)  = 0;
        virtual void Serialize(i16& value) = 0;
        virtual void Serialize(i32& value) = 0;
        virtual void Serialize(i64& value) = 0;
        virtual void Serialize(f32& value) = 0;
        virtual void Serialize(f64& value) = 0;
    };

    class IFRIT_APIDECL TrivialArchive : public Archive
    {
    public:
        void BeginObject(const String& name) override;
        void EndObject() override;
        void BeginArray(const String& name) override;
        void EndArray() override;

        void Serialize(u8& value) override;
        void Serialize(u16& value) override;
        void Serialize(u32& value) override;
        void Serialize(u64& value) override;
        void Serialize(i8& value) override;
        void Serialize(i16& value) override;
        void Serialize(i32& value) override;
        void Serialize(i64& value) override;
        void Serialize(f32& value) override;
        void Serialize(f64& value) override;
    };

} // namespace Ifrit::Reflection