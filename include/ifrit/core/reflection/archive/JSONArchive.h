#pragma once

#include "ifrit/core/reflection/Archive.h"

namespace Ifrit::Reflection
{
    struct JSONArchivePrivate;
    class JSONArchiveHelper;

    class IFRIT_APIDECL JSONArchive : public Archive
    {
    private:
        JSONArchivePrivate* PrivateData;

    public:
        JSONArchive();
        ~JSONArchive() override;

        void   BeginObject(const String& name) override;
        void   EndObject() override;
        void   BeginArray(const String& name) override;
        void   EndArray() override;

        // Reading operations
        bool   HasObject(const String& name) override;
        bool   HasArray(const String& name) override;
        usize  GetArraySize() override;
        void   BeginArrayIteration() override;
        bool   HasNextArrayElement() override;
        void   NextArrayElement() override;

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
        void   Serialize(bool& value) override;

        void   LoadFromString(const String& data) override;
        String GetResult() const override;

        friend class JSONArchiveHelper;
    };

} // namespace Ifrit::Reflection