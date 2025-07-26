#include "ifrit/core/reflection/Archive.h"
#include <iostream>
namespace Ifrit::Reflection
{
    void TrivialArchive::BeginObject(const String& name) { std::cout << "BeginObject: " << name << std::endl; }
    void TrivialArchive::EndObject() { std::cout << "EndObject" << std::endl; }

    void TrivialArchive::BeginArray(const String& name) { std::cout << "BeginArray: " << name << std::endl; }
    void TrivialArchive::EndArray() { std::cout << "EndArray" << std::endl; }
    void TrivialArchive::Serialize(u8& value) { std::cout << static_cast<int>(value) << " \n"; }
    void TrivialArchive::Serialize(u16& value) { std::cout << value << " \n"; }
    void TrivialArchive::Serialize(u32& value) { std::cout << value << " \n"; }
    void TrivialArchive::Serialize(u64& value) { std::cout << value << " \n"; }
    void TrivialArchive::Serialize(i8& value) { std::cout << static_cast<int>(value) << " \n"; }
    void TrivialArchive::Serialize(i16& value) { std::cout << value << " \n"; }
    void TrivialArchive::Serialize(i32& value) { std::cout << value << " \n"; }
    void TrivialArchive::Serialize(i64& value) { std::cout << value << " \n"; }
    void TrivialArchive::Serialize(f32& value) { std::cout << value << " \n"; }
    void TrivialArchive::Serialize(f64& value) { std::cout << value << " \n"; }
} // namespace Ifrit::Reflection