#include "ifrit/core/reflection/Reflection.h"
#include <iostream>
#include <string>
#include "ifrit/core/reflection/Serializer.h"

using namespace Ifrit::Reflection;

class Cat
{
public:
    int         v    = 43;
    double      d    = 3.14;
    std::string name = "Cat";
};

class Dog
{
public:
    int v = 42;
};

int main()
{

    RegisterType<Cat>();
    RegisterType<Dog>();
    RegisterPropertyField<&Cat::v>("v");
    RegisterPropertyField<&Cat::d>("d");
    RegisterPropertyField<&Cat::name>("name");
    RegisterPropertyField<&Dog::v>("v");

    auto catInstance = ConstructObject(Ifrit::FMetaTypeInfo::Create<Cat>());
    auto dogInstance = ConstructObject(Ifrit::FMetaTypeInfo::Create<Dog>());
    auto catProps    = GetPropertyList(catInstance);
    auto dogProps    = GetPropertyList(dogInstance);
    for (const auto& [k, v] : catProps)
    {
        std::cout << "CatProps Name: " << k << ", Value: " << v.Value() << std::endl;
    }
    for (const auto& [k, v] : dogProps)
    {
        std::cout << "DogProps Name: " << k << ", Value: " << v.Value() << std::endl;
    }

    std::vector<int> vec = { 1, 2, 3, 4, 5 };

    TrivialArchive   archive;
    InvokeSerialize(vec, &archive);
    return 0;
}