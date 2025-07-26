#include "ifrit/core/reflection/Reflection.h"
#include <iostream>
#include <string>
#include "ifrit/core/reflection/Serializer.h"
#include <map>

using namespace Ifrit::Reflection;

class Cat
{
public:
    int              v    = 43;
    double           d    = 3.14;
    std::vector<int> meow = { 0x0d, 0x00, 0x07 };
};

class Dog
{
public:
    std::unique_ptr<Cat>         v  = std::make_unique<Cat>();
    float                        q  = 1919810;
    std::unique_ptr<Cat>         nk = nullptr;
    std::unordered_map<int, Cat> fvck;
    std::string                  str = "Goodbye World";
};

int main()
{
    Dog sv;
    sv.fvck[114]    = Cat();
    sv.fvck[1919]   = Cat();
    sv.fvck[114].v  = 514;
    sv.fvck[1919].d = 810;

    RegisterType<Cat>();
    RegisterType<Dog>();
    RegisterPropertyField<&Cat::v>("v");
    RegisterPropertyField<&Cat::d>("d");
    RegisterPropertyField<&Cat::meow>("meow");

    RegisterPropertyField<&Dog::v>("v");
    RegisterPropertyField<&Dog::q>("q");
    RegisterPropertyField<&Dog::nk>("nk");
    RegisterPropertyField<&Dog::fvck>("fvck");
    RegisterPropertyField<&Dog::str>("str");

    auto           catInstance = ConstructObject(Ifrit::FMetaTypeInfo::Create<Cat>());
    auto           dogInstance = ReferenceObject(&sv);

    TrivialArchive archive;
    dogInstance.ObjectValue.Serialize(&archive);
    std::cout << archive.GetResult();
    return 0;
}