#include "ifrit/core/reflection/Reflection.h"
#include <iostream>
#include <string>
#include "ifrit/core/reflection/SerializeHelper.h"
#include <map>

using namespace Ifrit::Reflection;

struct Cell
{
    std::string av = "meo";
};

class Cat
{
public:
    int               catv = 43;
    double            catd = 3.14;
    std::vector<int>  meow = { 0x0d, 0x00, 0x07 };
    std::vector<Cell> d    = { Cell(), Cell() };
};

class Dog
{
public:
    std::unique_ptr<Cat>         v  = std::make_unique<Cat>();
    float                        q  = 1919810;
    std::unique_ptr<Cat>         nk = nullptr;
    std::unordered_map<int, Cat> fvck;
    std::string                  str = "Goodbye World";
    std::vector<Cat>             a   = { Cat(), Cat(), Cat() };
};

int main()
{
    Dog sv;
    sv.fvck[114]       = Cat();
    sv.fvck[1919]      = Cat();
    sv.fvck[114].catv  = 514;
    sv.fvck[1919].catd = 810;
    sv.q               = 11451519;
    sv.a[0].meow[2]    = 114514;

    RegisterType<Cell>();
    RegisterType<Cat>();
    RegisterType<Dog>();

    RegisterPropertyField<&Cell::av>("av");
    RegisterPropertyField<&Cat::catv>("catv");
    RegisterPropertyField<&Cat::catd>("catd");
    RegisterPropertyField<&Cat::d>("d");
    RegisterPropertyField<&Cat::meow>("meow");

    RegisterPropertyField<&Dog::v>("v");
    RegisterPropertyField<&Dog::q>("q");
    RegisterPropertyField<&Dog::nk>("nk");
    RegisterPropertyField<&Dog::fvck>("fvck");
    // RegisterPropertyField<&Dog::str>("str");
    RegisterPropertyField<&Dog::a>("a");

    auto  catInstance = ConstructObject(Ifrit::FMetaTypeInfo::Create<Cat>());
    auto  dogInstance = ReferenceObject(&sv);

    auto& actDog = dogInstance.ObjectValue.As<Dog>();

    auto  value = SerializeToJSON(actDog);
    std::cout << "=============" << std::endl;
    std::cout << value << std::endl;
    std::cout << "=============" << std::endl;
    Dog newDog;
    DeserializeFromJSON(newDog, value);
    std::cout << "=============" << std::endl;
    std::cout << SerializeToJSON(newDog) << std::endl;
    std::cout << "=============" << std::endl;

    auto value2 = SerializeToJSON(newDog);

    std::cout << (value == value2) << std::endl;
    return 0;
}