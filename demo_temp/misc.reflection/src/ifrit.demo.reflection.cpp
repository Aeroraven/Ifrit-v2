#include "ifrit/core/reflection/Reflection.h"
#include <iostream>
#include <string>
#include "ifrit/core/reflection/SerializeHelper.h"
#include <map>
#include "misc.reflection.generated.h"
#include "ifrit/core/tasks/TaskScheduler.h"
#include "ifrit/runtime/rendercore/threading/RenderingThread.h"
#include <Windows.h>

using namespace Ifrit::Reflection;

struct Cell
{
    std::string av = "meo";
    bool        aa = true;
};

class Cat
{
public:
    int               catv = 43;
    double            catd = 3.14;
    std::vector<int>  meow = { 0x0d, 0x00, 0x07 };
    std::vector<Cell> d    = { Cell(), Cell() };

public:
    virtual void Meow(int a, int b)
    {
        std::cout << "Meow"
                  << " " << a << " " << b << std::endl;
    }
};

class Tabby : public Cat
{
public:
    int fww = 1919810;
};

enum class DogBreed
{
    Labrador,
    Beagle,
    Bulldog,
    Poodle
};

class Dog
{
public:
    DogBreed                     breed = DogBreed::Labrador;
    int                  id;
    std::unique_ptr<Cat>         v  = std::make_unique<Tabby>();
    float                        q  = 1919810;
    std::unique_ptr<Cat>         nk = std::make_unique<Cat>();
    std::unordered_map<int, Cat> fvck;
    std::string                  str = "Goodbye World";
    std::vector<Cat>             a   = { Cat(), Cat(), Cat() };
    Vector4f                     a2  = Vector4f(1.0f, 2.0f, 3.0f, 4.0f);
};

int reflTest()
{
    // Ifrit::Reflection::RegisterReflectionTypes();

    using C = Ifrit::TMemberFunctionTrait<decltype(&Cat::Meow)>::ReturnType;

    Dog sv;
    sv.fvck[114]       = Cat();
    sv.fvck[1919]      = Cat();
    sv.fvck[114].catv  = 514;
    sv.fvck[1919].catd = 810;
    sv.q               = 11451519;
    sv.a[0].meow[2]    = 114514;
    sv.id              = 1;

    RegisterType<Cell>();
    RegisterType<Cat>();
    RegisterType<Dog>();
    RegisterType<Tabby>();

    RegisterMethodField<&Cat::Meow>("Meow");
    Cat  d;
    auto p = ReferenceObject(&d);
    std::vector<Ifrit::Reflection::Object> aw3;
    aw3.push_back(std::move(Ifrit::Reflection::Object::CreateClone(1)));
    aw3.push_back(std::move(Ifrit::Reflection::Object::CreateClone(2)));

    InvokeMethod(p, Ifrit::FMetaMethodInfo::Create<&Cat::Meow>(),aw3);

    RegisterPropertyField<&Cell::av>("av");
    RegisterPropertyField<&Cell::aa>("aa");

    RegisterPropertyField<&Cat::catv>("catv");
    RegisterPropertyField<&Cat::catd>("catd");
    RegisterPropertyField<&Cat::d>("d");
    RegisterPropertyField<&Cat::meow>("meow");

    RegisterPolymorphicRelation<Tabby, Cat>();
    RegisterPropertyField<&Tabby::fww>("fww");

    // RegisterPropertyField<&Dog::v>("v");
    // RegisterPropertyField<&Dog::q>("q");
    // RegisterPropertyField<&Dog::nk>("nk");
    // RegisterPropertyField<&Dog::fvck>("fvck");
    //// RegisterPropertyField<&Dog::str>("str");
    // RegisterPropertyField<&Dog::a>("a");
    RegisterPropertyField<&Dog::id>("id");
    RegisterPropertyField<&Dog::a2>("a2");
    RegisterPropertyField<&Dog::breed>("breed");

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

    constexpr bool poly = std::is_polymorphic_v<Cat>;
    return 0;
}

int taskTest()
{
    using namespace Ifrit;
    auto taskScheduler = Task::GetTaskScheduler();

    Runtime::RegisterRenderCoreThreading();

    auto task1 = taskScheduler->EnqueueTask(
        [](Task::Task* t, void* p) {
            Sleep(1000);
            std::cout << "<1 " << Runtime::IsInRenderingThread() << " >" << std::endl;
        },
        Task::ENamedTaskThread::AnyThread, {}, nullptr);

    auto task3 = taskScheduler->EnqueueTask(
        [](Task::Task* t, void* p) {
            Sleep(2000);
            std::cout << "<3 " << Runtime::IsInRenderingThread() << ">" << std::endl;
        },
        Task::ENamedTaskThread::AnyThread, {}, nullptr);


    auto task2 = taskScheduler->EnqueueTask(
        [](Task::Task* t, void* p) {
            Sleep(5000);
            std::cout << "<2 " << Runtime::IsInRenderingThread() << ">" << std::endl;
        },
        Task::ENamedTaskThread::RenderThread, { task1, task3 }, nullptr);

    auto task4 = taskScheduler->EnqueueTask(
        [](Task::Task* t, void* p) {
            Sleep(5000);
            std::cout << "<4 "<< Runtime::IsInRenderingThread() <<">" << std::endl;
        },
        Task::ENamedTaskThread::AnyThread, { task2, task3 }, nullptr);
    auto task5 = taskScheduler->EnqueueTask(
        [](Task::Task* t, void* p) {
            Sleep(5000);
            std::cout << "<5 "<< Runtime::IsInRenderingThread() <<">" << std::endl;
        },
        Task::ENamedTaskThread::RenderThread, {}, nullptr);

    taskScheduler->WaitForTask(task1);
    taskScheduler->WaitForTask(task2);
    taskScheduler->WaitForTask(task3);
    taskScheduler->WaitForTask(task4);
    taskScheduler->WaitForTask(task5);
    return 0;
}

int main()
{ 
    return taskTest(); 
}