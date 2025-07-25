#include "ifrit/core/reflection/Reflection.h"
#include <iostream>
using namespace Ifrit::Reflection;
class Neko
{
public:
    int v;
};

int main()
{
    RegisterType<Neko>();
    RegisterPropertyField<&Neko::v>("v");

    auto nekoInstance = ConstructObject("Neko");

    return 0;
}