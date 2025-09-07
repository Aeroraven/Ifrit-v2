#pragma once

#include <iostream>

template <typename... Args> void LogInfo(Args&&... args)
{
    std::cout << "[Ifrit.ReflParser]: ";
    (std::cout << ... << args) << std::endl;
}