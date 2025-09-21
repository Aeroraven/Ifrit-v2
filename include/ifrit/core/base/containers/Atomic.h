#pragma once

#include <atomic>

namespace Ifrit
{
    template <typename T> using TAtomic = std::atomic<T>;
} // namespace Ifrit