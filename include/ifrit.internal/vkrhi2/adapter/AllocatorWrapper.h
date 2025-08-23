#pragma once

#include <vma/vk_mem_alloc.h>

namespace Ifrit::RHI::VulkanRHI2
{
    struct VA_Allocator
    {
        VmaAllocator mAllocator = nullptr;
    };
} // namespace Ifrit::RHI::VulkanRHI2