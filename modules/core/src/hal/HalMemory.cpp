#include "ifrit/core/hal/HalMemory.h"
#include "mimalloc/include/mimalloc.h"

namespace Ifrit::HAL
{

    IFRIT_CORE_API void* MemAllocAligned(usize size, u32 alignment) { return mi_malloc_aligned(size, alignment); }
    IFRIT_CORE_API void* MemAlloc(usize size) { return mi_malloc(size); }
    IFRIT_CORE_API void  MemFree(void* ptr) { mi_free(ptr); }

} // namespace Ifrit::HAL