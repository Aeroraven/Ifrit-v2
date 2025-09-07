#include "ifrit/core/hal/HalDllImport.h"
#include "ifrit/core/logging/Logging.h"

#ifdef _WIN32
    #include <Windows.h>
#else
    #include <dlfcn.h>
#endif

namespace Ifrit::HAL
{
    struct FDynamicLinkedLibModule
    {
#ifdef _WIN32
        HMODULE m_ModuleHandle = nullptr;
#else
        void*                    m_ModuleHandle = nullptr;
#endif
    };

    FDynamicLinkedLibModule* LoadDynamicLinkedLibrary(const String& path)
    {
#ifdef _WIN32
        FDynamicLinkedLibModule* mod = new FDynamicLinkedLibModule();
        mod->m_ModuleHandle          = LoadLibraryA(path.c_str());
        if (!mod->m_ModuleHandle)
        {
            IF_LOG_ERROR("HAL", "Failed to load dynamic linked library: {}", path);
            delete mod;
            return nullptr;
        }
        IF_LOG_INFO("HAL", "Successfully loaded dynamic linked library: {}", path);
        return mod;
#else
        FDynamicLinkedLibModule* mod            = new FDynamicLinkedLibModule();
        mod->m_ModuleHandle                     = dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD);
        if (!mod->m_ModuleHandle)
        {
            IF_LOG_ERROR("HAL", "Failed to load dynamic linked library: {}", path);
            delete mod;
            return nullptr;
        }
        IF_LOG_INFO("HAL", "Successfully loaded dynamic linked library: {}", path);
        return mod;
#endif
    }

    void* LoadDllFunction(FDynamicLinkedLibModule* lib, const String& functionName)
    {
#ifdef _WIN32
        if (!lib || !lib->m_ModuleHandle)
        {
            IF_LOG_ERROR("HAL", "Invalid dynamic linked library module.");
            return nullptr;
        }
        void* funcPtr = GetProcAddress(lib->m_ModuleHandle, functionName.c_str());
        if (!funcPtr)
        {
            IF_LOG_ERROR("HAL", "Failed to load function '{}' from dynamic linked library.", functionName);
            return nullptr;
        }
        IF_LOG_INFO("HAL", "Successfully loaded function '{}' from dynamic linked library.", functionName);
        return funcPtr;
#else
        if (!lib || !lib->m_ModuleHandle)
        {
            IF_LOG_ERROR("HAL", "Invalid dynamic linked library module.");
            return nullptr;
        }
        void* funcPtr = dlsym(lib->m_ModuleHandle, functionName.c_str());
        if (!funcPtr)
        {
            IF_LOG_ERROR("HAL", "Failed to load function '{}' from dynamic linked library.", functionName);
            return nullptr;
        }
        IF_LOG_INFO("HAL", "Successfully loaded function '{}' from dynamic linked library.", functionName);
        return funcPtr;
#endif
        return nullptr;
    }
} // namespace Ifrit::HAL