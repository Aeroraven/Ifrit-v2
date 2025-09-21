/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once

#include <format>
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/typing/Traits.h"

namespace Ifrit::Logging
{
    enum ELoggingLevel
    {
        Trace,
        Debug,
        Info,
        Warning,
        Error,
        Critical
    };
    struct InternalLogEntries
    {
        String        m_Time;
        String        m_Message;
        ELoggingLevel m_Level;
    };

    // v3
    IFRIT_CORE_API void   LogImpl(ELoggingLevel level, const String& message);
    IFRIT_CORE_API String LogAppendModuleInfo(
        const String& formatted, const char* moduleName, const char* subModuleName);
    IFRIT_CORE_API VecView<InternalLogEntries> GetLogEntries();

    template <ELoggingLevel Level, typename... Args>
        requires IFormattableAll<Args...>
    inline void LogWrapper(
        const char* moduleName, const char* subModule, std::format_string<Args...> fmt, Args&&... args)
    {
        auto formatted = std::format(fmt, std::forward<Args>(args)...);
        formatted      = LogAppendModuleInfo(formatted, moduleName, subModule);
        LogImpl(Level, formatted);
        if IF_CONSTEXPR (Level == ELoggingLevel::Critical)
        {
            std::terminate();
        }
    }

    template <typename... Args>
        requires IFormattableAll<Args...>
    inline void LogAssertion(
        const char* moduleName, const char* subModule, bool condition, std::format_string<Args...> fmt, Args&&... args)
    {
        if (!condition) IF_UNLIKELY
        {
            LogWrapper<ELoggingLevel::Critical>(moduleName, subModule, fmt, std::forward<Args>(args)...);
        }
    }

#ifdef __INTELLISENSE__
    #ifndef IFRIT_LOG_MODULE_NAME
        #define IFRIT_LOG_MODULE_NAME "IFRIT_LOG_MODULE_NAME"
    #endif
#else
    #ifndef IFRIT_LOG_MODULE_NAME
        #define IFRIT_LOG_MODULE_NAME "Ifrit.Common"
    #endif
#endif

#ifdef IFRIT_LOG_MODULE_NAME

    #define IF_LOG_GENERAL(level, module, submodule, ...) \
        Ifrit::Logging::LogWrapper<Ifrit::Logging::level>(module, submodule, __VA_ARGS__)
    #define IF_LOG_ASSERTION_IMPL(module, submodule, condition, ...) \
        Ifrit::Logging::LogAssertion(module, submodule, condition, __VA_ARGS__)

    #define IF_LOG_TRACE(submodule, ...) \
        IF_LOG_GENERAL(ELoggingLevel::Trace, IFRIT_LOG_MODULE_NAME, submodule, __VA_ARGS__)
    #define IF_LOG_DEBUG(submodule, ...) \
        IF_LOG_GENERAL(ELoggingLevel::Debug, IFRIT_LOG_MODULE_NAME, submodule, __VA_ARGS__)
    #define IF_LOG_INFO(submodule, ...) \
        IF_LOG_GENERAL(ELoggingLevel::Info, IFRIT_LOG_MODULE_NAME, submodule, __VA_ARGS__)
    #define IF_LOG_WARNING(submodule, ...) \
        IF_LOG_GENERAL(ELoggingLevel::Warning, IFRIT_LOG_MODULE_NAME, submodule, __VA_ARGS__)
    #define IF_LOG_ERROR(submodule, ...) \
        IF_LOG_GENERAL(ELoggingLevel::Error, IFRIT_LOG_MODULE_NAME, submodule, __VA_ARGS__)
    #define IF_LOG_CRITICAL(submodule, ...) \
        IF_LOG_GENERAL(ELoggingLevel::Critical, IFRIT_LOG_MODULE_NAME, submodule, __VA_ARGS__)
    #define IF_LOG_REMOVED_FEATURE(submodule, ...)                                \
        IF_LOG_GENERAL(ELoggingLevel::Critical, IFRIT_LOG_MODULE_NAME, submodule, \
            "Following features are removed. Requesting feature:" __VA_ARGS__)

    #define IF_LOG_ASSERTION(submodule, condition, ...) \
        IF_LOG_ASSERTION_IMPL(IFRIT_LOG_MODULE_NAME, submodule, condition, __VA_ARGS__)

#else
    static_assert(false, "IFRIT_LOG_MODULE_NAME is not defined");
#endif
} // namespace Ifrit::Logging