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

#ifndef FMT_UNICODE
    #define FMT_UNICODE 0
#endif

#if FMT_UNICODE
    #undef FMT_UNICODE
    #define FMT_UNICODE 0
#endif

#include <format>
#define SPDLOG_HEADER_ONLY
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <tuple>

namespace Ifrit::Logging
{

    template <typename... Args>
    void Info(const char* fmt, Args&&... args)
    {
        spdlog::info(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void Warn(const char* fmt, Args&&... args)
    {
        spdlog::warn(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void Error(const char* fmt, Args&&... args)
    {
        spdlog::error(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void Debug(const char* fmt, Args&&... args)
    {
        spdlog::debug(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void Trace(const char* fmt, Args&&... args)
    {
        spdlog::trace(fmt, std::forward<Args>(args)...);
    }

    template <typename T>
    void Info(const T& msg)
    {
        spdlog::info(msg);
    }

    template <typename T>
    void Warn(const T& msg)
    {
        spdlog::warn(msg);
    }

    template <typename T>
    void Error(const T& msg)
    {
        spdlog::error(msg);
    }

    template <typename T>
    void Debug(const T& msg)
    {
        spdlog::debug(msg);
    }

    template <typename T>
    void Trace(const T& msg)
    {
        spdlog::trace(msg);
    }

    template <typename T>
    void Assertion(bool condition, const T& msg)
    {
        if (!condition)
        {
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
    }

    // v2

    inline void RegisterLoggerModule(const std::string& name)
    {
        spdlog::set_pattern("[%H:%M:%S %z] [%n] [%^%l%$] %v");
        auto stdoutSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto logger     = std::make_shared<spdlog::logger>(name, stdoutSink);
        logger->set_pattern("[%Y/%m/%d %H:%M:%S %z] [%^%-7l%$] [%n] %v");
        logger->set_level(spdlog::level::trace);
        spdlog::register_logger(logger);
    }

    inline std::shared_ptr<spdlog::logger> GetLoggerModule(const std::string& name)
    {
        auto logger = spdlog::get(name);
        if (!logger)
        {
            RegisterLoggerModule(name);
            return spdlog::get(name);
        }
        return logger;
    }

    template <typename... Args>
    inline void Info2(const char* moduleName, std::format_string<Args...> fmt, Args&&... args)
    {
        auto formatted = std::format(fmt, std::forward<Args>(args)...);
        GetLoggerModule(moduleName)->info(formatted);
    }

    template <typename... Args>
    inline void Warn2(const char* moduleName, std::format_string<Args...> fmt, Args&&... args)
    {
        auto formatted = std::format(fmt, std::forward<Args>(args)...);
        GetLoggerModule(moduleName)->warn(formatted);
    }

    template <typename... Args>
    inline void Error2(const char* moduleName, std::format_string<Args...> fmt, Args&&... args)
    {
        auto formatted = std::format(fmt, std::forward<Args>(args)...);
        GetLoggerModule(moduleName)->error(formatted);
    }

    template <typename... Args>
    inline void Debug2(const char* moduleName, std::format_string<Args...> fmt, Args&&... args)
    {
        auto formatted = std::format(fmt, std::forward<Args>(args)...);
        GetLoggerModule(moduleName)->debug(formatted);
    }

    template <typename... Args>
    inline void Trace2(const char* moduleName, std::format_string<Args...> fmt, Args&&... args)
    {
        auto formatted = std::format(fmt, std::forward<Args>(args)...);
        GetLoggerModule(moduleName)->trace(formatted);
    }

    template <typename T>
    inline void Info2(const char* moduleName, const T& msg)
    {
        auto s = GetLoggerModule(moduleName);
        s->info(msg);
    }

    template <typename T>
    inline void Warn2(const char* moduleName, const T& msg)
    {
        GetLoggerModule(moduleName)->warn(msg);
    }

    template <typename T>
    inline void error2(const char* moduleName, const T& msg)
    {
        GetLoggerModule(moduleName)->error(msg);
    }

    template <typename T>
    inline void Debug2(const char* moduleName, const T& msg)
    {
        GetLoggerModule(moduleName)->debug(msg);
    }

    template <typename T>
    inline void Trace2(const char* moduleName, const T& msg)
    {
        GetLoggerModule(moduleName)->trace(msg);
    }

    template <typename T>
    inline void Assertion2(const char* moduleName, bool condition, const T& msg)
    {
        if (!condition)
        {
            GetLoggerModule(moduleName)->error(msg);
            throw std::runtime_error(msg);
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
    #define iInfo(...) Logging::Info2(IFRIT_LOG_MODULE_NAME, __VA_ARGS__)
    #define iWarn(...) Logging::Warn2(IFRIT_LOG_MODULE_NAME, __VA_ARGS__)
    #define iError(...) Logging::Error2(IFRIT_LOG_MODULE_NAME, __VA_ARGS__)
    #define iDebug(...) Logging::Debug2(IFRIT_LOG_MODULE_NAME, __VA_ARGS__)
    #define iTrace(...) Logging::Trace2(IFRIT_LOG_MODULE_NAME, __VA_ARGS__)
    #define iAssertion(condition, ...) Logging::Assertion2(IFRIT_LOG_MODULE_NAME, condition, __VA_ARGS__)
#else
    static_assert(false, "IFRIT_LOG_MODULE_NAME is not defined");
#endif
} // namespace Ifrit::Logging