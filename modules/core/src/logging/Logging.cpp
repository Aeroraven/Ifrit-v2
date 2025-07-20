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

#ifndef FMT_UNICODE
    #define FMT_UNICODE 0
#endif

#if FMT_UNICODE
    #undef FMT_UNICODE
    #define FMT_UNICODE 0
#endif

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <tuple>
#include <chrono>
#include "ifrit/core/logging/Logging.h"

namespace Ifrit::Logging
{
    static Atomic<u32>             sLogEntries = 0;
    static Vec<InternalLogEntries> sLogEntriesVec(1145141);

    IFRIT_APIDECL VecView<InternalLogEntries> GetLogEntries()
    {
        return VecView<InternalLogEntries>(sLogEntriesVec.data(), sLogEntries.load());
    }

    inline void RegisterLoggerModule(const std::string& name)
    {
        auto stdoutSink = MakeRef<spdlog::sinks::stdout_color_sink_mt>();
        auto logger     = MakeRef<spdlog::logger>(name, stdoutSink);
        logger->set_pattern("[%Y/%m/%d %H:%M:%S %z] [%^%-7l%$] %v");
        logger->set_level(spdlog::level::trace);
        spdlog::register_logger(logger);
    }

    std::shared_ptr<spdlog::logger> GetLoggerModule(const std::string& name)
    {
        auto logger = spdlog::get(name);
        if (!logger)
        {
            static std::mutex           loggerMutex;
            std::lock_guard<std::mutex> lock(loggerMutex);
            logger = spdlog::get(name);
            if (logger)
            {
                logger->set_pattern("[%Y/%m/%d %H:%M:%S %z] [%^%-7l%$] %v");
                return logger;
            }
            RegisterLoggerModule(name);
            return spdlog::get(name);
        }
        logger->set_pattern("[%Y/%m/%d %H:%M:%S %z] [%^%-7l%$] %v");
        return logger;
    }

    IFRIT_APIDECL void LogImpl(ELoggingLevel level, const String& message)
    {
        auto logger = GetLoggerModule("IfritLogger");
        switch (level)
        {
            case ELoggingLevel::Trace:
                logger->trace(message);
                break;
            case ELoggingLevel::Debug:
                logger->debug(message);
                break;
            case ELoggingLevel::Info:
                logger->info(message);
                break;
            case ELoggingLevel::Warning:
                logger->warn(message);
                break;
            case ELoggingLevel::Error:
                logger->error(message);
                break;
            case ELoggingLevel::Critical:
                logger->critical(message);
                break;
            default:
                logger->info(message);
        }

        InternalLogEntries entry;
        entry.m_Time    = std::format("{:%Y-%m-%d %H:%M:%S}", std::chrono::system_clock::now());
        entry.m_Message = message;
        entry.m_Level   = level;
        auto entryId    = sLogEntries.fetch_add(1);

        sLogEntriesVec[entryId] = std::move(entry);
    }

    IFRIT_APIDECL String LogAppendModuleInfo(const String& formatted, const char* moduleName, const char* subModuleName)
    {
        String ret = std::format("[{}] ({}) {}", moduleName, subModuleName, formatted);
        return ret;
    }

} // namespace Ifrit::Logging