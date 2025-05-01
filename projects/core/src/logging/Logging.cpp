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

#include "ifrit/core/logging/Logging.h"

namespace Ifrit::Logging
{

    inline void RegisterLoggerModule(const std::string& name)
    {

        spdlog::set_pattern("[%H:%M:%S %z] [%n] [%^%l%$] %v");
        auto stdoutSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto logger     = std::make_shared<spdlog::logger>(name, stdoutSink);
        logger->set_pattern("[%Y/%m/%d %H:%M:%S %z] [%^%-7l%$] [%n] %v");
        logger->set_level(spdlog::level::trace);
        spdlog::register_logger(logger);
    }

    IFRIT_APIDECL std::shared_ptr<spdlog::logger> GetLoggerModule(const std::string& name)
    {
        auto logger = spdlog::get(name);
        if (!logger)
        {
            static std::mutex           loggerMutex;
            std::lock_guard<std::mutex> lock(loggerMutex);
            logger = spdlog::get(name);
            if (logger)
            {
                logger->set_pattern("[%Y/%m/%d %H:%M:%S %z] [%^%-7l%$] [%n] %v");
                return logger;
            }
            RegisterLoggerModule(name);
            printf("Registered logger module: %s\n", name.c_str());
            return spdlog::get(name);
        }
        logger->set_pattern("[%Y/%m/%d %H:%M:%S %z] [%^%-7l%$] [%n] %v");
        return logger;
    }
} // namespace Ifrit::Logging