
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
#define _CRT_SECURE_NO_WARNINGS
#include "ifrit/softgraphics/core/definition/CoreDefs.h"
namespace Ifrit::Graphics::SoftGraphics::Core::Utility
{
	class CoreLogger
	{
	private:
	public:
		static std::mutex& getMutex()
		{
			static std::mutex logMutex;
			return logMutex;
		}
		template <typename... Args>
		static void log(int32_t level, const char* caller, Args... args)
		{
			getMutex().lock();
			std::string logLevel;
			switch (level)
			{
				case 0:
					logLevel = "DEBUG";
					break;
				case 1:
					logLevel = "INFO";
					break;
				case 2:
					logLevel = "WARNING";
					break;
				case 3:
					logLevel = "ERROR";
					break;
				case 4:
					logLevel = "CRITICAL";
					break;
				default:
					logLevel = "UNKNOWN";
					break;
			}
			std::time_t t = std::time(0);
			std::tm*	now = std::localtime(&t);
			std::cout << "[" << logLevel << "][" << now->tm_year + 1900 << "-"
					  << now->tm_mon + 1 << "-" << now->tm_mday << " " << now->tm_hour
					  << ":" << now->tm_min << ":" << now->tm_sec << "][" << caller
					  << "]: ";
			((std::cout << args << " "), ...);
			std::cout << std::endl;

			// flush
			std::cout.flush();
			getMutex().unlock();
		}

		template <typename... Args>
		static void assertfx(bool condition, const char* caller, Args... args)
		{
			if (!condition)
			{
				log(4, caller, args...);
			}
		}
	};
#define ifritLog(level, ...)                                       \
	Ifrit::Graphics::SoftGraphics::Core::Utility::CoreLogger::log( \
		level, __FUNCTION__, __VA_ARGS__)
#define ifritLog1(...)                                             \
	Ifrit::Graphics::SoftGraphics::Core::Utility::CoreLogger::log( \
		0, __FUNCTION__, __VA_ARGS__)
#define ifritLog2(...)                                             \
	Ifrit::Graphics::SoftGraphics::Core::Utility::CoreLogger::log( \
		1, __FUNCTION__, __VA_ARGS__)
#define ifritLog3(...)                                             \
	Ifrit::Graphics::SoftGraphics::Core::Utility::CoreLogger::log( \
		2, __FUNCTION__, __VA_ARGS__)
#define ifritLog4(...)                                             \
	Ifrit::Graphics::SoftGraphics::Core::Utility::CoreLogger::log( \
		3, __FUNCTION__, __VA_ARGS__)
#define ifritLog5(...)                                             \
	Ifrit::Graphics::SoftGraphics::Core::Utility::CoreLogger::log( \
		4, __FUNCTION__, __VA_ARGS__)
#define ifritAssert(condition, ...)                                     \
	Ifrit::Graphics::SoftGraphics::Core::Utility::CoreLogger::assertfx( \
		condition, __FUNCTION__, __VA_ARGS__)
#define ifritError(...)                                            \
	Ifrit::Graphics::SoftGraphics::Core::Utility::CoreLogger::log( \
		3, __FUNCTION__, __VA_ARGS__);                             \
	std::abort();

} // namespace Ifrit::Graphics::SoftGraphics::Core::Utility