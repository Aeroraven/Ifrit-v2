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

namespace Ifrit::Logging {

template <typename... Args> void info(const char *fmt, Args &&...args) {
  spdlog::info(fmt, std::forward<Args>(args)...);
}

template <typename... Args> void warn(const char *fmt, Args &&...args) {
  spdlog::warn(fmt, std::forward<Args>(args)...);
}

template <typename... Args> void error(const char *fmt, Args &&...args) {
  spdlog::error(fmt, std::forward<Args>(args)...);
}

template <typename... Args> void debug(const char *fmt, Args &&...args) {
  spdlog::debug(fmt, std::forward<Args>(args)...);
}

template <typename... Args> void trace(const char *fmt, Args &&...args) {
  spdlog::trace(fmt, std::forward<Args>(args)...);
}

template <typename T> void info(const T &msg) { spdlog::info(msg); }

template <typename T> void warn(const T &msg) { spdlog::warn(msg); }

template <typename T> void error(const T &msg) { spdlog::error(msg); }

template <typename T> void debug(const T &msg) { spdlog::debug(msg); }

template <typename T> void trace(const T &msg) { spdlog::trace(msg); }

template <typename T> void assertion(bool condition, const T &msg) {
  if (!condition) {
    spdlog::error(msg);
    throw std::runtime_error(msg);
  }
}

// v2

inline void registerLoggerModule(const std::string &name) {
  spdlog::set_pattern("[%H:%M:%S %z] [%n] [%^%l%$] %v");
  auto stdoutSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto logger = std::make_shared<spdlog::logger>(name, stdoutSink);
  logger->set_pattern("[%Y/%m/%d %H:%M:%S %z] [%^%-7l%$] [%n] %v");
  logger->set_level(spdlog::level::trace);
  spdlog::register_logger(logger);
}

inline std::shared_ptr<spdlog::logger> getLoggerModule(const std::string &name) {
  auto logger = spdlog::get(name);
  if (!logger) {
    registerLoggerModule(name);
    return spdlog::get(name);
  }
  return logger;
}

template <typename... Args> inline void info2(const char *moduleName, std::format_string<Args...> fmt, Args &&...args) {
  auto formatted = std::format(fmt, std::forward<Args>(args)...);
  getLoggerModule(moduleName)->info(formatted);
}

template <typename... Args> inline void warn2(const char *moduleName, std::format_string<Args...> fmt, Args &&...args) {
  auto formatted = std::format(fmt, std::forward<Args>(args)...);
  getLoggerModule(moduleName)->warn(formatted);
}

template <typename... Args>
inline void error2(const char *moduleName, std::format_string<Args...> fmt, Args &&...args) {
  auto formatted = std::format(fmt, std::forward<Args>(args)...);
  getLoggerModule(moduleName)->error(formatted);
}

template <typename... Args>
inline void debug2(const char *moduleName, std::format_string<Args...> fmt, Args &&...args) {
  auto formatted = std::format(fmt, std::forward<Args>(args)...);
  getLoggerModule(moduleName)->debug(formatted);
}

template <typename... Args>
inline void trace2(const char *moduleName, std::format_string<Args...> fmt, Args &&...args) {
  auto formatted = std::format(fmt, std::forward<Args>(args)...);
  getLoggerModule(moduleName)->trace(formatted);
}

template <typename T> inline void info2(const char *moduleName, const T &msg) {
  auto s = getLoggerModule(moduleName);
  s->info(msg);
}

template <typename T> inline void warn2(const char *moduleName, const T &msg) {
  getLoggerModule(moduleName)->warn(msg);
}

template <typename T> inline void error2(const char *moduleName, const T &msg) {
  getLoggerModule(moduleName)->error(msg);
}

template <typename T> inline void debug2(const char *moduleName, const T &msg) {
  getLoggerModule(moduleName)->debug(msg);
}

template <typename T> inline void trace2(const char *moduleName, const T &msg) {
  getLoggerModule(moduleName)->trace(msg);
}

template <typename T> inline void assertion2(const char *moduleName, bool condition, const T &msg) {
  if (!condition) {
    getLoggerModule(moduleName)->error(msg);
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
#define iInfo(...) Logging::info2(IFRIT_LOG_MODULE_NAME, __VA_ARGS__)
#define iWarn(...) Logging::warn2(IFRIT_LOG_MODULE_NAME, __VA_ARGS__)
#define iError(...) Logging::error2(IFRIT_LOG_MODULE_NAME, __VA_ARGS__)
#define iDebug(...) Logging::debug2(IFRIT_LOG_MODULE_NAME, __VA_ARGS__)
#define iTrace(...) Logging::trace2(IFRIT_LOG_MODULE_NAME, __VA_ARGS__)
#define iAssertion(condition, ...) Logging::assertion2(IFRIT_LOG_MODULE_NAME, condition, __VA_ARGS__)
#else
static_assert(false, "IFRIT_LOG_MODULE_NAME is not defined");
#endif
} // namespace Ifrit::Logging