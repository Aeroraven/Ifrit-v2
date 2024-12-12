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

} // namespace Ifrit::Logging