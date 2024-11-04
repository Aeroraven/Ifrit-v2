#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include "ifrit/softgraphics/core/definition/CoreDefs.h"
namespace Ifrit::GraphicsBackend::SoftGraphics::Core::Utility {
class CoreLogger {
private:
public:
  static std::mutex &getMutex() {
    static std::mutex logMutex;
    return logMutex;
  }
  template <typename... Args>
  static void log(int32_t level, const char *caller, Args... args) {
    getMutex().lock();
    std::string logLevel;
    switch (level) {
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
    std::tm *now = std::localtime(&t);
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
  static void assertfx(bool condition, const char *caller, Args... args) {
    if (!condition) {
      log(4, caller, args...);
    }
  }
};
#define ifritLog(level, ...)                                                   \
  Ifrit::GraphicsBackend::SoftGraphics::Core::Utility::CoreLogger::log(        \
      level, __FUNCTION__, __VA_ARGS__)
#define ifritLog1(...)                                                         \
  Ifrit::GraphicsBackend::SoftGraphics::Core::Utility::CoreLogger::log(        \
      0, __FUNCTION__, __VA_ARGS__)
#define ifritLog2(...)                                                         \
  Ifrit::GraphicsBackend::SoftGraphics::Core::Utility::CoreLogger::log(        \
      1, __FUNCTION__, __VA_ARGS__)
#define ifritLog3(...)                                                         \
  Ifrit::GraphicsBackend::SoftGraphics::Core::Utility::CoreLogger::log(        \
      2, __FUNCTION__, __VA_ARGS__)
#define ifritLog4(...)                                                         \
  Ifrit::GraphicsBackend::SoftGraphics::Core::Utility::CoreLogger::log(        \
      3, __FUNCTION__, __VA_ARGS__)
#define ifritLog5(...)                                                         \
  Ifrit::GraphicsBackend::SoftGraphics::Core::Utility::CoreLogger::log(        \
      4, __FUNCTION__, __VA_ARGS__)
#define ifritAssert(condition, ...)                                            \
  Ifrit::GraphicsBackend::SoftGraphics::Core::Utility::CoreLogger::assertfx(   \
      condition, __FUNCTION__, __VA_ARGS__)
#define ifritError(...)                                                        \
  Ifrit::GraphicsBackend::SoftGraphics::Core::Utility::CoreLogger::log(        \
      3, __FUNCTION__, __VA_ARGS__);                                           \
  std::abort();

} // namespace Ifrit::GraphicsBackend::SoftGraphics::Core::Utility