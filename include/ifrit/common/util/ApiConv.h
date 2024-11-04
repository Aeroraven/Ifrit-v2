#pragma once

#ifdef WIN32
// Note that this will influence the C# binding
#define IFRIT_APICALL __stdcall
#else
#define IFRIT_APICALL
#endif

#if _WINDLL
#ifndef IFRIT_DLL
#define IFRIT_DLL
#endif
#ifndef IFRIT_API_EXPORT
#define IFRIT_API_EXPORT
#endif
#endif

// We guarantee that the same STL is used across the library
#ifdef _MSC_VER
#pragma warning(disable : 4251)
#endif

// Platform specific dllexport semantics
// https://stackoverflow.com/questions/2164827/explicitly-exporting-shared-library-functions-in-linux
#if defined(_MSC_VER)
#define IFRIT_DLLEXPORT __declspec(dllexport)
#define IFRIT_DLLIMPORT __declspec(dllimport)
#elif defined(__MINGW64__)
#define IFRIT_DLLEXPORT __declspec(dllexport)
#define IFRIT_DLLIMPORT __declspec(dllimport)
#elif defined(__clang__)
#define IFRIT_DLLEXPORT __attribute__((visibility("default")))
#define IFRIT_DLLIMPORT
#elif defined(__GNUC__)
#define IFRIT_DLLEXPORT __attribute__((visibility("default")))
#define IFRIT_DLLIMPORT
#else
static_assert(false, "Unsupported compiler");
#endif

// x64 & x32 platform detection
#if _WIN32 || _WIN64
#if _WIN64
#define IFRIT_ENV64
#else
#define IFRIT_ENV32
static_assert(false, "Lacking x32 support");
#endif
#elif __GNUC__
#if __x86_64__ || __ppc64__
#define IFRIT_ENV64
#else
#define IFRIT_ENV32
static_assert(false, "Lacking x32 support");
#endif
#endif

#ifdef IFRIT_DLL
#ifndef __cplusplus
#define IFRIT_API_EXPORT_COMPATIBLE_MODE
#endif // !__cplusplus

#ifdef IFRIT_API_EXPORT_COMPATIBLE_MODE
#ifdef IFRIT_API_EXPORT
#define IFRIT_APIDECL
#define IFRIT_APIDECL_IMPORT IFRIT_DLLIMPORT
#define IFRIT_APIDECL_FORCED IFRIT_DLLEXPORT
#define IFRIT_APIDECL_COMPAT extern "C" IFRIT_DLLEXPORT
#else
#define IFRIT_APIDECL
#define IFRIT_APIDECL_IMPORT IFRIT_DLLIMPORT
#define IFRIT_APIDECL_FORCED IFRIT_DLLIMPORT
#define IFRIT_APIDECL_COMPAT extern "C" IFRIT_DLLIMPORT
#define IRTIT_IGNORE_PRESENTATION_DEPS
#endif
#else
#ifdef IFRIT_API_EXPORT
#define IFRIT_APIDECL IFRIT_DLLEXPORT
#define IFRIT_APIDECL_IMPORT IFRIT_DLLIMPORT
#define IFRIT_APIDECL_FORCED IFRIT_DLLEXPORT
#define IFRIT_APIDECL_COMPAT extern "C" IFRIT_DLLEXPORT
#else
#define IFRIT_APIDECL IFRIT_DLLIMPORT
#define IFRIT_APIDECL_IMPORT IFRIT_DLLIMPORT
#define IFRIT_APIDECL_FORCED IFRIT_DLLIMPORT
#define IFRIT_APIDECL_COMPAT extern "C" IFRIT_DLLIMPORT
#define IRTIT_IGNORE_PRESENTATION_DEPS
#endif
#endif
#else
#define IFRIT_APIDECL_FORCED IFRIT_DLLEXPORT
#define IFRIT_APIDECL
#define IFRIT_APIDECL_IMPORT
#define IFRIT_APIDECL_COMPAT
#endif