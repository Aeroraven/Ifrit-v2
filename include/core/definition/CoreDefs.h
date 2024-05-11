#pragma once
#include <memory>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <optional>
#include <set>
#include <functional>
#include <fstream>
#include <ctime>
#include <regex>
#include <concepts>
#include <cmath>
#include <cstring>
#include <random>
#include <array>
#include <map>
#include <unordered_map>
#include <set>
#include <variant>
#include <unordered_set>
#include <chrono>
#include <ranges>
#include <typeinfo>
#include <sstream>
#include <deque>
#include <mutex>
#include <thread>
#include <any>

#ifdef _HAS_CXX23
	#define IFRIT_CXX23_ENABLED 1
#endif
#ifndef _HAS_CXX23
	#if __cplusplus >= 202302L
		#define IFRIT_CXX23_ENABLED 1
	#endif
#endif

#ifdef IFRIT_CXX23_ENABLED
#include <print>
#endif 

#ifndef __INTELLISENSE__
	#ifndef IFRIT_SHADER_PATH
		static_assert(false, "IFRIT_SHADER_PATH is not defined");
	#endif
#endif

#define IFRIT_VERBOSE_SAFETY_CHECK

#ifdef IFRIT_FEATURE_SIMD
#include <emmintrin.h>
#define IFRIT_USE_SIMD_128 1
#endif	