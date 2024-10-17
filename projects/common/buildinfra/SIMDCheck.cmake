# Deps Check / AVX2
set(OLD_CMAKE_REQUIRED_FLAG ${CMAKE_REQUIRED_FLAGS})
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(AVX2_FLAGS "/arch:AVX2")
else()
    set(AVX2_FLAGS "-mavx2")
endif()
set(CMAKE_REQUIRED_FLAGS ${AVX2_FLAGS})
set(CHECK_AVX2_SOURCE "
#include <immintrin.h>
int main() {
    __m256i vec = _mm256_set1_epi32(1);  // AVX2 intrinsic
    return 0;
}
")
check_c_source_compiles("${CHECK_AVX2_SOURCE}" IFRIT_ENABLE_SIMD_AVX2)
set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAG})
message(STATUS "[IFRIT/EnvCheck]: AVX2 Support - ${IFRIT_ENABLE_SIMD_AVX2}")

# Deps Check / SSE
set(OLD_CMAKE_REQUIRED_FLAG ${CMAKE_REQUIRED_FLAGS})
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	set(SSE_FLAGS "/arch:SSE")
else()
	set(SSE_FLAGS "-msse")
endif()
set(CMAKE_REQUIRED_FLAGS ${SSE_FLAGS})
set(CHECK_SSE_SOURCE "
#include <xmmintrin.h>
int main() {
    __m128 vec = _mm_set1_ps(1.0f);  // SSE intrinsic
    return 0;
}
")
check_c_source_compiles("${CHECK_SSE_SOURCE}" IFRIT_ENABLE_SIMD_SSE)
set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAG})
message(STATUS "[IFRIT/EnvCheck]: SSE Support - ${IFRIT_ENABLE_SIMD_SSE}")