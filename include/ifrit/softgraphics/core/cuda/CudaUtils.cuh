
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
#include "ifrit/softgraphics/core/definition/CoreDefs.h"

#ifdef IFRIT_FEATURE_CUDA
namespace Ifrit::SoftRenderer::Core::CUDA {

template <typename T> IFRIT_KERNEL void kernFixVTable(T *devicePtr) {
  T temp(*devicePtr);
  memcpy(devicePtr, &temp, sizeof(T));
}

template <typename T> IFRIT_HOST T *hostGetDeviceObjectCopy(T *hostObject) {
  T *deviceHandle;
  cudaMalloc(&deviceHandle, sizeof(T));
  cudaMemcpy(deviceHandle, hostObject, sizeof(T), cudaMemcpyHostToDevice);
  kernFixVTable<T> CU_KARG2(1, 1)(deviceHandle);
  cudaDeviceSynchronize();
  return deviceHandle;
}

template <typename T> class DeviceMemoryAllocator {
public:
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using void_pointer = void *;
  using const_void_pointer = const void *;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  T *allocate(size_t n) {
    T *ptr;
    cudaMalloc(&ptr, n * sizeof(T));
    return ptr;
  }

  void deallocate(T *ptr, size_t n) { cudaFree(ptr); }
};

template <typename T, typename Allocator = DeviceMemoryAllocator<T>>
class DeviceVector {
private:
  size_t vectorSize = 0;
  size_t vectorCapacity = 0;
  Allocator allocator = Allocator();
  T *vecData = nullptr;

public:
  IF_CONSTEXPR size_t size() const noexcept { return vectorSize; }
  IF_CONSTEXPR size_t capacity() const noexcept { return vectorCapacity; }
  IF_CONSTEXPR T *data() noexcept { return vecData; }
  IF_CONSTEXPR const T *data() const noexcept { return vecData; }
  IF_CONSTEXPR T &operator[](size_t idx) noexcept { return vecData[idx]; }
  IF_CONSTEXPR const T &operator[](size_t idx) const noexcept {
    return vecData[idx];
  }
  void reserve(size_t n) {
    if (n > vectorCapacity) {
      T *newPtr = allocator.allocate(n);
      memcpy(newPtr, vecData, vectorSize * sizeof(T));
      allocator.deallocate(vecData, vectorCapacity);
      vecData = newPtr;
      vectorCapacity = n;
    }
  }
  void resize(size_t n) {
    if (n > vectorCapacity) {
      reserve(n);
    }
    vectorSize = n;
  }
  void push_back(const T &val) {
    if (vectorSize >= vectorCapacity) {
      reserve(vectorCapacity * 2);
    }
    vecData[vectorSize++] = val;
  }
};
} // namespace Ifrit::SoftRenderer::Core::CUDA
#define ifritCudaGetDeviceCopy() Core::CUDA::hostGetDeviceObjectCopy(this);

#else

#define ifritCudaGetDeviceCopy() nullptr

#endif