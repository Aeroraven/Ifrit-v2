#pragma once
#include "core/definition/CoreDefs.h"

#ifdef IFRIT_FEATURE_CUDA
namespace Ifrit::Core::CUDA {
	
	template<typename T>
	IFRIT_KERNEL void kernFixVTable(T* devicePtr) {
		T temp(*devicePtr);
		memcpy(devicePtr, &temp, sizeof(T));
	}

	template<typename T>
	IFRIT_HOST T* hostGetDeviceObjectCopy(T* hostObject) {
		T* deviceHandle;
		cudaMalloc(&deviceHandle, sizeof(T));
		cudaMemcpy(deviceHandle, hostObject, sizeof(T), cudaMemcpyHostToDevice);
		kernFixVTable<T> CU_KARG2(1, 1)(deviceHandle);
		cudaDeviceSynchronize();
		printf("Object copied to CUDA\n");
		return deviceHandle;
	}

	template<typename T>
	class DeviceMemoryAllocator {
	public:
		using value_type = T;
		using pointer = T*;
		using const_pointer = const T*;
		using void_pointer = void*;
		using const_void_pointer = const void*;
		using size_type = size_t;
		using difference_type = std::ptrdiff_t;

		T* allocate(size_t n) {
			T* ptr;
			cudaMalloc(&ptr, n * sizeof(T));
			return ptr;
		}

		void deallocate(T* ptr, size_t n) {
			cudaFree(ptr);
		}
	};

	template<typename T, typename Allocator = DeviceMemoryAllocator<T>>
	class DeviceVector {
	private:
		size_t vectorSize = 0;
		size_t vectorCapacity = 0;
		Allocator allocator = Allocator();
		T* vecData = nullptr;

	public:

		constexpr size_t size() const noexcept {
			return vectorSize;
		}
		constexpr size_t capacity() const noexcept {
			return vectorCapacity;
		}
		constexpr T* data() noexcept {
			return vecData;
		}
		constexpr const T* data() const noexcept {
			return vecData;
		}
		constexpr T& operator[](size_t idx) noexcept {
			return vecData[idx];
		}
		constexpr const T& operator[](size_t idx) const noexcept {
			return vecData[idx];
		}
		void reserve(size_t n) {
			if (n > vectorCapacity) {
				T* newPtr = allocator.allocate(n);
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
		void push_back(const T& val) {
			if (vectorSize >= vectorCapacity) {
				reserve(vectorCapacity * 2);
			}
			vecData[vectorSize++] = val;
		}
	};
}
#define ifritCudaGetDeviceCopy()  Core::CUDA::hostGetDeviceObjectCopy(this);

#else

#define ifritCudaGetDeviceCopy()  nullptr

#endif