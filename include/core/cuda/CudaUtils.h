#pragma once
#include "core/definition/CoreExports.h"

#ifdef IFRIT_FEATURE_CUDA
namespace Core::CUDA {
	template<typename T>
	__global__ void kernFixVTable(T* devicePtr) {
		T temp(*devicePtr);
		memcpy(devicePtr, &temp, sizeof(T));
	}
	template<typename T>
	__host__ T* hostGetDeviceObjectCopy(T* hostObject) {
		T* deviceHandle;
		cudaMalloc(&deviceHandle, sizeof(T));
		cudaMemcpy(deviceHandle, hostObject, sizeof(T), cudaMemcpyHostToDevice);
		kernFixVTable<T> CU_KARG2(1, 1)(deviceHandle);
		return deviceHandle;
	}
}
#define ifritCudaGetDeviceCopy()  Core::CUDA::hostGetDeviceObjectCopy(this);

#else

#define ifritCudaGetDeviceCopy()  nullptr

#endif