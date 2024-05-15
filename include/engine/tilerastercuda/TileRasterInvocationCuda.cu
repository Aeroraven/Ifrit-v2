#pragma once

#include "core/definition/CoreDefs.h"
#include "engine/base/TypeDescriptor.h"
#include "engine/tileraster/TileRasterCommon.h"
#include "engine/base/VertexShader.h"
#include "engine/base/FragmentShader.h"
#include <cuda_runtime.h>


namespace Ifrit::Engine::TileRaster::CUDA::Invocation {

	IFRIT_DUAL bool devTriangleCull(float4 v1, float4 v2, float4 v3) {
		float d1 = (v1.x * v2.y);
		float d2 = (v2.x * v3.y);
		float d3 = (v3.x * v1.y);
		float n1 = (v3.x * v2.y);
		float n2 = (v1.x * v3.y);
		float n3 = (v2.x * v1.y);
		float d = d1 + d2 + d3 - n1 - n2 - n3;
		if (d < 0.0) return false;
		return true;
	}
	IFRIT_DUAL void* devGetBufferAddress(char* dBuffer, TypeDescriptorEnum typeDesc, uint32_t element) {
		if (typeDesc == TypeDescriptorEnum::IFTP_FLOAT1) {
			return reinterpret_cast<float*>(dBuffer) + element;
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_FLOAT2) {
			return reinterpret_cast<float2*>(dBuffer) + element;
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_FLOAT3) {
			return reinterpret_cast<float3*>(dBuffer) + element;
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_FLOAT4) {
			return reinterpret_cast<float4*>(dBuffer) + element;
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_INT1) {
			return reinterpret_cast<int*>(dBuffer) + element;
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_INT2) {
			return reinterpret_cast<int2*>(dBuffer) + element;
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_INT3) {
			return reinterpret_cast<int3*>(dBuffer) + element;
		}
		else if (typeDesc == TypeDescriptorEnum::IFTP_INT4) {
			return reinterpret_cast<int4*>(dBuffer) + element;
		}
		else {
			return nullptr;
		}
	}
	
	IFRIT_KERNEL void vertexProcessingKernel(
		VertexShader* vertexShader,
		char** dVertexBuffer,
		TypeDescriptorEnum* dVertexTypeDescriptor,
		char** dVaryingBuffer,
		TypeDescriptorEnum* dVaryingTypeDescriptor,
		float4* dPosBuffer,
		TileRasterDeviceConstants* deviceConstants
	) {
		const auto globalInvoIdx = blockIdx.x * blockDim.x + threadIdx.x;
		const auto numAttrs = deviceConstants->attributeCount;
		const auto numVaryings = deviceConstants->varyingCount;
		const void** vertexInputPtrs = new const void* [numAttrs];
		VaryingStore** varyingOutputPtrs = new VaryingStore * [numVaryings];
		for (int i = 0; i < numAttrs; i++) {
			vertexInputPtrs[i] = devGetBufferAddress(dVertexBuffer[i], dVertexTypeDescriptor[i], globalInvoIdx);
		}
		for (int i = 0; i < numVaryings; i++) {
			varyingOutputPtrs[i] = reinterpret_cast<VaryingStore*>(devGetBufferAddress(dVaryingBuffer[i], dVaryingTypeDescriptor[i], globalInvoIdx));
		}
		vertexShader->execute(vertexInputPtrs, &dPosBuffer[globalInvoIdx], varyingOutputPtrs);
		delete[] vertexInputPtrs;
		delete[] varyingOutputPtrs;
	}

	IFRIT_KERNEL void geometryProcessingKernel(
		float4* dPosBuffer,
		int* dIndexBuffer,
		AssembledTriangleProposal** dAssembledTriangles,
		uint32_t* dAssembledTriangleCount,
		TileBinProposal** dRasterQueue,
		uint32_t* dRasterQueueCount,
		TileBinProposal** dCoverQueue,
		uint32_t* dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	) {
		const auto globalInvoIdx = blockIdx.x * blockDim.x + threadIdx.x;
		const auto indexStart = globalInvoIdx * deviceConstants->vertexStride;
		float4 v1 = dPosBuffer[dIndexBuffer[indexStart]];
		float4 v2 = dPosBuffer[dIndexBuffer[indexStart + 1]];
		float4 v3 = dPosBuffer[dIndexBuffer[indexStart + 2]];
		if (deviceConstants->counterClockwise) {
			float4 temp = v1;
			v1 = v3;
			v3 = temp;
		}
	}


	IFRIT_KERNEL void tilingRasterizationKernel(
		AssembledTriangleProposal** dAssembledTriangles,
		uint32_t* dAssembledTriangleCount,
		TileBinProposal*** dRasterQueue,
		uint32_t** dRasterQueueCount,
		TileBinProposal*** dCoverQueue,
		TileRasterDeviceConstants* deviceConstants
	);

	IFRIT_KERNEL void fragmentShadingKernel(
		AssembledTriangleProposal** dAssembledTriangles,
		uint32_t* dAssembledTriangleCount,
		TileBinProposal*** dCoverQueue,
		float** dColorBuffer,
		float* dDepthBuffer,
		TileRasterDeviceConstants* deviceConstants
	);

	
}