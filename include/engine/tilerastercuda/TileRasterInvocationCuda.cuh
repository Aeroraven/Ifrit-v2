#pragma once

#include "core/definition/CoreDefs.h"
#include "engine/base/TypeDescriptor.h"
#include "engine/tileraster/TileRasterCommon.h"
#include "engine/base/VertexShader.h"
#include "engine/base/FragmentShader.h"


namespace Ifrit::Engine::TileRaster::CUDA::Invocation::Impl {
	// Constants
	constexpr float CU_EPS = 1e-7f;

	// Procedures
	IFRIT_DEVICE float devEdgeFunction(ifloat4 a, ifloat4 b, ifloat4 c); //DONE
	IFRIT_DEVICE bool devTriangleCull(ifloat4 v1, ifloat4 v2, ifloat4 v3); //DONE
	IFRIT_DEVICE int devTriangleHomogeneousClip(const int primitiveId, ifloat4 v1, ifloat4 v2, ifloat4 v3,
		AssembledTriangleProposal** dProposals, uint32_t* dProposalCount, int* startingIdx); //DONE
	IFRIT_DEVICE bool devTriangleSimpleClip(ifloat4 v1, ifloat4 v2, ifloat4 v3, irect2Df& bbox); //DONE
	IFRIT_DEVICE void* devGetBufferAddress(char* dBuffer, TypeDescriptorEnum typeDesc, uint32_t element); //DONE

	IFRIT_DEVICE void devGetAcceptRejectCoords(ifloat3 edgeCoefs[3], int chosenCoordTR[3], int chosenCoordTA[3]); //DONE
	IFRIT_DEVICE void devExecuteBinner(
		int primitiveId,
		AssembledTriangleProposal& atp,
		irect2Df bbox,
		TileBinProposal** dRasterQueue,
		uint32_t* dRasterQueueCount,
		TileBinProposal** dCoverQueue,
		uint32_t* dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	); //DONE


	// Kernels

	IFRIT_KERNEL void testingKernel(); //DONE
	
	IFRIT_KERNEL void vertexProcessingKernel(
		VertexShader* vertexShader,
		char** dVertexBuffer,
		TypeDescriptorEnum* dVertexTypeDescriptor,
		char** dVaryingBuffer,
		TypeDescriptorEnum* dVaryingTypeDescriptor,
		ifloat4* dPosBuffer,
		TileRasterDeviceConstants* deviceConstants
	); //DONE

	IFRIT_KERNEL void geometryProcessingKernel(
		ifloat4* dPosBuffer,
		int* dIndexBuffer,
		AssembledTriangleProposal** dAssembledTriangles,
		uint32_t* dAssembledTriangleCount,
		TileBinProposal** dRasterQueue,
		uint32_t* dRasterQueueCount,
		TileBinProposal** dCoverQueue,
		uint32_t* dCoverQueueCount,
		TileRasterDeviceConstants* deviceConstants
	);//DONE


	// Dispatch Sub Kernels for a Single Tile Rasterization
	IFRIT_KERNEL void tilingRasterizationKernel(
		AssembledTriangleProposal** dAssembledTriangles,
		uint32_t* dAssembledTriangleCount,
		TileBinProposal** dRasterQueue,
		uint32_t* dRasterQueueCount,
		TileBinProposal** dCoverQueue,
		uint32_t* dCoverQueueCount,
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

namespace Ifrit::Engine::TileRaster::CUDA::Invocation {
	void testingKernelWrapper();
}