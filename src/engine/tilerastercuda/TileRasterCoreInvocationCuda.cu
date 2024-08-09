#ifdef IFRIT_FEATURE_CUDA
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"
#include "engine/math/ShaderOpsCuda.cuh"
#include "engine/tilerastercuda/TileRasterDeviceContextCuda.cuh"
#include "engine/tilerastercuda/TileRasterConstantsCuda.h"
#include "engine/base/Structures.h"
#include <cuda_profiler_api.h>
#include "engine/tilerastercuda/TileRasterCommonResourceCuda.cuh"

#define IFRIT_InvoGetThreadBlocks(tasks,blockSize) ((tasks)/(blockSize))+((tasks) % (blockSize) != 0)

namespace Ifrit::Engine::TileRaster::CUDA::Invocation::Impl {
	struct ImplBlendCoefs {
		int4 s; //SrcColor (x1/x[src.alpha]/x[dst.alpha]/x0)
		int4 d; //DstColor (x1/x[src.alpha]/x[dst.alpha]/x0)
	};

	IFRIT_DEVICE_CONST static IfritColorAttachmentBlendState csBlendState;
	IFRIT_DEVICE_CONST static int csFrameWidth = 0;
	IFRIT_DEVICE_CONST static int csFrameHeight = 0;
	IFRIT_DEVICE_CONST static float csFrameWidthInv = 0;
	IFRIT_DEVICE_CONST static float csFrameHeightInv = 0;
	IFRIT_DEVICE_CONST static bool csCounterClosewiseCull = false;
	IFRIT_DEVICE_CONST static int csVertexOffsets[CU_MAX_ATTRIBUTES];
	IFRIT_DEVICE_CONST static int csTotalVertexOffsets = 0;
	IFRIT_DEVICE_CONST static int csVaryingCounts = 0;
	IFRIT_DEVICE_CONST static int csAttributeCounts = 0;
	IFRIT_DEVICE_CONST static int csVertexCount = 0;
	IFRIT_DEVICE_CONST static int csTotalIndices = 0;

	IFRIT_DEVICE_CONST float* csTextures[CU_MAX_TEXTURE_SLOTS];
	IFRIT_DEVICE_CONST int csTextureWidth[CU_MAX_TEXTURE_SLOTS];
	IFRIT_DEVICE_CONST int csTextureHeight[CU_MAX_TEXTURE_SLOTS];
	IFRIT_DEVICE_CONST int csTextureMipLevels[CU_MAX_TEXTURE_SLOTS];
	IFRIT_DEVICE_CONST int csTextureArrayLayers[CU_MAX_TEXTURE_SLOTS];
	IFRIT_DEVICE_CONST IfritSamplerT csSamplers[CU_MAX_SAMPLER_SLOTS];
	IFRIT_DEVICE_CONST char* csGeneralBuffer[CU_MAX_BUFFER_SLOTS];
	IFRIT_DEVICE_CONST int csGeneralBufferSize[CU_MAX_BUFFER_SLOTS];

	static IfritColorAttachmentBlendState hsBlendState;
	static int hsFrameWidth = 0;
	static int hsFrameHeight = 0;
	static bool hsCounterClosewiseCull = false;
	static int hsVertexOffsets[CU_MAX_ATTRIBUTES];
	static int hsTotalVertexOffsets = 0;
	static int hsVaryingCounts = 0;
	static int hsAttributeCounts = 0;
	static int hsVertexCounts = 0;
	static int hsTotalIndices = 0;

	float* hsTextures[CU_MAX_TEXTURE_SLOTS];
	int hsTextureWidth[CU_MAX_TEXTURE_SLOTS];
	int hsTextureHeight[CU_MAX_TEXTURE_SLOTS];
	int hsTextureMipLevels[CU_MAX_TEXTURE_SLOTS];
	int hsTextureArrayLayers[CU_MAX_TEXTURE_SLOTS];
	IfritSamplerT hsSampler[CU_MAX_SAMPLER_SLOTS];
	char* hsGeneralBuffer[CU_MAX_BUFFER_SLOTS];
	int hsGeneralBufferSize[CU_MAX_BUFFER_SLOTS];
	
	IFRIT_DEVICE static uint32_t dCoverQueueFullM2Buffer[CU_ALTERNATIVE_BUFFER_SIZE];
	IFRIT_DEVICE static uint64_t dCoverQueueFullM2SortKeys[CU_ALTERNATIVE_BUFFER_SIZE];
	IFRIT_DEVICE static int dCoverQueueFullM2Size[CU_MAX_BIN_X * CU_MAX_BIN_X];
	IFRIT_DEVICE static int dCoverQueueFullM2Start[CU_MAX_BIN_X * CU_MAX_BIN_X];
	IFRIT_DEVICE static int dCoverQueueFullM2CurInd[CU_MAX_BIN_X * CU_MAX_BIN_X];
	IFRIT_DEVICE static int dCoverQueueFullM2GlobalSize;
	
	IFRIT_DEVICE static int dSecondBinnerFinerBufferSize[CU_MAX_TILE_X * CU_MAX_TILE_X * CU_MAX_SUBTILES_PER_TILE];
	IFRIT_DEVICE static int dSecondBinnerFinerBufferStart[CU_MAX_TILE_X * CU_MAX_TILE_X * CU_MAX_SUBTILES_PER_TILE];
	IFRIT_DEVICE static int dSecondBinnerFinerBufferCurInd[CU_MAX_TILE_X * CU_MAX_TILE_X * CU_MAX_SUBTILES_PER_TILE];
	IFRIT_DEVICE static int2 dSecondBinnerFinerBuffer[CU_ALTERNATIVE_BUFFER_SIZE_SECOND];
	IFRIT_DEVICE static uint64_t dSecondBinnerFinerBufferSortKeys[CU_ALTERNATIVE_BUFFER_SIZE_SECOND];
	IFRIT_DEVICE static uint64_t dSecondBinnerFinerBufferSortKeysAltern[CU_ALTERNATIVE_BUFFER_SIZE_SECOND];

	IFRIT_DEVICE static int dSecondBinnerFinerBufferGlobalSize;

	IFRIT_DEVICE static int dSecondBinnerFinerBufferSizePoint[CU_MAX_FRAMEBUFFER_SIZE];
	IFRIT_DEVICE static int dSecondBinnerFinerBufferStartPoint[CU_MAX_FRAMEBUFFER_SIZE];
	IFRIT_DEVICE static int dSecondBinnerFinerBufferCurIndPoint[CU_MAX_FRAMEBUFFER_SIZE];
	IFRIT_DEVICE static int dSecondBinnerFinerBufferPoint[CU_ALTERNATIVE_BUFFER_SIZE_SECOND];

	IFRIT_DEVICE static int dCoverQueueSuperTileFullM3Size[CU_MAX_LARGE_BIN_X * CU_MAX_LARGE_BIN_X];
	IFRIT_DEVICE static int dCoverQueueSuperTileFullM3Start[CU_MAX_LARGE_BIN_X * CU_MAX_LARGE_BIN_X];
	IFRIT_DEVICE static int dCoverQueueSuperTileFullM3CurInd[CU_MAX_LARGE_BIN_X * CU_MAX_LARGE_BIN_X];
	IFRIT_DEVICE static int2 dCoverQueueSuperTileFullM3Buffer[CU_ALTERNATIVE_BUFFER_SIZE];
	IFRIT_DEVICE static int dCoverQueueSuperTileFullM3BufferFinal[CU_ALTERNATIVE_BUFFER_SIZE];
	IFRIT_DEVICE static int dCoverQueueSuperTileFullM3BufferFinalSortKeys[CU_ALTERNATIVE_BUFFER_SIZE];
	IFRIT_DEVICE static int dCoverQueueSuperTileFullM3GlobalSize;
	IFRIT_DEVICE static int dCoverQueueSuperTileFullM3TotlCands;

	IFRIT_DEVICE static uint2 dRasterQueueWorklistPrimTile[CU_MAX_BIN_X * 2 * CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER];
	
	IFRIT_DEVICE static uint32_t dRasterQueueWorklistCounter;

	IFRIT_DEVICE static float4 dBoundingBoxM2[CU_PRIMITIVE_BUFFER_SIZE * 2];
	IFRIT_DEVICE static float4 dAtriEdgeCoefs1[CU_PRIMITIVE_BUFFER_SIZE * 2];
	IFRIT_DEVICE static float4 dAtriEdgeCoefs2[CU_PRIMITIVE_BUFFER_SIZE * 2];
	IFRIT_DEVICE static float dAtriEdgeCoefs3[CU_PRIMITIVE_BUFFER_SIZE * 2];
	IFRIT_DEVICE static float4 dAtriInterpolBase1[CU_PRIMITIVE_BUFFER_SIZE * 2];
	IFRIT_DEVICE static float4 dAtriInterpolBase2[CU_PRIMITIVE_BUFFER_SIZE * 2];
	IFRIT_DEVICE static float dAtriInterpolBase3[CU_PRIMITIVE_BUFFER_SIZE * 2];

	IFRIT_DEVICE static float4 dAtriBaryCenter12[CU_PRIMITIVE_BUFFER_SIZE * 2];
	IFRIT_DEVICE static float2 dAtriBaryCenter3[CU_PRIMITIVE_BUFFER_SIZE * 2];

	IFRIT_DEVICE static float2 dAtriDepthVal12[CU_PRIMITIVE_BUFFER_SIZE * 2];
	IFRIT_DEVICE static float dAtriDepthVal3[CU_PRIMITIVE_BUFFER_SIZE * 2];
	IFRIT_DEVICE static int dAtriOriginalPrimId[CU_PRIMITIVE_BUFFER_SIZE * 2];
	IFRIT_DEVICE static int dAtriOriginalPrimIdGs[CU_PRIMITIVE_BUFFER_SIZE * 2];
	IFRIT_DEVICE static int dAtriPixelBelong[CU_PRIMITIVE_BUFFER_SIZE * 2];

	IFRIT_DEVICE static int2 dCoverPrimsTile[CU_ALTERNATIVE_BUFFER_SIZE];
	IFRIT_DEVICE static int2 dCoverPrimsTileLarge[CU_ALTERNATIVE_BUFFER_SIZE];

	IFRIT_DEVICE static int2 dSecondBinnerPendingPrim[CU_ALTERNATIVE_BUFFER_SIZE_SECOND];

	IFRIT_DEVICE static int dCoverPrimsCounter;
	IFRIT_DEVICE static int dCoverPrimsLargeTileCounter;
	IFRIT_DEVICE static int dSecondBinnerCandCounter;

	IFRIT_DEVICE static int dFragmentShadingPrimId[CU_MAX_FRAMEBUFFER_SIZE];
	IFRIT_DEVICE static uint32_t dAssembledTriangleCounterM2;

	// Mesh Shader
	
	IFRIT_DEVICE static float4 dMeshShaderPosition[CU_MESHSHADER_BUFFER_SIZE];
	IFRIT_DEVICE static VaryingStore dMeshShaderVaryings[CU_MESHSHADER_BUFFER_SIZE];
	IFRIT_DEVICE static int dMeshShaderIndices[CU_MESHSHADER_BUFFER_SIZE];
	IFRIT_DEVICE static int dMeshShaderNumVertices[CU_MESHSHADER_BUFFER_SIZE];
	IFRIT_DEVICE static int dMeshShaderNumIndices[CU_MESHSHADER_BUFFER_SIZE];
	
	IFRIT_DEVICE static float4 dMeshShaderPositionAfter[CU_MESHSHADER_BUFFER_SIZE];
	IFRIT_DEVICE static VaryingStore dMeshShaderVaryingsAfter[CU_MESHSHADER_BUFFER_SIZE];
	IFRIT_DEVICE static int dMeshShaderIndicesAfter[CU_MESHSHADER_BUFFER_SIZE];
	IFRIT_DEVICE static int dMeshShaderAggOffsetVert[CU_MESHSHADER_BUFFER_SIZE];
	IFRIT_DEVICE static int dMeshShaderAggOffsetVertStart[CU_MESHSHADER_BUFFER_SIZE];
	IFRIT_DEVICE static int dMeshShaderAggOffsetVertGlobal;
	IFRIT_DEVICE static int dMeshShaderAggOffsetIndex[CU_MESHSHADER_BUFFER_SIZE];
	IFRIT_DEVICE static int dMeshShaderAggOffsetIndexStart[CU_MESHSHADER_BUFFER_SIZE];
	IFRIT_DEVICE static int dMeshShaderAggOffsetIndexGlobal;


	// Depth Test
	IFRIT_DEVICE static bool dDepthTestEnable;
	IFRIT_DEVICE static IfritCompareOp csDepthFunc;

	// Culling
	IFRIT_DEVICE static IfritCullMode csCullMode;

	// Alpha Blending
	IFRIT_DEVICE static ImplBlendCoefs dGlobalBlendCoefs;
	IFRIT_DEVICE static ImplBlendCoefs dGlobalBlendCoefsAlpha;

	// Line Rasterization
	IFRIT_DEVICE static int2 dLineRasterOut[CU_MAX_FRAMEBUFFER_SIZE];
	IFRIT_DEVICE static int dLineRasterOutSize;
	
	// Geometry Shader
	IFRIT_DEVICE static int dGeometryShaderOutSize;
	IFRIT_DEVICE static float4 dGeometryShaderOutPos[CU_GS_OUT_BUFFER_SIZE];
	IFRIT_DEVICE static float4 dGeometryShaderOutVaryings[CU_GS_OUT_BUFFER_SIZE * CU_MAX_VARYINGS];
	IFRIT_DEVICE static int dGeometryShaderOutSortKeys[CU_GS_OUT_BUFFER_SIZE];

	// Pixel Shader Tags
	IFRIT_DEVICE static int dPixelShaderTags[CU_MAX_FRAMEBUFFER_SIZE];

	// Profiler: Overdraw
	IFRIT_DEVICE static int dOverDrawCounter;
	IFRIT_DEVICE static int dOverZTestCounter;
	IFRIT_DEVICE static int dIrrelevantPixel;
	IFRIT_DEVICE static int dTotalBinnedPixel;
	IFRIT_DEVICE static int dSingleBinnedPixel;

	// Profiler: Second Binner Utilizations
	IFRIT_DEVICE static int dSecondBinnerTotalReqs;
	IFRIT_DEVICE static int dSecondBinnerActiveReqs;
	IFRIT_DEVICE static int dSecondBinnerEmptyTiles;

	// Profiler: Tiny Triangle Efficiency
	IFRIT_DEVICE static int dSmallTriangleCount = 0;
	IFRIT_DEVICE static int dSmallTriangleThreadUtil = 0;
	IFRIT_DEVICE static int dSmallTriangleThreadUtilTotal = 0;

	// Profiler: Second Binner Thread Divergence
	IFRIT_DEVICE static int dSbtdSplit = 0;
	IFRIT_DEVICE static int dLock = 0;
	IFRIT_DEVICE static int dLockAt = 0;

	namespace GeneralFunction {
		IFRIT_DEVICE float devAbort() {
			printf("Kernel aborted\n");
			asm("trap;");
		}

		IFRIT_DEVICE float devEdgeFunction(float4 a, float4 b, float4 c) {
			return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
		}
		IFRIT_DEVICE bool devTriangleCull(float4 v1, float4 v2, float4 v3) {
			float d1 = (v1.x * (v2.y - v3.y));
			float d2 = (v2.x * (v3.y - v1.y));
			float d3 = (v3.x * (v1.y - v2.y));
			return (d1 + d2 + d3) >= 0.0f;
		}

		IFRIT_DEVICE bool devViewSpaceClip(float4 v1, float4 v2, float4 v3) {
			auto maxX = max(v1.x, max(v2.x, v3.x));
			auto maxY = max(v1.y, max(v2.y, v3.y));
			auto minX = min(v1.x, min(v2.x, v3.x));
			auto minY = min(v1.y, min(v2.y, v3.y));
			auto isIllegal = maxX < 0.0f || maxY < 0.0f || minX > 1.0f || minY > 1.0f;
			return isIllegal;
		}

		IFRIT_DEVICE void devGetAcceptRejectCoords(float4 edgeCoefs[3], int chosenCoordTR[3], int chosenCoordTA[3]) {
			constexpr const int VLB = 0, VLT = 1, VRT = 3, VRB = 2;
			for (int i = 0; i < 3; i++) {
				bool normalRight = edgeCoefs[i].x < 0;
				bool normalDown = edgeCoefs[i].y < 0;
				if (normalRight) {
					if (normalDown) {
						chosenCoordTR[i] = VRB;
						chosenCoordTA[i] = VLT;
					}
					else {
						chosenCoordTR[i] = VRT;
						chosenCoordTA[i] = VLB;
					}
				}
				else {
					if (normalDown) {
						chosenCoordTR[i] = VLB;
						chosenCoordTA[i] = VRT;
					}
					else {
						chosenCoordTR[i] = VLT;
						chosenCoordTA[i] = VRB;
					}
				}
			}
		}

		IFRIT_DEVICE bool devGetBBox(const float4& v1, const float4& v2, const float4& v3, float4& bbox) {
			float minx = min(v1.x, min(v2.x, v3.x));
			float miny = min(v1.y, min(v2.y, v3.y));
			float maxx = max(v1.x, max(v2.x, v3.x));
			float maxy = max(v1.y, max(v2.y, v3.y));
			bbox.x = minx;
			bbox.y = miny;
			bbox.z = maxx;
			bbox.w = maxy;
			return true;
		}
	}

	// Kernel Implementations
	namespace VertexStage {
		IFRIT_KERNEL void vertexProcessingKernel(
			VertexShader* vertexShader,
			uint32_t vertexCount,
			char* IFRIT_RESTRICT_CUDA dVertexBuffer,
			TypeDescriptorEnum* IFRIT_RESTRICT_CUDA dVertexTypeDescriptor,
			float4* IFRIT_RESTRICT_CUDA dVaryingBuffer,
			float4* IFRIT_RESTRICT_CUDA dPosBuffer
		) {
			const auto globalInvoIdx = blockIdx.x * blockDim.x + threadIdx.x;
			if (globalInvoIdx >= vertexCount) return;
			const auto numAttrs = csAttributeCounts;
			const auto numVaryings = csVaryingCounts;

			const void* vertexInputPtrs[CU_MAX_ATTRIBUTES];
			float4* varyingOutputPtrs[CU_MAX_VARYINGS];

			for (int i = 0; i < numAttrs; i++) {
				vertexInputPtrs[i] = globalInvoIdx * csTotalVertexOffsets + dVertexBuffer + csVertexOffsets[i];
			}
			for (int i = 0; i < numVaryings; i++) {
				varyingOutputPtrs[i] = dVaryingBuffer + globalInvoIdx * numVaryings + i;
			}
			vertexShader->execute(vertexInputPtrs, (ifloat4*)&dPosBuffer[globalInvoIdx], (VaryingStore**)varyingOutputPtrs);
		}
	}
	namespace MeshShadingStage {
		IFRIT_KERNEL void meshShadingKernel(
			int numPerVertexAttributes,
			MeshShader* meshShader
		) {
			iint3 localInvocation = { threadIdx.x,threadIdx.y,threadIdx.z };
			int workGroupId = blockIdx.x;

			int* pNumIndices = &(dMeshShaderNumIndices[workGroupId]);
			int* pNumVertices = &(dMeshShaderNumVertices[workGroupId]);
			VaryingStore* pOutVaryings = &(dMeshShaderVaryings[workGroupId * numPerVertexAttributes * CU_MESHSHADER_MAX_VERTICES]);
			float4* pOutPos = &(dMeshShaderPosition[workGroupId * CU_MESHSHADER_MAX_VERTICES]);
			int* pOutIndex = &(dMeshShaderIndices[workGroupId * CU_MESHSHADER_MAX_INDICES]);
			
			ifloat4* pOutPosR = reinterpret_cast<ifloat4*>(pOutPos);
			meshShader->execute(localInvocation, workGroupId, pOutVaryings, pOutPosR, pOutIndex, *pNumVertices, *pNumIndices);
		}

		IFRIT_KERNEL void meshShadingCompactionKernel(int totalWorkGroups, int numPerVertexAttributes) {
			int workGroupId = threadIdx.x + blockDim.x * blockIdx.x;
			if (workGroupId >= totalWorkGroups)return;
			int locPosBuffer = atomicAdd(&dMeshShaderAggOffsetVertGlobal, dMeshShaderNumVertices[workGroupId]);
			int locIndBuffer = atomicAdd(&dMeshShaderAggOffsetIndexGlobal, dMeshShaderNumIndices[workGroupId]);
			dMeshShaderAggOffsetVertStart[workGroupId] = locPosBuffer;
			dMeshShaderAggOffsetIndexStart[workGroupId] = locIndBuffer;
			for (int i = 0; i < dMeshShaderNumVertices[workGroupId]; i++) {
				dMeshShaderPositionAfter[i + locPosBuffer] = dMeshShaderPosition[workGroupId * CU_MESHSHADER_MAX_VERTICES + i];
			}
			for (int i = 0; i < dMeshShaderNumVertices[workGroupId]* numPerVertexAttributes; i++) {
				dMeshShaderVaryingsAfter[i + locPosBuffer * numPerVertexAttributes] = dMeshShaderVaryings[workGroupId * CU_MESHSHADER_MAX_VERTICES * numPerVertexAttributes + i];
			}
			for (int i = 0; i < dMeshShaderNumIndices[workGroupId]; i++) {
				dMeshShaderIndicesAfter[i + locIndBuffer] = dMeshShaderIndices[workGroupId * CU_MESHSHADER_MAX_INDICES + i] + locPosBuffer;
			}
		}

		IFRIT_KERNEL void meshShadingResetKernel() {
			dMeshShaderAggOffsetVertGlobal = 0;
			dMeshShaderAggOffsetIndexGlobal = 0;
		}

		IFRIT_KERNEL void meshShadingEntryPoint(
			iint3 workGroupSize,
			int totalWorkGroups,
			MeshShader* meshShader,
			int numAttributes
		) {
			meshShadingResetKernel CU_KARG2(1, 1) ();
			meshShadingKernel CU_KARG2(dim3(totalWorkGroups, 1, 1), dim3(workGroupSize.x, workGroupSize.y, workGroupSize.z))(numAttributes, meshShader);
			int dispatchBlocks = IFRIT_InvoGetThreadBlocks(totalWorkGroups, 32);
			meshShadingCompactionKernel CU_KARG2(dispatchBlocks, 32)(totalWorkGroups, numAttributes);
		}
	}
	namespace GeneralGeometryStage {
		IFRIT_KERNEL void geometryShadingKernel(
			uint32_t startingIndexId,
			uint32_t indexCount,
			GeometryShader* geometryShader,
			float4* IFRIT_RESTRICT_CUDA dPosBuffer,
			int* IFRIT_RESTRICT_CUDA dIndexBuffer,
			const float4* IFRIT_RESTRICT_CUDA dVaryingBuffer
		) {
			const auto globalInvoIdx = blockIdx.x * blockDim.x + threadIdx.x;
			if (globalInvoIdx >= indexCount / CU_TRIANGLE_STRIDE) return;
			const auto indexStart = globalInvoIdx * CU_TRIANGLE_STRIDE + startingIndexId;
			auto dPosBufferAligned = reinterpret_cast<float4*>(__builtin_assume_aligned(dPosBuffer, 16));
			float4 v1, v2, v3;

			//TODO: Bank Conflict
			IFRIT_SHARED int outVertices[CU_GEOMETRY_SHADER_THREADS];
			IFRIT_SHARED float4* posPtr[3 * CU_GEOMETRY_SHADER_THREADS];
			IFRIT_SHARED const float4* varyingPtr[3 * CU_MAX_VARYINGS * CU_GEOMETRY_SHADER_THREADS];
			IFRIT_SHARED float4 posOut[CU_MAX_GS_OUT_VERTICES * CU_GEOMETRY_PROCESSING_THREADS];
			IFRIT_SHARED float4 varyingOut[CU_MAX_GS_OUT_VERTICES * CU_MAX_VARYINGS * CU_GEOMETRY_SHADER_THREADS];
			int varyingSpv = csVaryingCounts;

			posPtr[threadIdx.x * 3 + 0] = dPosBufferAligned + dIndexBuffer[indexStart];
			posPtr[threadIdx.x * 3 + 1] = dPosBufferAligned + dIndexBuffer[indexStart + 1];
			posPtr[threadIdx.x * 3 + 2] = dPosBufferAligned + dIndexBuffer[indexStart + 2];
			varyingPtr[threadIdx.x * 3 + 0] = dVaryingBuffer + dIndexBuffer[indexStart] * csVaryingCounts;
			varyingPtr[threadIdx.x * 3 + 1] = dVaryingBuffer + dIndexBuffer[indexStart + 1] * csVaryingCounts;
			varyingPtr[threadIdx.x * 3 + 2] = dVaryingBuffer + dIndexBuffer[indexStart + 2] * csVaryingCounts;

			geometryShader->execute(
				(ifloat4**)(posPtr + 3 * threadIdx.x),
				(VaryingStore**)(varyingPtr + 3 * threadIdx.x),
				(ifloat4*)(posOut + 3 * threadIdx.x),
				(VaryingStore*)(varyingOut + 3 * threadIdx.x * csVaryingCounts),
				outVertices + threadIdx.x
			);
			int outW = outVertices[threadIdx.x];
			int insPos = atomicAdd(&dGeometryShaderOutSize, outW);
			for (int i = 0; i < outW; i++) {
				dGeometryShaderOutPos[insPos + i] = (posOut + 3 * threadIdx.x)[i];
				if constexpr (CU_OPT_FORCE_DETERMINISTIC_BEHAVIOR) {
					dGeometryShaderOutSortKeys[insPos + i] = indexStart * CU_MAX_GS_OUT_VERTICES + i;
				}
				for (int j = 0; j < csVaryingCounts; j++) {
					dGeometryShaderOutVaryings[insPos * csVaryingCounts + i * csVaryingCounts + j] = (varyingOut + 3 * threadIdx.x * csVaryingCounts)[i * csVaryingCounts + j];
				}
			}
		}
		IFRIT_DEVICE void devCallGeometryShader(
			GeometryShader* dGeometryShader,
			int stride,
			int startOffset,
			int indexCount,
			float4* IFRIT_RESTRICT_CUDA dPositionBuffer,
			int* IFRIT_RESTRICT_CUDA dIndexBuffer,
			float4* IFRIT_RESTRICT_CUDA dVaryingBuffer
		) {
			int geometryShaderBlocks = IFRIT_InvoGetThreadBlocks(indexCount / CU_TRIANGLE_STRIDE, CU_GEOMETRY_SHADER_THREADS);
			GeneralGeometryStage::geometryShadingKernel CU_KARG2(geometryShaderBlocks, CU_GEOMETRY_SHADER_THREADS) (startOffset, indexCount, dGeometryShader, (float4*)dPositionBuffer,
				dIndexBuffer, dVaryingBuffer);
		}
	}
	
	namespace TriangleSortStage {
		template<typename T>
		IFRIT_DEVICE void devSwap(T& x, T& y) {
			T temp = x;
			x = y;
			y = temp;
		}

		template<typename T,typename U>
		IFRIT_DEVICE void devHeapSortByKeyImpl(T* keys, U* values, int count) {
			if (count == 0)return;
			auto siftDown = [&](int st, int ed) {
				auto ch = 0;
				while ((ch = (st << 1) + 1) < ed) {
					if (ch + 1 < ed && keys[ch] < keys[ch + 1]) ch++;
					if (keys[ch] > keys[st]) {
						auto tk = keys[ch];
						keys[ch] = keys[st];
						keys[st] = tk;
						auto tv = values[ch];
						values[ch] = values[st];
						values[st] = tv;
						st = ch;
					}
					else {
						return;
					}
				}
			};
			
			for (int i = (count >> 1); i >= 0; i--) {
				siftDown(i, count);
			}
			for (int i = count - 1; i > 0; i--) {
				auto tk = keys[i];
				keys[i] = keys[0];
				keys[0] = tk;
				auto tv = values[i];
				values[i] = values[0];
				values[0] = tv;
				siftDown(0, i);
			}
		}

		template<typename T,typename U>
		IFRIT_DEVICE void devInsertionSortByKeyImpl(T* keys, U* values, int count) {
			for (int i = 1; i < count; i++) {
				auto key = keys[i];
				auto value = values[i];
				int j = i - 1;
				while (j >= 0 && keys[j] < key) {
					keys[j + 1] = keys[j];
					values[j + 1] = values[j];
					j--;
				}
				keys[j + 1] = key;
				values[j + 1] = value;
			}
		}

		template<typename T,typename U>
		IFRIT_DEVICE void devSortByKeyImpl(T* keys, U* values, int count) {
			devHeapSortByKeyImpl(keys, values, count);
		}

		IFRIT_KERNEL void sortPixelKeyGenerationKernel() {
			//TODO: Geometry shader integration
			int globalInvo = blockIdx.x * blockDim.x + threadIdx.x;
			if(globalInvo>= dSecondBinnerFinerBufferGlobalSize) return;
			auto primId = dSecondBinnerFinerBuffer[globalInvo].y;
			auto orgPrimid = dAtriOriginalPrimId[primId];

		}
		IFRIT_KERNEL void sortPixelKeysByTileKernel(int maxTileX,int maxTileY) {
			int subtileId = threadIdx.x;
			int tileX = threadIdx.y + blockDim.y * blockIdx.y;
			int tileY = threadIdx.z + blockDim.z * blockIdx.z;
			if(tileX>=maxTileX || tileY>=maxTileY) return;
			int tileId = tileX + tileY * CU_MAX_TILE_X;
			int totalId = tileId * CU_MAX_SUBTILES_PER_TILE + subtileId;

			int start = dSecondBinnerFinerBufferStart[totalId];
			int end = dSecondBinnerFinerBufferCurInd[totalId];
			int lim = dSecondBinnerFinerBufferSize[totalId];
			if(start==end) return;
			auto invoX = threadIdx.x + blockIdx.x * blockDim.x;
			auto invoY = threadIdx.y + blockIdx.y * blockDim.y;
			auto invoZ = threadIdx.z + blockIdx.z * blockDim.z;
			auto globalInvo = invoX + invoY * blockDim.x * gridDim.x + invoZ * blockDim.y * gridDim.y;
			devSortByKeyImpl(dSecondBinnerFinerBufferSortKeys + start,
				dSecondBinnerFinerBuffer + start, end - start); //dSecondBinnerFinerBuffer
		}
		IFRIT_DEVICE void devSortPixelKeys() {
			dLock = 0;
			dLockAt = 0;
			int dispatchBlocks = IFRIT_InvoGetThreadBlocks(dSecondBinnerFinerBufferGlobalSize, CU_SORT_KEYGEN_THREADS);
			sortPixelKeyGenerationKernel CU_KARG2(dispatchBlocks, CU_SORT_KEYGEN_THREADS) ();
			
			int maxTileX = IFRIT_InvoGetThreadBlocks(csFrameWidth, CU_TILE_WIDTH);
			int maxTileY = IFRIT_InvoGetThreadBlocks(csFrameHeight, CU_TILE_WIDTH);
			int dispX = IFRIT_InvoGetThreadBlocks(maxTileX, 2);
			int dispY = IFRIT_InvoGetThreadBlocks(maxTileY, 2);
			dim3 blocks(1, dispX, dispY);
			dim3 threads(CU_MAX_SUBTILES_PER_TILE, 2, 2);
			sortPixelKeysByTileKernel CU_KARG2(blocks, threads) (maxTileX, maxTileY);
		}

		IFRIT_KERNEL void sortBinKeyGenerationKernel() {
			int globalInvo = blockIdx.x * blockDim.x + threadIdx.x;
			if(globalInvo>= dCoverQueueFullM2GlobalSize) return;
			auto primId = dCoverQueueFullM2Buffer[globalInvo];
			auto orgPrimid = dAtriOriginalPrimId[primId];
			dCoverQueueFullM2SortKeys[globalInvo] = orgPrimid;
		}

		IFRIT_KERNEL void sortBinKeyByTileKernel(int maxBinX,int maxBinY) {
			int binx = threadIdx.x + blockDim.x * blockIdx.x;
			int biny = threadIdx.y + blockDim.y * blockIdx.y;
			if (binx >= maxBinX || biny >= maxBinY)return;

			int binId = binx + biny * CU_MAX_BIN_X;
			auto start = dCoverQueueFullM2Start[binId];
			auto end = dCoverQueueFullM2CurInd[binId];
			if (start == end)return;
			devSortByKeyImpl(dCoverQueueFullM2SortKeys + start, dCoverQueueFullM2Buffer + start, end - start);
		}

		IFRIT_DEVICE void devSortBinKeys() {
			int dispatchBlocks = IFRIT_InvoGetThreadBlocks(dCoverQueueFullM2GlobalSize, CU_SORT_KEYGEN_THREADS);
			sortBinKeyGenerationKernel CU_KARG2(dispatchBlocks, CU_SORT_KEYGEN_THREADS) ();
			int maxBinX = IFRIT_InvoGetThreadBlocks(csFrameWidth, CU_BIN_WIDTH);
			int maxBinY = IFRIT_InvoGetThreadBlocks(csFrameHeight, CU_BIN_WIDTH);
			int dispX = IFRIT_InvoGetThreadBlocks(maxBinX, 8);
			int dispY = IFRIT_InvoGetThreadBlocks(maxBinY, 8);
			sortBinKeyByTileKernel CU_KARG2(dim3(dispX, dispY, 1), (8, 8, 1))(maxBinX, maxBinY);
		}

		IFRIT_KERNEL void sortSuperTileKeyGenerationKernel() {
			int globalInvo = blockIdx.x * blockDim.x + threadIdx.x;
			if(globalInvo>= dCoverQueueSuperTileFullM3GlobalSize) return;
			auto primId = dCoverQueueSuperTileFullM3BufferFinal[globalInvo];
			auto orgPrimid = dAtriOriginalPrimId[primId];
			dCoverQueueSuperTileFullM3BufferFinalSortKeys[globalInvo] = orgPrimid;
		}

		IFRIT_KERNEL void sortSuperTileKeyByTileKernel(int maxBinX, int maxBinY) {
			int binx = threadIdx.x + blockDim.x * blockIdx.x;
			int biny = threadIdx.y + blockDim.y * blockIdx.y;
			if (binx >= maxBinX || biny >= maxBinY)return;

			int binId = binx + biny * CU_MAX_BIN_X;
			auto start = dCoverQueueSuperTileFullM3Start[binId];
			auto end = dCoverQueueSuperTileFullM3CurInd[binId];
			if (start == end)return;
			devSortByKeyImpl(dCoverQueueSuperTileFullM3BufferFinalSortKeys + start,
				dCoverQueueSuperTileFullM3Buffer + start, end - start);
		}

		IFRIT_DEVICE void devSortSuperTileKeys() {
			int dispatchBlocks = IFRIT_InvoGetThreadBlocks(dCoverQueueSuperTileFullM3GlobalSize, CU_SORT_KEYGEN_THREADS);
			sortSuperTileKeyGenerationKernel CU_KARG2(dispatchBlocks, CU_SORT_KEYGEN_THREADS) ();
			
			int maxBinX = IFRIT_InvoGetThreadBlocks(csFrameWidth, CU_LARGE_BIN_WIDTH);
			int maxBinY = IFRIT_InvoGetThreadBlocks(csFrameHeight, CU_LARGE_BIN_WIDTH);
			int dispX = IFRIT_InvoGetThreadBlocks(maxBinX, 8);
			int dispY = IFRIT_InvoGetThreadBlocks(maxBinY, 8);
			sortSuperTileKeyByTileKernel CU_KARG2(dim3(dispX, dispY, 1), (8, 8, 1))(maxBinX, maxBinY);
		}

		IFRIT_KERNEL void sortEntryKernel() {
			devSortPixelKeys();
			devSortBinKeys();
			devSortSuperTileKeys();
		}
	}
	namespace TriangleRasterizationStage {
		IFRIT_DEVICE void devExecuteBinnerLargeTile(int primitiveId, irect2Df bbox) {
			float minx = bbox.x;
			float miny = bbox.y;
			float maxx = bbox.w;
			float maxy = bbox.h;

			int tileLargeMinx = max(0, (int)(minx * csFrameWidth / CU_LARGE_BIN_WIDTH));
			int tileLargeMiny = max(0, (int)(miny * csFrameHeight / CU_LARGE_BIN_WIDTH));
			int tileLargeMaxx = min(CU_MAX_LARGE_BIN_X - 1, (int)(maxx * csFrameWidth / CU_LARGE_BIN_WIDTH));
			int tileLargeMaxy = min(CU_MAX_LARGE_BIN_X - 1, (int)(maxy * csFrameHeight / CU_LARGE_BIN_WIDTH));

			float4 ec1, ec2;
			float ec3;
			float4 edgeCoefs[3];
			ec1 = dAtriEdgeCoefs1[primitiveId];
			ec2 = dAtriEdgeCoefs2[primitiveId];
			ec3 = dAtriEdgeCoefs3[primitiveId];
			edgeCoefs[0] = { ec1.x,ec1.y,ec1.z };
			edgeCoefs[1] = { ec1.w,ec2.x,ec2.y };
			edgeCoefs[2] = { ec2.z,ec2.w,ec3 };

			ifloat2 tileCoordsT[4], tileCoordsS[4];
			int chosenCoordTR[3], chosenCoordTA[3];

			GeneralFunction::devGetAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);
			for (int y = tileLargeMiny + threadIdx.y; y <= tileLargeMaxy; y += CU_FIRST_BINNER_STRIDE_LARGE) {

				auto curTileLargeY = y * CU_LARGE_BIN_WIDTH;
				auto curTileLargeY2 = (y + 1) * CU_LARGE_BIN_WIDTH;
				auto cty1 = 1.0f * curTileLargeY, cty2 = 1.0f * (curTileLargeY2 - 1);

				float criteriaTRLocalY[3];
				float criteriaTALocalY[3];

#define getY(w) (((w)&1)?cty1:cty2)
				for (int i = 0; i < 3; i++) {
					criteriaTRLocalY[i] = edgeCoefs[i].y * getY(chosenCoordTR[i]);
					criteriaTALocalY[i] = edgeCoefs[i].y * getY(chosenCoordTA[i]);
				}
#undef getY
				for (int x = tileLargeMinx + threadIdx.z; x <= tileLargeMaxx; x += CU_FIRST_BINNER_STRIDE_LARGE) {
					auto curTileLargeX = x * CU_LARGE_BIN_WIDTH;
					auto curTileLargeX2 = (x + 1) * CU_LARGE_BIN_WIDTH;
					auto ctx1 = 1.0f * curTileLargeX;
					auto ctx2 = 1.0f * (curTileLargeX2 - 1);

					int criteriaTR = 0, criteriaTA = 0;

#define getX(w) (((w)>>1)?ctx2:ctx1)
					for (int i = 0; i < 3; i++) {
						float criteriaTRLocal = edgeCoefs[i].x * getX(chosenCoordTR[i]) + criteriaTRLocalY[i];
						float criteriaTALocal = edgeCoefs[i].x * getX(chosenCoordTA[i]) + criteriaTALocalY[i];
						criteriaTR += criteriaTRLocal < edgeCoefs[i].z;
						criteriaTA += criteriaTALocal < edgeCoefs[i].z;
					}
#undef getX
					if (criteriaTR != 3)
						continue;
					auto tileLargeId = y * CU_MAX_LARGE_BIN_X + x;
					if (criteriaTA == 3) {
						int x = atomicAdd(&dCoverQueueSuperTileFullM3TotlCands, 1);
						dCoverQueueSuperTileFullM3Buffer[x].x = primitiveId;
						dCoverQueueSuperTileFullM3Buffer[x].y = tileLargeId;

						atomicAdd(&dCoverQueueSuperTileFullM3Size[tileLargeId], 1);
					}
					else {
						auto plId = atomicAdd(&dCoverPrimsLargeTileCounter, 1);
						dCoverPrimsTileLarge[plId].x = primitiveId;
						dCoverPrimsTileLarge[plId].y = tileLargeId;
					}
				}
			}
		}

		IFRIT_DEVICE void devExecuteBinner(int primitiveId, irect2Df bbox, int stride) {
			float minx = bbox.x;
			float miny = bbox.y;
			float maxx = bbox.w;
			float maxy = bbox.h;

			int tileMinx = max(0, (int)(minx * csFrameWidth / CU_BIN_WIDTH));
			int tileMiny = max(0, (int)(miny * csFrameHeight / CU_BIN_WIDTH));
			int tileMaxx = min(CU_MAX_BIN_X - 1, (int)(maxx * csFrameWidth / CU_BIN_WIDTH));
			int tileMaxy = min(CU_MAX_BIN_X - 1, (int)(maxy * csFrameHeight / CU_BIN_WIDTH));

			float4 ec1, ec2;
			float ec3;
			float4 edgeCoefs[3];
			ec1 = dAtriEdgeCoefs1[primitiveId];
			ec2 = dAtriEdgeCoefs2[primitiveId];
			ec3 = dAtriEdgeCoefs3[primitiveId];
			edgeCoefs[0] = { ec1.x,ec1.y,ec1.z };
			edgeCoefs[1] = { ec1.w,ec2.x,ec2.y };
			edgeCoefs[2] = { ec2.z,ec2.w,ec3 };

			int chosenCoordTR[3];
			int chosenCoordTA[3];
			auto frameBufferHeight = csFrameHeight;
			GeneralFunction::devGetAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);

			int mHeight = tileMaxy - tileMiny + 1;
			int mWidth = tileMaxx - tileMinx + 1;
			int mTotal = mHeight * mWidth;

			auto curY = tileMiny + threadIdx.y / mWidth;
			auto curX = tileMinx + threadIdx.y % mWidth;

			auto curTileY = curY * CU_BIN_WIDTH;
			auto curTileY2 = (curY + 1) * CU_BIN_WIDTH;
			auto cty1 = 1.0f * curTileY, cty2 = 1.0f * (curTileY2 - 1);

			float criteriaTRLocalY[3];
			float criteriaTALocalY[3];

#define getY(w) (((w)&1)?cty1:cty2)
#define getX(w) (((w)>>1)?ctx2:ctx1)
			for (int i = 0; i < 3; i++) {
				criteriaTRLocalY[i] = edgeCoefs[i].y * getY(chosenCoordTR[i]);
				criteriaTALocalY[i] = edgeCoefs[i].y * getY(chosenCoordTA[i]);
			}
			for (int i2 = threadIdx.y; i2 < mTotal;) {
				auto curTileX = curX * CU_BIN_WIDTH;
				auto curTileX2 = (curX + 1) * CU_BIN_WIDTH;
				auto ctx1 = 1.0f * curTileX;
				auto ctx2 = 1.0f * (curTileX2 - 1);
				int criteriaTR = 0, criteriaTA = 0;
				for (int i = 0; i < 3; i++) {
					float criteriaTRLocal = edgeCoefs[i].x * getX(chosenCoordTR[i]) + criteriaTRLocalY[i];
					float criteriaTALocal = edgeCoefs[i].x * getX(chosenCoordTA[i]) + criteriaTALocalY[i];
					criteriaTR += criteriaTRLocal < edgeCoefs[i].z;
					criteriaTA += criteriaTALocal < edgeCoefs[i].z;
				}
				auto tileId = curY * CU_MAX_BIN_X + curX;
				if (criteriaTA == 3) {
					auto plId = atomicAdd(&dCoverPrimsCounter, 1);
					dCoverPrimsTile[plId].x = primitiveId;
					dCoverPrimsTile[plId].y = tileId;
					atomicAdd(&dCoverQueueFullM2Size[tileId], 1);
				}
				else if (criteriaTR == 3) {
					auto plId = atomicAdd(&dRasterQueueWorklistCounter, 1);
					dRasterQueueWorklistPrimTile[plId].y = tileId;
					dRasterQueueWorklistPrimTile[plId].x = primitiveId;
				}
				curX += stride;
				i2 += stride;
				if (curX > tileMaxx) {
					curY = tileMiny + i2 / mWidth;
					curX = tileMinx + i2 % mWidth;
					curTileY = curY * CU_BIN_WIDTH;
					curTileY2 = (curY + 1) * CU_BIN_WIDTH;
					cty1 = 1.0f * curTileY, cty2 = 1.0f * (curTileY2 - 1);
					for (int i = 0; i < 3; i++) {
						criteriaTRLocalY[i] = edgeCoefs[i].y * getY(chosenCoordTR[i]);
						criteriaTALocalY[i] = edgeCoefs[i].y * getY(chosenCoordTA[i]);
					}
				}
			}
#undef getY
#undef getX
		}
		IFRIT_DEVICE void devFinerTilingRasterizationChildProcess(
			uint32_t tileId,
			uint32_t primitiveSrcId,
			uint32_t subTileId,
			int xidSrc,
			float4 e1,
			float4 e2,
			float4 e3
		) {

			const auto tileIdX = tileId % CU_MAX_TILE_X;
			const auto tileIdY = tileId / CU_MAX_TILE_X;

			float4 edgeCoefs[3];
			edgeCoefs[0] = e1;
			edgeCoefs[1] = e2;
			edgeCoefs[2] = e3;

			auto curTileX = tileIdX * CU_TILE_WIDTH;
			auto curTileY = tileIdY * CU_TILE_WIDTH;

			constexpr int numSubtilesX = CU_TILE_WIDTH / CU_EXPERIMENTAL_SUBTILE_WIDTH;

			auto subTileIX = subTileId % numSubtilesX;
			auto subTileIY = subTileId / numSubtilesX;

			int criteriaTR = 0;
			int criteriaTA = 0;

			int subTilePixelX = curTileX + subTileIX * CU_EXPERIMENTAL_SUBTILE_WIDTH;
			int subTilePixelY = curTileY + subTileIY * CU_EXPERIMENTAL_SUBTILE_WIDTH;
			int subTilePixelX2 = curTileX + (subTileIX + 1) * CU_EXPERIMENTAL_SUBTILE_WIDTH;
			int subTilePixelY2 = curTileY + (subTileIY + 1) * CU_EXPERIMENTAL_SUBTILE_WIDTH;

			float subTileMinX = 1.0f * subTilePixelX;
			float subTileMinY = 1.0f * subTilePixelY;
			float subTileMaxX = 1.0f * (subTilePixelX2 - 1);
			float subTileMaxY = 1.0f * (subTilePixelY2 - 1);

			float ccTRX[3], ccTRY[3], ccTAX[3], ccTAY[3];

			for (int i = 0; i < 3; i++) {
				bool normalRight = edgeCoefs[i].x < 0;
				bool normalDown = edgeCoefs[i].y < 0;
				if (normalRight) {
					ccTRX[i] = normalDown ? subTileMaxX : subTileMaxX;
					ccTRY[i] = normalDown ? subTileMaxY : subTileMinY;
					ccTAX[i] = normalDown ? subTileMinX : subTileMinX;
					ccTAY[i] = normalDown ? subTileMinY : subTileMaxY;
				}
				else {
					ccTRX[i] = normalDown ? subTileMinX : subTileMinX;
					ccTRY[i] = normalDown ? subTileMaxY : subTileMinY;
					ccTAX[i] = normalDown ? subTileMaxX : subTileMaxX;
					ccTAY[i] = normalDown ? subTileMinY : subTileMaxY;
				}
			}

			int mask = 0;
			float criteriaY[3];
			float criteriaX[3];

			const auto rsX1 = subTilePixelX * edgeCoefs[0].x;
			const auto rsX2 = subTilePixelX * edgeCoefs[1].x;
			const auto rsX3 = subTilePixelX * edgeCoefs[2].x;

			criteriaY[0] = subTilePixelY * edgeCoefs[0].y;
			criteriaY[1] = subTilePixelY * edgeCoefs[1].y;
			criteriaY[2] = subTilePixelY * edgeCoefs[2].y;
			criteriaX[0] = rsX1;
			criteriaX[1] = rsX2;
			criteriaX[2] = rsX3;
			constexpr auto dEps = CU_OPT_PATCH_STRICT_BOUNDARY ? CU_EPS * 1e7f : 0;
			for (int i2 = 0; i2 < CU_EXPERIMENTAL_PIXELS_PER_SUBTILE; i2++) {
				bool accept1 = (criteriaX[0] + criteriaY[0]) < edgeCoefs[0].z + dEps;
				bool accept2 = (criteriaX[1] + criteriaY[1]) < edgeCoefs[1].z + dEps;
				bool accept3 = (criteriaX[2] + criteriaY[2]) < edgeCoefs[2].z + dEps;

				int cond = (accept1 && accept2 && accept3);
				mask |= (cond << i2);

				if ((i2 + 1) % CU_EXPERIMENTAL_SUBTILE_WIDTH == 0) {
					criteriaY[0] += edgeCoefs[0].y;
					criteriaY[1] += edgeCoefs[1].y;
					criteriaY[2] += edgeCoefs[2].y;
					criteriaX[0] = rsX1;
					criteriaX[1] = rsX2;
					criteriaX[2] = rsX3;
				}
				else {
					criteriaX[0] += edgeCoefs[0].x;
					criteriaX[1] += edgeCoefs[1].x;
					criteriaX[2] += edgeCoefs[2].x;
				}
			}
			//if (mask == 0) {
			//	return;
			//}
			int xid = xidSrc;
			int pw = atomicAdd(&dSecondBinnerFinerBufferCurInd[xid], 1);
			dSecondBinnerFinerBuffer[pw].x = mask;
			dSecondBinnerFinerBuffer[pw].y = primitiveSrcId;
			dSecondBinnerFinerBufferSortKeys[pw] = dAtriOriginalPrimId[primitiveSrcId];
		}

		IFRIT_DEVICE void devTilingRasterizationChildProcess(
			uint32_t tileIdX,
			uint32_t tileIdY,
			uint32_t primitiveSrcId,
			uint32_t subTileId,
			float4 e1,
			float4 e2,
			float4 e3
		) {
			const auto tileId = tileIdY * CU_MAX_TILE_X + tileIdX;
			const auto frameWidth = csFrameWidth;
			const auto frameHeight = csFrameHeight;

			float4 edgeCoefs[3];
			edgeCoefs[0] = e1;
			edgeCoefs[1] = e2;
			edgeCoefs[2] = e3;

			auto curTileX = tileIdX * CU_TILE_WIDTH;
			auto curTileY = tileIdY * CU_TILE_WIDTH;

			constexpr int numSubtilesX = CU_TILE_WIDTH / CU_EXPERIMENTAL_SUBTILE_WIDTH;

			auto subTileIX = subTileId % numSubtilesX;
			auto subTileIY = subTileId / numSubtilesX;

			int criteriaTR = 0;
			int criteriaTA = 0;

			int subTilePixelX = curTileX + subTileIX * CU_EXPERIMENTAL_SUBTILE_WIDTH;
			int subTilePixelY = curTileY + subTileIY * CU_EXPERIMENTAL_SUBTILE_WIDTH;
			int subTilePixelX2 = curTileX + (subTileIX + 1) * CU_EXPERIMENTAL_SUBTILE_WIDTH;
			int subTilePixelY2 = curTileY + (subTileIY + 1) * CU_EXPERIMENTAL_SUBTILE_WIDTH;

			float subTileMinX = 1.0f * subTilePixelX;
			float subTileMinY = 1.0f * subTilePixelY;
			float subTileMaxX = 1.0f * (subTilePixelX2 - 1);
			float subTileMaxY = 1.0f * (subTilePixelY2 - 1);

			float ccTRX[3], ccTRY[3], ccTAX[3], ccTAY[3];

#pragma unroll
			for (int i = 0; i < 3; i++) {
				bool normalRight = edgeCoefs[i].x < 0;
				bool normalDown = edgeCoefs[i].y < 0;
				if (normalRight) {
					ccTRX[i] = normalDown ? subTileMaxX : subTileMaxX;
					ccTRY[i] = normalDown ? subTileMaxY : subTileMinY;
					ccTAX[i] = normalDown ? subTileMinX : subTileMinX;
					ccTAY[i] = normalDown ? subTileMinY : subTileMaxY;
				}
				else {
					ccTRX[i] = normalDown ? subTileMinX : subTileMinX;
					ccTRY[i] = normalDown ? subTileMaxY : subTileMinY;
					ccTAX[i] = normalDown ? subTileMaxX : subTileMaxX;
					ccTAY[i] = normalDown ? subTileMinY : subTileMaxY;
				}
			}
#pragma unroll
			for (int k = 0; k < 3; k++) {
				float criteriaTRLocal = edgeCoefs[k].x * ccTRX[k] + edgeCoefs[k].y * ccTRY[k];
				float criteriaTALocal = edgeCoefs[k].x * ccTAX[k] + edgeCoefs[k].y * ccTAY[k];
				criteriaTR += criteriaTRLocal < edgeCoefs[k].z;
				criteriaTA += criteriaTALocal < edgeCoefs[k].z;
			}

			int mask = 0;
			if (criteriaTR == 3) {
				int x = atomicAdd(&dSecondBinnerCandCounter, 1);
				dSecondBinnerPendingPrim[x].x = primitiveSrcId;
				dSecondBinnerPendingPrim[x].y = tileId * CU_MAX_SUBTILES_PER_TILE + subTileId;
				atomicAdd(&dSecondBinnerFinerBufferSize[tileId * CU_MAX_SUBTILES_PER_TILE + subTileId], 1);
			}
		}
		IFRIT_KERNEL void firstBinnerRasterizerSeparateTinyKernel(uint32_t startOffset, uint32_t bound) {
			auto globalInvo = blockIdx.x * blockDim.x + threadIdx.x;
			if (globalInvo >= bound)return;
			auto dBoundingBoxM2Aligned = reinterpret_cast<irect2Df*>(__builtin_assume_aligned(dBoundingBoxM2, 16));
			auto bbox = dBoundingBoxM2Aligned[startOffset + globalInvo];
			float bboxSz = min(bbox.w - bbox.x, bbox.h - bbox.y);
			bool isTiny = (bbox.w - bbox.x) < CU_EXPERIMENTAL_TINY_THRESHOLD && (bbox.h - bbox.y) < CU_EXPERIMENTAL_TINY_THRESHOLD;
			if (bboxSz <= CU_LARGE_TRIANGLE_THRESHOLD && isTiny) {
				devExecuteBinner(startOffset + globalInvo, bbox, CU_FIRST_BINNER_STRIDE_TINY);
			}
		}

		IFRIT_KERNEL void firstBinnerRasterizerSeparateSmallKernel(uint32_t startOffset, uint32_t bound) {
			auto globalInvo = blockIdx.x * blockDim.x + threadIdx.x;
			if (globalInvo >= bound)return;
			auto dBoundingBoxM2Aligned = reinterpret_cast<irect2Df*>(__builtin_assume_aligned(dBoundingBoxM2, 16));
			auto bbox = dBoundingBoxM2Aligned[startOffset + globalInvo];
			float bboxSz = min(bbox.w - bbox.x, bbox.h - bbox.y);
			bool isTiny = (bbox.w - bbox.x) < CU_EXPERIMENTAL_TINY_THRESHOLD && (bbox.h - bbox.y) < CU_EXPERIMENTAL_TINY_THRESHOLD;
			if (bboxSz <= CU_LARGE_TRIANGLE_THRESHOLD && !isTiny) {
				devExecuteBinner(startOffset + globalInvo, bbox, CU_FIRST_BINNER_STRIDE);
			}
		}

		IFRIT_KERNEL void firstBinnerRasterizerSeparateLargeKernel(uint32_t startOffset, uint32_t bound) {
			auto globalInvo = blockIdx.x * blockDim.x + threadIdx.x;
			if (globalInvo >= bound)return;
			auto dBoundingBoxM2Aligned = reinterpret_cast<irect2Df*>(__builtin_assume_aligned(dBoundingBoxM2, 16));
			auto bbox = dBoundingBoxM2Aligned[startOffset + globalInvo];
			float bboxSz = min(bbox.w - bbox.x, bbox.h - bbox.y);
			if (bboxSz > CU_LARGE_TRIANGLE_THRESHOLD) {
				devExecuteBinnerLargeTile(startOffset + globalInvo, bbox);
			}
		}

		IFRIT_KERNEL void firstBinnerRasterizerSeparateLargeFinerKernel(uint32_t bound) {
			auto globalInvo = blockIdx.x * blockDim.x + threadIdx.x;
			if (globalInvo >= bound)return;
			int primitiveId = dCoverPrimsTileLarge[globalInvo].x;
			int orgPrimId = primitiveId;
			int orgLargeTileId = dCoverPrimsTileLarge[globalInvo].y;
			auto frameBufferHeight = csFrameHeight;
			auto frameBufferWidth = csFrameWidth;
			auto x = orgLargeTileId % CU_MAX_LARGE_BIN_X;
			auto y = orgLargeTileId / CU_MAX_LARGE_BIN_X;

			float4 ec1, ec2;
			float ec3;
			float4 edgeCoefs[3];
			ec1 = dAtriEdgeCoefs1[orgPrimId];
			ec2 = dAtriEdgeCoefs2[orgPrimId];
			ec3 = dAtriEdgeCoefs3[orgPrimId];
			edgeCoefs[0] = { ec1.x,ec1.y,ec1.z };
			edgeCoefs[1] = { ec1.w,ec2.x,ec2.y };
			edgeCoefs[2] = { ec2.z,ec2.w,ec3 };

			int chosenCoordTR[3];
			int chosenCoordTA[3];
			GeneralFunction::devGetAcceptRejectCoords(edgeCoefs, chosenCoordTR, chosenCoordTA);

			auto dx = threadIdx.y;
			auto dy = threadIdx.z;

			int ty = y * CU_BINS_PER_LARGE_BIN + dy;
			int tx = x * CU_BINS_PER_LARGE_BIN + dx;
			auto curTileY = ty * CU_BIN_WIDTH;
			auto curTileY2 = (ty + 1) * CU_BIN_WIDTH;
			auto cty1 = 1.0f * curTileY, cty2 = 1.0f * (curTileY2 - 1);
			auto curTileX = tx * CU_BIN_WIDTH;
			auto curTileX2 = (tx + 1) * CU_BIN_WIDTH;
			auto ctx1 = 1.0f * curTileX;
			auto ctx2 = 1.0f * (curTileX2 - 1);

			int criteriaTR = 0, criteriaTA = 0;
#define getX(w) (((w)>>1)?ctx2:ctx1)
#define getY(w) (((w)&1)?cty1:cty2)
			for (int i = 0; i < 3; i++) {
				float criteriaTRLocal = edgeCoefs[i].x * getX(chosenCoordTR[i]) + edgeCoefs[i].y * getY(chosenCoordTR[i]);
				float criteriaTALocal = edgeCoefs[i].x * getX(chosenCoordTA[i]) + edgeCoefs[i].y * getY(chosenCoordTA[i]);
				criteriaTR += criteriaTRLocal < edgeCoefs[i].z;
				criteriaTA += criteriaTALocal < edgeCoefs[i].z;
			}
#undef getX
#undef getY
			if (criteriaTR != 3) return;
			auto tileId = ty * CU_MAX_BIN_X + tx;
			if (criteriaTA == 3) {
				auto plId = atomicAdd(&dCoverPrimsCounter, 1);
				dCoverPrimsTile[plId].x = primitiveId;
				dCoverPrimsTile[plId].y = tileId;
				atomicAdd(&dCoverQueueFullM2Size[tileId], 1);
			}
			else {
				auto plId = atomicAdd(&dRasterQueueWorklistCounter, 1);
				dRasterQueueWorklistPrimTile[plId].y = tileId;
				dRasterQueueWorklistPrimTile[plId].x = primitiveId;
			}
		}

		IFRIT_KERNEL void firstBinnerRasterizerSeparateLargeFinerEntryKernel() {
			auto dispatchBlocks = dCoverPrimsLargeTileCounter / CU_FIRST_RASTERIZATION_LARGE_FINER_PROC + (dCoverPrimsLargeTileCounter % CU_FIRST_RASTERIZATION_LARGE_FINER_PROC != 0);
			auto total = dCoverPrimsLargeTileCounter;
			if (dCoverPrimsLargeTileCounter == 0)return;
			firstBinnerRasterizerSeparateLargeFinerKernel CU_KARG2(
				dim3(dispatchBlocks, 1, 1),
				dim3(CU_FIRST_RASTERIZATION_LARGE_FINER_PROC, CU_BINS_PER_LARGE_BIN, CU_BINS_PER_LARGE_BIN)
			)(total);
		}
		IFRIT_KERNEL void firstBinnerGatherAllocKernel() {
			auto globalInvo = threadIdx.x + blockIdx.x * blockDim.x;
			auto binX = globalInvo % CU_MAX_BIN_X;
			auto binY = globalInvo / CU_MAX_BIN_X;
			dCoverQueueFullM2Start[globalInvo] = atomicAdd(&dCoverQueueFullM2GlobalSize, dCoverQueueFullM2Size[globalInvo]);
			dCoverQueueFullM2CurInd[globalInvo] = dCoverQueueFullM2Start[globalInvo];
			//printf("%lld %d\n", globalInvo, dCoverQueueFullM2Size[globalInvo]);
		}
		IFRIT_KERNEL void firstBinnerGatherKernel(uint32_t bound) {
			auto globalInvo = blockIdx.x * blockDim.x + threadIdx.x;
			if (globalInvo >= bound)return;
			int primId = dCoverPrimsTile[globalInvo].x;
			int tileId = dCoverPrimsTile[globalInvo].y;
			//dCoverQueueFullM2[tileId].push_back(primId);
			int v = atomicAdd(&dCoverQueueFullM2CurInd[tileId], 1);
			dCoverQueueFullM2Buffer[v] = primId;
			dCoverQueueFullM2SortKeys[v] = dAtriOriginalPrimId[primId];
		}

		IFRIT_KERNEL void firstBinnerGatherEntryKernel() {
			auto dispatchBlocks = dCoverPrimsCounter / CU_FIRST_RASTERIZATION_GATHER_THREADS + (dCoverPrimsCounter % CU_FIRST_RASTERIZATION_GATHER_THREADS != 0);
			if (dCoverPrimsCounter == 0)return;
			firstBinnerGatherKernel CU_KARG2(dim3(dispatchBlocks, 1, 1), dim3(CU_FIRST_RASTERIZATION_GATHER_THREADS, 1, 1)) (dCoverPrimsCounter);
		}

		IFRIT_KERNEL void largeTileBufferAllocationKernel() {
			int globalInvo = threadIdx.x + blockIdx.x * blockDim.x;
			dCoverQueueSuperTileFullM3Start[globalInvo] = atomicAdd(&dCoverQueueSuperTileFullM3GlobalSize, dCoverQueueSuperTileFullM3Size[globalInvo]);
			dCoverQueueSuperTileFullM3CurInd[globalInvo] = dCoverQueueSuperTileFullM3Start[globalInvo];
		}

		IFRIT_KERNEL void largeTileBufferPlacementKernel(int total) {
			int globalInvo = threadIdx.x + blockIdx.x * blockDim.x;
			if (globalInvo >= total) return;
			int primId = dCoverQueueSuperTileFullM3Buffer[globalInvo].x;
			int largeTileId = dCoverQueueSuperTileFullM3Buffer[globalInvo].y;
			int v = atomicAdd(&dCoverQueueSuperTileFullM3CurInd[largeTileId], 1);
			dCoverQueueSuperTileFullM3BufferFinal[v] = primId;
			dCoverQueueSuperTileFullM3BufferFinalSortKeys[v] = largeTileId;
		}

		IFRIT_KERNEL void largeTileBufferPlacementEntryKernel() {
			int total = dCoverQueueSuperTileFullM3TotlCands;
			int dispBlocksLp = IFRIT_InvoGetThreadBlocks(total, 64);
			if (dCoverQueueSuperTileFullM3TotlCands == 0)return;
			largeTileBufferPlacementKernel CU_KARG2(dispBlocksLp, 64)(total);
		}

		IFRIT_KERNEL void firstBinnerRasterizerEntryKernel(uint32_t firstTriangle, uint32_t triangleNum) {
			auto totalTriangles = triangleNum;
			auto dispatchBlocks = totalTriangles / CU_FIRST_RASTERIZATION_THREADS + (totalTriangles % CU_FIRST_RASTERIZATION_THREADS != 0);
			auto dispatchBlocksLarge = totalTriangles / CU_FIRST_RASTERIZATION_THREADS_LARGE + (totalTriangles % CU_FIRST_RASTERIZATION_THREADS_LARGE != 0);
			auto dispatchBlocksTiny = totalTriangles / CU_FIRST_RASTERIZATION_THREADS_TINY + (totalTriangles % CU_FIRST_RASTERIZATION_THREADS_TINY != 0);

			firstBinnerRasterizerSeparateLargeKernel CU_KARG2(dim3(dispatchBlocksLarge, 1, 1), dim3(CU_FIRST_RASTERIZATION_THREADS_LARGE, CU_FIRST_BINNER_STRIDE_LARGE, CU_FIRST_BINNER_STRIDE_LARGE)) (
				firstTriangle, totalTriangles);
			firstBinnerRasterizerSeparateLargeFinerEntryKernel CU_KARG2(1, 1)();
			largeTileBufferAllocationKernel CU_KARG2(CU_MAX_LARGE_BIN_X, CU_MAX_LARGE_BIN_X)();
			largeTileBufferPlacementEntryKernel CU_KARG2(1, 1)();
			firstBinnerRasterizerSeparateSmallKernel CU_KARG2(dim3(dispatchBlocks, 1, 1), dim3(CU_FIRST_RASTERIZATION_THREADS, CU_FIRST_BINNER_STRIDE, 1)) (
				firstTriangle, totalTriangles);
			firstBinnerRasterizerSeparateTinyKernel CU_KARG2(dim3(dispatchBlocksTiny, 1, 1), dim3(CU_FIRST_RASTERIZATION_THREADS_TINY, CU_FIRST_BINNER_STRIDE_TINY, 1)) (
				firstTriangle, totalTriangles);
		}

		IFRIT_HOST void firstBinnerRasterizerEntryProfileKernel(uint32_t firstTriangle, uint32_t triangleNum) {
			cudaDeviceSynchronize();
			auto totalTriangles = triangleNum;
			auto dispatchBlocks = totalTriangles / CU_FIRST_RASTERIZATION_THREADS + (totalTriangles % CU_FIRST_RASTERIZATION_THREADS != 0);
			auto dispatchBlocksLarge = totalTriangles / CU_FIRST_RASTERIZATION_THREADS_LARGE + (totalTriangles % CU_FIRST_RASTERIZATION_THREADS_LARGE != 0);
			auto dispatchBlocksTiny = totalTriangles / CU_FIRST_RASTERIZATION_THREADS_TINY + (totalTriangles % CU_FIRST_RASTERIZATION_THREADS_TINY != 0);

			firstBinnerRasterizerSeparateLargeKernel CU_KARG2(dim3(dispatchBlocksLarge, 1, 1), dim3(CU_FIRST_RASTERIZATION_THREADS_LARGE,
				CU_FIRST_RASTERIZATION_THREADS_LARGE, CU_FIRST_BINNER_STRIDE_LARGE)) (
					firstTriangle, totalTriangles);
			firstBinnerRasterizerSeparateLargeFinerEntryKernel CU_KARG2(1, 1)();
			//===
			auto largeTileProposals = 0;
			cudaMemcpyFromSymbol(&largeTileProposals, dCoverQueueSuperTileFullM3TotlCands, sizeof(int));
			largeTileBufferAllocationKernel CU_KARG2(CU_MAX_LARGE_BIN_X, CU_MAX_LARGE_BIN_X)();
			int dispBlocksLp = IFRIT_InvoGetThreadBlocks(largeTileProposals, 64);
			if (dispBlocksLp != 0) largeTileBufferPlacementKernel CU_KARG2(dispBlocksLp, 64)(largeTileProposals);
			//===
			if (dispatchBlocks != 0) {
				firstBinnerRasterizerSeparateSmallKernel CU_KARG2(dim3(dispatchBlocks, 1, 1), dim3(CU_FIRST_RASTERIZATION_THREADS, CU_FIRST_BINNER_STRIDE, 1)) (
					firstTriangle, totalTriangles);
			}
			if (dispatchBlocksTiny != 0) {
				firstBinnerRasterizerSeparateTinyKernel CU_KARG2(dim3(dispatchBlocksTiny, 1, 1), dim3(CU_FIRST_RASTERIZATION_THREADS_TINY, CU_FIRST_BINNER_STRIDE_TINY, 1)) (
					firstTriangle, totalTriangles);
			}
			if (dispatchBlocks == 0) {
				printf("ABNORMAL \n");
			}
		}
		IFRIT_KERNEL void secondBinnerWorklistRasterizerKernel(int totalElements) {
			const auto globalInvo = (blockIdx.x * CU_ELEMENTS_PER_SECOND_BINNER_BLOCK) + threadIdx.z;
			if (globalInvo > totalElements)return;
			int binId = dRasterQueueWorklistPrimTile[globalInvo].y;
			int prim = dRasterQueueWorklistPrimTile[globalInvo].x;
			int binIdxY = binId / CU_MAX_BIN_X;
			int binIdxX = binId % CU_MAX_BIN_X;
			int tileIdxX = binIdxX * CU_TILES_PER_BIN + threadIdx.x % CU_TILES_PER_BIN;
			int tileIdxY = binIdxY * CU_TILES_PER_BIN + threadIdx.x / CU_TILES_PER_BIN;

			float4 ec1, ec2;
			float ec3;
			float4 edgeCoefs[3];
			ec1 = dAtriEdgeCoefs1[prim];
			ec2 = dAtriEdgeCoefs2[prim];
			ec3 = dAtriEdgeCoefs3[prim];
			edgeCoefs[0] = { ec1.x,ec1.y,ec1.z };
			edgeCoefs[1] = { ec1.w,ec2.x,ec2.y };
			edgeCoefs[2] = { ec2.z,ec2.w,ec3 };

			devTilingRasterizationChildProcess(tileIdxX, tileIdxY, prim, threadIdx.y, edgeCoefs[0], edgeCoefs[1], edgeCoefs[2]);
		}

		IFRIT_KERNEL void secondBinnerWorklistRasterizerEntryKernel() {
			const auto totalElements = dRasterQueueWorklistCounter;
			const auto dispatchBlocks = (totalElements / CU_ELEMENTS_PER_SECOND_BINNER_BLOCK) + (totalElements % CU_ELEMENTS_PER_SECOND_BINNER_BLOCK != 0);
			if (totalElements == 0)return;
			if constexpr (CU_PROFILER_SECOND_BINNER_WORKQUEUE) {
				printf("Second Binner Work Queue Size: %d\n", totalElements);
				printf("Dispatched Thread Blocks %d\n", dispatchBlocks);
			}
			secondBinnerWorklistRasterizerKernel CU_KARG2(dim3(dispatchBlocks, 1, 1), dim3(CU_TILES_PER_BIN * CU_TILES_PER_BIN, 16, CU_ELEMENTS_PER_SECOND_BINNER_BLOCK)) (totalElements);
		}

		IFRIT_HOST void secondBinnerWorklistRasterizerEntryProfileKernel() {
			auto totalElements = 0;
			cudaDeviceSynchronize();
			cudaMemcpyFromSymbol(&totalElements, dRasterQueueWorklistCounter, sizeof(int));
			const auto dispatchBlocks = (totalElements / CU_ELEMENTS_PER_SECOND_BINNER_BLOCK) + (totalElements % CU_ELEMENTS_PER_SECOND_BINNER_BLOCK != 0);
			if (totalElements == 0)return;
			if constexpr (CU_PROFILER_SECOND_BINNER_WORKQUEUE) {
				printf("Second Binner Work Queue Size: %d\n", totalElements);
				printf("Dispatched Thread Blocks %d\n", dispatchBlocks);
			}
			secondBinnerWorklistRasterizerKernel CU_KARG2(dim3(dispatchBlocks, 1, 1), dim3(CU_TILES_PER_BIN * CU_TILES_PER_BIN, CU_MAX_SUBTILES_PER_TILE, CU_ELEMENTS_PER_SECOND_BINNER_BLOCK)) (totalElements);
		}

		IFRIT_KERNEL void secondFinerBinnerAllocationKernel() {
			int globalInvo = threadIdx.x + blockIdx.x * blockDim.x;
			dSecondBinnerFinerBufferStart[globalInvo] = atomicAdd(&dSecondBinnerFinerBufferGlobalSize, dSecondBinnerFinerBufferSize[globalInvo]);
			dSecondBinnerFinerBufferCurInd[globalInvo] = dSecondBinnerFinerBufferStart[globalInvo];
		}

		IFRIT_KERNEL void secondFinerBinnerRasterizationKernel(int totalCount) {
			int globalInvo = threadIdx.x + blockIdx.x * blockDim.x;
			if (globalInvo >= totalCount)return;
			auto dv = dSecondBinnerPendingPrim[globalInvo].y;
			int tileId = dv / CU_MAX_SUBTILES_PER_TILE;
			int subTileId = dv % CU_MAX_SUBTILES_PER_TILE;
			int prim = dSecondBinnerPendingPrim[globalInvo].x;

			float4 ec1, ec2;
			float ec3;
			float4 edgeCoefs[3];
			ec1 = dAtriEdgeCoefs1[prim];
			ec2 = dAtriEdgeCoefs2[prim];
			ec3 = dAtriEdgeCoefs3[prim];
			edgeCoefs[0] = { ec1.x,ec1.y,ec1.z };
			edgeCoefs[1] = { ec1.w,ec2.x,ec2.y };
			edgeCoefs[2] = { ec2.z,ec2.w,ec3 };

			devFinerTilingRasterizationChildProcess(tileId, prim, subTileId, dv, edgeCoefs[0], edgeCoefs[1], edgeCoefs[2]);
		}

		IFRIT_KERNEL void secondFinerBinnerRasterizationEntryKernel() {
			int dispatchBlock = IFRIT_InvoGetThreadBlocks(dSecondBinnerCandCounter, 128);
			secondFinerBinnerRasterizationKernel CU_KARG2(dispatchBlock, 128)(dSecondBinnerCandCounter);
		}
	}
	namespace TriangleGeometryStage {
		template<bool tpGeometryShaderEnabled,IfritCullMode tpCullMode>
		IFRIT_DEVICE void devGeometryCullClip(float4 v1, float4 v2, float4 v3, int primId) {
			using Ifrit::Engine::Math::ShaderOps::CUDA::lerp;
			constexpr int possibleTris = 5;

			IFRIT_SHARED TileRasterClipVertexCUDA retd[2 * CU_GEOMETRY_PROCESSING_THREADS];
			IFRIT_SHARED int retdIndex[5 * CU_GEOMETRY_PROCESSING_THREADS];

			int retdTriCnt = 3;
			const auto retOffset = threadIdx.x * 2;
			const auto retIdxOffset = threadIdx.x * 5;

			uint32_t retCnt = 0;
			TileRasterClipVertexCUDA pc;
			pc.barycenter = { 1,0,0 };
			pc.pos = v1;
			auto npc = -pc.pos.w;

			float4 vSrc[3] = { v1,v2,v3 };
			for (int j = 0; j < 3; j++) {
				auto curNid = (j + 1) % 3;
				auto pnPos = vSrc[curNid];
				auto npn = -pnPos.w;

				float3 pnBary = { 0,0,0 };
				if (j == 0) pnBary.y = 1;
				if (j == 1) pnBary.z = 1;
				if (j == 2) pnBary.x = 1;

				if constexpr (!CU_OPT_HOMOGENEOUS_DISCARD) {
					if (npc * npn < 0) {
						float numo = pc.pos.w;
						float deno = -(pnPos.w - pc.pos.w);
						float t = (-1e-3 + numo) / deno;
						float4 intersection = lerp(pc.pos, pnPos, t);
						float3 barycenter = lerp(pc.barycenter, pnBary, t);

						TileRasterClipVertexCUDA newp;
						newp.barycenter = barycenter;
						newp.pos = intersection;
						retd[retdTriCnt + retOffset - 3] = newp;
						retdIndex[retCnt + retIdxOffset] = retdTriCnt;
						retCnt++;
						retdTriCnt++;
					}
				}

				if (npn < 0) {
					retdIndex[retCnt + retIdxOffset] = curNid;
					retCnt++;
				}
				npc = npn;
				pc.pos = pnPos;
				pc.barycenter = pnBary;
			}
			if (retCnt < 3) {
				return;
			}

			const auto clipOdd = 0;

#define normalizeVertex(v) v.w=1/v.w; v.x*=v.w; v.y*=v.w; v.z*=v.w;v.x=v.x*0.5f+0.5f;v.y=v.y*0.5f+0.5f;
			normalizeVertex(v1);
			normalizeVertex(v2);
			normalizeVertex(v3);

#pragma unroll
			for (int i = 3; i < retdTriCnt; i++) {
				normalizeVertex(retd[i + retOffset - 3].pos);
			}
#undef normalizeVertex

			// Atomic Insertions
			const auto frameHeight = csFrameHeight;
			for (int i = 0; i < retCnt - 2; i++) {
				int temp;
#define getBary(fx,fy) (temp = retdIndex[(fx)* possibleTris+(fy)+ retIdxOffset], (temp==0)?float3{1,0,0}:((temp==1)?float3{0,1,0}:(temp==2)?float3{0,0,1}:retd[temp - 3 + retOffset].barycenter))
#define getPos(fx,fy) (temp = retdIndex[(fx)* possibleTris+(fy)+ retIdxOffset], (temp==0)?v1:((temp==1)?v2:(temp==2)?v3:retd[temp - 3 + retOffset].pos))
				auto b1 = getBary(clipOdd, 0);
				auto b2 = getBary(clipOdd, i + 1);
				auto b3 = getBary(clipOdd, i + 2);

				float4 dv1 = getPos(clipOdd, 0);
				float4 dv2 = getPos(clipOdd, i + 1);
				float4 dv3 = getPos(clipOdd, i + 2);
#undef getBary
#undef getPos
				if (GeneralFunction::devViewSpaceClip(dv1, dv2, dv3)) {
					continue;
				}

				// TODO: templated constexpr
				constexpr auto cullMode = tpCullMode;
				if constexpr (cullMode == IF_CULL_MODE_BACK) {
					if (!GeneralFunction::devTriangleCull(dv1, dv2, dv3)) {
						continue;
					}
				}
				if constexpr (cullMode == IF_CULL_MODE_FRONT) {
					if (!GeneralFunction::devTriangleCull(dv3, dv2, dv1)) {
						continue;
					}
					auto dvt = dv1;
					dv1 = dv3;
					dv3 = dvt;
					auto bt = b1;
					b1 = b3;
					b3 = bt;
				}
				if constexpr (cullMode == IF_CULL_MODE_FRONT_AND_BACK) {
					continue;
				}
				if constexpr (cullMode == IF_CULL_MODE_NONE) {
					if (!GeneralFunction::devTriangleCull(dv1, dv2, dv3)) {
						auto dvt = dv1;
						dv1 = dv3;
						dv3 = dvt;
						auto bt = b1;
						b1 = b3;
						b3 = bt;
					}
				}
				
				if constexpr (CU_OPT_SMALL_PRIMITIVE_CULL) {
					float4 bbox;
					GeneralFunction::devGetBBox(dv1, dv2, dv3, bbox);
					bbox.x *= csFrameWidth - 0.5f;
					bbox.z *= csFrameWidth - 0.5f;
					bbox.y *= csFrameHeight - 0.5f;
					bbox.w *= csFrameHeight - 0.5f;
					if (round(bbox.x) == round(bbox.z) || round(bbox.w) == round(bbox.y)) {
						continue;
					}
				}

				auto curIdx = atomicAdd(&dAssembledTriangleCounterM2, 1);
				dAtriInterpolBase1[curIdx] = { dv1.x,dv1.y,dv1.z,dv1.w };
				dAtriInterpolBase2[curIdx] = { dv2.x,dv2.y,dv2.z,dv2.w };
				dBoundingBoxM2[curIdx] = { dv3.x,dv3.y,dv3.z,dv3.w };
				dAtriBaryCenter12[curIdx] = { b1.x,b1.y,b2.x,b2.y };
				dAtriBaryCenter3[curIdx] = { b3.x,b3.y };
				dAtriOriginalPrimId[curIdx] = primId;

				if constexpr (tpGeometryShaderEnabled) {
					dAtriOriginalPrimId[curIdx] = dGeometryShaderOutSortKeys[primId * 3];
					dAtriOriginalPrimIdGs[curIdx] = primId;
				}
			}
#undef ret
		}
		template <int geometryShaderEnabled,IfritCullMode tpCullMode>
		IFRIT_KERNEL void geometryClippingKernel(
			ifloat4* IFRIT_RESTRICT_CUDA dPosBuffer,
			int* IFRIT_RESTRICT_CUDA dIndexBuffer,
			uint32_t startingIndexId,
			uint32_t indexCount
		) {
			//TODO: Culling face (Templated)
			float4 v1, v2, v3;
			int primId;
			const auto globalInvoIdx = blockIdx.x * blockDim.x + threadIdx.x;
			if constexpr (geometryShaderEnabled) {
				if (globalInvoIdx >= indexCount / 3) return;
				auto dGSPosBufferAligned = static_cast<float4*>(__builtin_assume_aligned(dGeometryShaderOutPos, 16));
				v1 = dGSPosBufferAligned[globalInvoIdx * 3];
				v2 = dGSPosBufferAligned[globalInvoIdx * 3 + 1];
				v3 = dGSPosBufferAligned[globalInvoIdx * 3 + 2];
				primId = globalInvoIdx; //dGeometryShaderOutSortKeys[globalInvoIdx];
			}
			else {
				if (globalInvoIdx >= indexCount / CU_TRIANGLE_STRIDE) return;
				const auto indexStart = globalInvoIdx * CU_TRIANGLE_STRIDE + startingIndexId;
				auto dPosBufferAligned = static_cast<float4*>(__builtin_assume_aligned(dPosBuffer, 16));
				v1 = dPosBufferAligned[dIndexBuffer[indexStart]];
				v2 = dPosBufferAligned[dIndexBuffer[indexStart + 1]];
				v3 = dPosBufferAligned[dIndexBuffer[indexStart + 2]];
				primId = globalInvoIdx + startingIndexId / CU_TRIANGLE_STRIDE;
			}
			if (v1.w < 0 && v2.w < 0 && v3.w < 0)
				return;
			devGeometryCullClip<geometryShaderEnabled, tpCullMode>(v1, v2, v3, primId);
		}
		IFRIT_KERNEL void geometryClippingKernelEntryWithGS() {
			int numTriangles = dGeometryShaderOutSize / 3;
			if (dGeometryShaderOutSize % 3 != 0) {
				printf("Invalid GS Output. Aborted\n");
				GeneralFunction::devAbort();
			}
			int dispatchBlocks = IFRIT_InvoGetThreadBlocks(numTriangles, CU_GEOMETRY_PROCESSING_THREADS);
			
#define invokeGeometryClippingKernel(tpCullMode) if(csCullMode == tpCullMode) geometryClippingKernel<1,(tpCullMode)> CU_KARG2(dispatchBlocks, CU_GEOMETRY_PROCESSING_THREADS)(nullptr, nullptr, 0, numTriangles * 3);
			invokeGeometryClippingKernel(IF_CULL_MODE_FRONT);
			invokeGeometryClippingKernel(IF_CULL_MODE_BACK);
			invokeGeometryClippingKernel(IF_CULL_MODE_FRONT_AND_BACK);
			invokeGeometryClippingKernel(IF_CULL_MODE_NONE);
#undef invokeGeometryClippingKernel
		}
		IFRIT_KERNEL void geometryParamPostprocKernel(uint32_t bound) {
			int globalInvo = threadIdx.x + blockDim.x * blockIdx.x;
			if (globalInvo >= bound)return;

			const auto dAtriInterpolBase1Aligned = static_cast<float4*>(__builtin_assume_aligned(dAtriInterpolBase1, 16));
			const auto dAtriInterpolBase2Aligned = static_cast<float4*>(__builtin_assume_aligned(dAtriInterpolBase2, 16));
			const auto dAtriInterpolBase3AltnAligned = static_cast<float4*>(__builtin_assume_aligned(dBoundingBoxM2, 16));

			float4 dv1 = dAtriInterpolBase1Aligned[globalInvo];
			float4 dv2 = dAtriInterpolBase2Aligned[globalInvo];
			float4 dv3 = dAtriInterpolBase3AltnAligned[globalInvo];

			const auto frameHeight = csFrameHeight;
			const auto frameWidth = csFrameWidth;
			const auto invFrameHeight = csFrameHeightInv;
			const auto invFrameWidth = csFrameWidthInv;

			const float ar = 1.0f / GeneralFunction::devEdgeFunction(dv1, dv2, dv3);
			const float sV2V1y = dv2.y - dv1.y;
			const float sV2V1x = dv1.x - dv2.x;
			const float sV3V2y = dv3.y - dv2.y;
			const float sV3V2x = dv2.x - dv3.x;
			const float sV1V3y = dv1.y - dv3.y;
			const float sV1V3x = dv3.x - dv1.x;

			float4 f3 = { (float)(sV2V1y * ar) * invFrameWidth, (float)(sV2V1x * ar) * invFrameHeight,(float)((-dv1.x * sV2V1y - dv1.y * sV2V1x) * ar) };
			float4 f1 = { (float)(sV3V2y * ar) * invFrameWidth, (float)(sV3V2x * ar) * invFrameHeight,(float)((-dv2.x * sV3V2y - dv2.y * sV3V2x) * ar) };
			float4 f2 = { (float)(sV1V3y * ar) * invFrameWidth, (float)(sV1V3x * ar) * invFrameHeight,(float)((-dv3.x * sV1V3y - dv3.y * sV1V3x) * ar) };

			constexpr auto dEps = -CU_EPS * 7.85e5f;

			float v1 = dv1.z * f1.x + dv2.z * f2.x + dv3.z * f3.x;
			float v2 = dv1.z * f1.y + dv2.z * f2.y + dv3.z * f3.y;
			float v3 = dv1.z * f1.z + dv2.z * f2.z + dv3.z * f3.z;

			f1.x *= dv1.w;
			f1.y *= dv1.w;
			f1.z *= dv1.w;

			f2.x *= dv2.w;
			f2.y *= dv2.w;
			f2.z *= dv2.w;

			f3.x *= dv3.w;
			f3.y *= dv3.w;
			f3.z *= dv3.w;

			const auto dAtriEdgeCoefs1Aligned = static_cast<float4*>(__builtin_assume_aligned(dAtriEdgeCoefs1, 16));
			const auto dAtriEdgeCoefs2Aligned = static_cast<float4*>(__builtin_assume_aligned(dAtriEdgeCoefs2, 16));
			const auto dAtriEdgeCoefs3Aligned = dAtriEdgeCoefs3;

			dAtriEdgeCoefs1Aligned[globalInvo] = {
				(float)(sV2V1y)*frameHeight,
				(float)(sV2V1x)*frameWidth ,
				(float)(-dv2.x * dv1.y + dv1.x * dv2.y) * frameHeight * frameWidth + dEps,
				(float)(sV3V2y)*frameHeight
			};
			dAtriEdgeCoefs2Aligned[globalInvo] = {
				(float)(sV3V2x)*frameWidth ,
				(float)(-dv3.x * dv2.y + dv2.x * dv3.y) * frameHeight * frameWidth + dEps,
				(float)(sV1V3y)*frameHeight,
				(float)(sV1V3x)*frameWidth ,
			};
			dAtriEdgeCoefs3Aligned[globalInvo] = (float)(-dv1.x * dv3.y + dv3.x * dv1.y) * frameHeight * frameWidth + dEps;

			dAtriInterpolBase1Aligned[globalInvo] = { f1.x,f1.y,f1.z,f2.x };
			dAtriInterpolBase2Aligned[globalInvo] = { f2.y,f2.z,f3.x,f3.y };
			dAtriInterpolBase3[globalInvo] = f3.z;

			dAtriDepthVal12[globalInvo].x = v1;
			dAtriDepthVal12[globalInvo].y = v2;
			dAtriDepthVal3[globalInvo] = v3;

			float4 bbox;
			GeneralFunction::devGetBBox(dv1, dv2, dv3, bbox);

			const auto dBoundingBoxM2Aligned = static_cast<float4*>(__builtin_assume_aligned(dBoundingBoxM2, 16));
			dBoundingBoxM2Aligned[globalInvo] = bbox;
		}
	}
	namespace TriangleFragmentStage {
		
		template<uint32_t tpDepthFunc, bool tpGeometryShaderEnabled>
		IFRIT_KERNEL void pixelShadingAlphaBlendKernel(
			FragmentShader* fragmentShader,
			int* IFRIT_RESTRICT_CUDA dIndexBuffer,
			const float4* IFRIT_RESTRICT_CUDA dVaryingBuffer,
			ifloat4** IFRIT_RESTRICT_CUDA dColorBuffer,
			float* IFRIT_RESTRICT_CUDA dDepthBuffer,
			ImplBlendCoefs blendParam,
			ImplBlendCoefs blendParamAlpha

		) {
			//TODO: Geometry shader integration
			
			// Process pixel-level data
			const int pixelXS = (threadIdx.x & 1) + (threadIdx.y + blockDim.y * blockIdx.y) * 2;
			const int pixelYS = (threadIdx.x >> 1) + (threadIdx.z + blockDim.z * blockIdx.z) * 2;
			const auto pixelId = pixelYS * CU_MAX_FRAMEBUFFER_WIDTH + pixelXS;
			const auto tileX = pixelXS / CU_TILE_WIDTH;
			const auto tileY = pixelYS / CU_TILE_WIDTH;
			const auto tileId = tileY * CU_MAX_TILE_X + tileX;
			if (pixelXS >= csFrameWidth || pixelYS >= csFrameHeight)return;

			auto inTileX = pixelXS % CU_TILE_WIDTH;
			auto inTileY = pixelYS % CU_TILE_WIDTH;
			auto inSubtileX = inTileX / CU_EXPERIMENTAL_SUBTILE_WIDTH;
			auto inSubtileY = inTileY / CU_EXPERIMENTAL_SUBTILE_WIDTH;
			constexpr int numSubtilesX = CU_TILE_WIDTH / CU_EXPERIMENTAL_SUBTILE_WIDTH;
			auto inSubtileId = inSubtileY * numSubtilesX + inSubtileX;

			auto dwX = inTileX % CU_EXPERIMENTAL_SUBTILE_WIDTH;
			auto dwY = inTileY % CU_EXPERIMENTAL_SUBTILE_WIDTH;
			auto dwId = CU_EXPERIMENTAL_SUBTILE_WIDTH * dwY + dwX;
			auto dwMask = (1 << dwId);

			auto dsMask = 0;
			auto dsX = (dwX >> 1) << 1;
			auto dsY = (dwY >> 1) << 1;
			for (int i = 0; i < 4; i++) {
				auto dsXv = dsX + (i & 1);
				auto dsYv = dsY + (i >> 1);
				auto dsIdv = CU_EXPERIMENTAL_SUBTILE_WIDTH * dsYv + dsXv;
				dsMask |= (1 << dsIdv);
			}

			// Process tile-level data
			auto largeTileX = pixelXS / CU_LARGE_BIN_WIDTH;
			auto largeTileY = pixelYS / CU_LARGE_BIN_WIDTH;
			auto largeTileId = largeTileY * CU_MAX_LARGE_BIN_X + largeTileX;
			auto binX = pixelXS / CU_BIN_WIDTH;
			auto binY = pixelYS / CU_BIN_WIDTH;
			auto binId = binY * CU_MAX_BIN_X + binX;

			// Shading
			IFRIT_SHARED float colorOutputSingle[256 * 4];
			IFRIT_SHARED float interpolatedVaryings[256 * 4 * CU_MAX_VARYINGS];
			float candidateBary[3];
			constexpr auto vertexStride = CU_TRIANGLE_STRIDE;
			auto varyingCount = csVaryingCounts;
			//TODO: Bug
			const auto threadId = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
			const auto pDx = pixelXS * 1.0f;
			const auto pDy = pixelYS * 1.0f;
			const float compareDepth = dDepthBuffer[pixelYS * csFrameWidth + pixelXS];

			auto getDepth = [&](int primId) {
				float interpolatedDepth;
				float2 v12 = dAtriDepthVal12[primId];
				float v3 = dAtriDepthVal3[primId];
				interpolatedDepth = v12.x * pDx + v12.y * pDy + v3;
				return interpolatedDepth;
			};

			auto shadingPass = [&](const int pId, const bool isHelperInvocation) {
				float4 f1o = dAtriInterpolBase1[pId];
				float4 f2o = dAtriInterpolBase2[pId];
				float f3o = dAtriInterpolBase3[pId];

				float3 f1 = { f1o.x,f1o.y,f1o.z };
				float3 f2 = { f1o.w,f2o.x,f2o.y };
				float3 f3 = { f2o.z,f2o.w,f3o };

				float4 b12 = dAtriBaryCenter12[pId];
				float2 b3 = dAtriBaryCenter3[pId];
				int orgPrimId = dAtriOriginalPrimId[pId];
				if constexpr (tpGeometryShaderEnabled) {
					orgPrimId = dAtriOriginalPrimIdGs[pId];
				}

				candidateBary[0] = (f1.x * pDx + f1.y * pDy + f1.z);
				candidateBary[1] = (f2.x * pDx + f2.y * pDy + f2.z);
				candidateBary[2] = (f3.x * pDx + f3.y * pDy + f3.z);
				float zCorr = 1.0f / (candidateBary[0] + candidateBary[1] + candidateBary[2]);
				candidateBary[0] *= zCorr;
				candidateBary[1] *= zCorr;
				candidateBary[2] *= zCorr;

				float desiredBary[3];
				desiredBary[0] = candidateBary[0] * b12.x + candidateBary[1] * b12.z + candidateBary[2] * b3.x;
				desiredBary[1] = candidateBary[0] * b12.y + candidateBary[1] * b12.w + candidateBary[2] * b3.y;
				desiredBary[2] = 1.0f - (desiredBary[0] + desiredBary[1]);
				auto addr = dIndexBuffer + orgPrimId * vertexStride;

				for (int k = 0; k < varyingCount; k++) {
					const auto va = dVaryingBuffer;
					float4 vd;
					vd = { 0,0,0,0 };
					for (int j = 0; j < 3; j++) {
						float4 vaf4;
						if constexpr (tpGeometryShaderEnabled) {
							vaf4 = dGeometryShaderOutVaryings[orgPrimId * 3 * varyingCount + k + j * varyingCount];
						}
						else {
							vaf4 = va[addr[j] * varyingCount + k];
						}
						vd.x += vaf4.x * desiredBary[j];
						vd.y += vaf4.y * desiredBary[j];
						vd.z += vaf4.z * desiredBary[j];
						vd.w += vaf4.w * desiredBary[j];
					}
					auto dest = (ifloat4s256*)(interpolatedVaryings + 1024 * k + threadId);
					dest->x = vd.x;
					dest->y = vd.y;
					dest->z = vd.z;
					dest->w = vd.w;
				}
				if (isHelperInvocation) {
					return;
				}
				auto tDepth = getDepth(pId);
				bool depthTestCond = true;

				if constexpr (tpDepthFunc == IF_COMPARE_OP_ALWAYS) depthTestCond = true;
				if constexpr (tpDepthFunc == IF_COMPARE_OP_NEVER) depthTestCond = false;
				if constexpr (tpDepthFunc == IF_COMPARE_OP_LESS) depthTestCond = (tDepth < compareDepth);
				if constexpr (tpDepthFunc == IF_COMPARE_OP_EQUAL) depthTestCond = (tDepth == compareDepth);
				if constexpr (tpDepthFunc == IF_COMPARE_OP_GREATER) depthTestCond = (tDepth > compareDepth);
				if constexpr (tpDepthFunc == IF_COMPARE_OP_LESS_OR_EQUAL) depthTestCond = (tDepth <= compareDepth);
				if constexpr (tpDepthFunc == IF_COMPARE_OP_GREATER_OR_EQUAL) depthTestCond = (tDepth >= compareDepth);
				if constexpr (tpDepthFunc == IF_COMPARE_OP_NOT_EQUAL) depthTestCond = (tDepth != compareDepth);

				if (!depthTestCond)return;
				fragmentShader->execute(interpolatedVaryings + threadId, colorOutputSingle + threadId);
				auto col0 = static_cast<ifloat4*>(__builtin_assume_aligned(dColorBuffer[0], 16));
				ifloat4 srcRgba;
				ifloat4 dstRgba = col0[pixelYS * csFrameWidth + pixelXS];
				ifloat4 mixRgba;
				ifloat4s256 midOutput = ((ifloat4s256*)(colorOutputSingle + threadId))[0];
				srcRgba.x = midOutput.x;
				srcRgba.y = midOutput.y;
				srcRgba.z = midOutput.z;
				srcRgba.w = midOutput.w;

				// Blend
				auto mxSrcX = (blendParam.s.x + blendParam.s.y * (1 - srcRgba.w) + blendParam.s.z * (1 - dstRgba.w)) * (1-blendParam.s.w);
				auto mxDstX = (blendParam.d.x + blendParam.d.y * (1 - srcRgba.w) + blendParam.d.z * (1 - dstRgba.w)) * (1-blendParam.d.w);
				auto mxSrcA = (blendParamAlpha.s.x + blendParamAlpha.s.y * (1 - srcRgba.w) + blendParamAlpha.s.z * (1 - dstRgba.w)) * (1-blendParamAlpha.s.w);
				auto mxDstA = (blendParamAlpha.d.x + blendParamAlpha.d.y * (1 - srcRgba.w) + blendParamAlpha.d.z * (1 - dstRgba.w)) * (1-blendParamAlpha.d.w);
	
				mixRgba.x = dstRgba.x * mxDstX + srcRgba.x * mxSrcX;
				mixRgba.y = dstRgba.y * mxDstX + srcRgba.y * mxSrcX;
				mixRgba.z = dstRgba.z * mxDstX + srcRgba.z * mxSrcX;
				mixRgba.w = dstRgba.w * mxDstA + srcRgba.w * mxSrcA;

				col0[pixelYS * csFrameWidth + pixelXS] = mixRgba;
				dDepthBuffer[pixelYS * csFrameWidth + pixelXS] = tDepth;
				if constexpr (CU_PROFILER_OVERDRAW) {
					atomicAdd(&dOverDrawCounter, 1);
				}
			};

			// Get cover queue data
			const auto pxId = tileId * CU_MAX_SUBTILES_PER_TILE + inSubtileId;
			auto pxStart = dSecondBinnerFinerBufferStart[pxId];
			auto pxEnd = dSecondBinnerFinerBufferCurInd[pxId];
			auto lxStart = dCoverQueueSuperTileFullM3Start[largeTileId];
			auto lxEnd = dCoverQueueSuperTileFullM3CurInd[largeTileId];
			auto bxStart = dCoverQueueFullM2Start[binId];
			auto bxEnd = dCoverQueueFullM2CurInd[binId];
			auto pxCur = pxStart, lxCur = lxStart, bxCur = bxStart;

			auto pxOrgPrim = INT_MAX;
			auto lxOrgPrim = INT_MAX;
			auto bxOrgPrim = INT_MAX;
			auto chosenOrgPrim = INT_MAX;
			auto chosenPrim = 0;

			while (true) {
				chosenOrgPrim = INT_MAX;
				pxOrgPrim = INT_MAX;
				lxOrgPrim = INT_MAX;
				bxOrgPrim = INT_MAX;
				if (pxCur < pxEnd) {
					auto dr = dSecondBinnerFinerBuffer[pxCur].y;
					pxOrgPrim = dAtriOriginalPrimId[dr];
				}
				if (lxCur < lxEnd) {
					auto dr = dCoverQueueSuperTileFullM3BufferFinal[lxCur];
					lxOrgPrim = dAtriOriginalPrimId[dr];
				}
				if (bxCur < bxEnd) {
					auto dr = dCoverQueueFullM2Buffer[bxCur];
					bxOrgPrim = dAtriOriginalPrimId[dr];
				}

				//Process Target Mask
				auto targetMask = -1;
				if (pxOrgPrim != INT_MAX && pxOrgPrim < lxOrgPrim && pxOrgPrim < bxOrgPrim) {
					targetMask = dSecondBinnerFinerBuffer[pxCur].x;
					chosenOrgPrim = pxOrgPrim;
					chosenPrim = dSecondBinnerFinerBuffer[pxCur].y;
					pxCur++;
				}
				else if (lxOrgPrim != INT_MAX && lxOrgPrim < pxOrgPrim && lxOrgPrim < bxOrgPrim) {
					targetMask = -1;
					chosenOrgPrim = lxOrgPrim;
					chosenPrim = dCoverQueueSuperTileFullM3BufferFinal[lxCur];
					lxCur++;
				}
				else if(bxOrgPrim != INT_MAX && bxOrgPrim < pxOrgPrim && bxOrgPrim < lxOrgPrim){
					targetMask = -1;
					chosenOrgPrim = bxOrgPrim;
					chosenPrim = dCoverQueueFullM2Buffer[bxCur];
					bxCur++;
				}
				if (chosenOrgPrim == INT_MAX)break;
				bool reqCalc = targetMask & dsMask;
				bool reqDraw = targetMask & dwMask;
				if (reqDraw) {
					shadingPass(chosenPrim, !reqDraw);
				}
			}
		}
		
		// tpDepthFunc: 0-Never / 1-Less / 2-Eq / 3-Leq / 4-Greater / 5-Neq / 6-Geq / 7-Always
		template<uint32_t tpDepthFunc>
		IFRIT_KERNEL void pixelTaggingExecKernel(
			FragmentShader* fragmentShader,
			int* IFRIT_RESTRICT_CUDA dIndexBuffer,
			const float4* IFRIT_RESTRICT_CUDA dVaryingBuffer,
			ifloat4** IFRIT_RESTRICT_CUDA dColorBuffer,
			float* IFRIT_RESTRICT_CUDA dDepthBuffer
		) {
			uint32_t tileX = blockIdx.x, tileY = blockIdx.y;
			uint32_t binX = tileX / CU_TILES_PER_BIN, binY = tileY / CU_TILES_PER_BIN;
			uint32_t superTileX = tileX / (CU_BINS_PER_LARGE_BIN * CU_TILES_PER_BIN);
			uint32_t superTileY = tileY / (CU_BINS_PER_LARGE_BIN * CU_TILES_PER_BIN);
			uint32_t tileId = tileY * CU_MAX_TILE_X + tileX;
			uint32_t binId = binY * CU_MAX_BIN_X + binX;
			uint32_t superTileId = superTileY * CU_MAX_LARGE_BIN_X + superTileX;

			const auto frameWidth = csFrameWidth;
			//const auto completeCandidates = dCoverQueueFullM2[binId].size;

			constexpr auto vertexStride = CU_TRIANGLE_STRIDE;
			const auto varyingCount = csVaryingCounts;

			const int threadX = threadIdx.x, threadY = threadIdx.y, threadZ = threadIdx.z;
			const auto threadId = threadZ * blockDim.x * blockDim.y + threadY * blockDim.x + threadX;

			const int pixelXS = threadX + threadZ * CU_EXPERIMENTAL_SUBTILE_WIDTH + tileX * CU_TILE_WIDTH;
			const int pixelYS = threadY + tileY * CU_TILE_WIDTH;

			float candidateBary[3];
			int candidatePrim = -1;
			const float compareDepth = dDepthBuffer[pixelYS * frameWidth + pixelXS];
			float localDepthBuffer = compareDepth;
			int localOrgPrimId = INT_MAX;
			float pDx = 1.0f * pixelXS, pDy = 1.0f * pixelYS;

			auto zPrePass = [&](int primId) {
				if constexpr (tpDepthFunc == IF_COMPARE_OP_NEVER) return;
				float interpolatedDepth;
				float2 v12 = dAtriDepthVal12[primId];
				float v3 = dAtriDepthVal3[primId];
				interpolatedDepth = v12.x * pDx + v12.y * pDy + v3;

#define PLACE_COND(cond) if ((cond)) { localDepthBuffer = interpolatedDepth; candidatePrim = primId;}
				if constexpr (tpDepthFunc == IF_COMPARE_OP_LESS) PLACE_COND(interpolatedDepth < localDepthBuffer);
				if constexpr (tpDepthFunc == IF_COMPARE_OP_LESS_OR_EQUAL)PLACE_COND(interpolatedDepth <= localDepthBuffer);
				if constexpr (tpDepthFunc == IF_COMPARE_OP_EQUAL) PLACE_COND(interpolatedDepth == localDepthBuffer);
				if constexpr (tpDepthFunc == IF_COMPARE_OP_GREATER) PLACE_COND(interpolatedDepth > localDepthBuffer);
				if constexpr (tpDepthFunc == IF_COMPARE_OP_GREATER_OR_EQUAL) PLACE_COND(interpolatedDepth >= localDepthBuffer);
				if constexpr (tpDepthFunc == IF_COMPARE_OP_NOT_EQUAL) PLACE_COND(interpolatedDepth != localDepthBuffer);
				if constexpr (tpDepthFunc == IF_COMPARE_OP_ALWAYS) {
					if (dAtriOriginalPrimId[primId] < localOrgPrimId) {
						localDepthBuffer = interpolatedDepth;
						localOrgPrimId = dAtriOriginalPrimId[primId];
						candidatePrim = primId;
					}
				}

#undef PLACE_COND
				if constexpr (CU_PROFILER_OVERDRAW) {
					atomicAdd(&dOverZTestCounter, 1);
				}
			};

			// Large Bin Level
			if constexpr (true) {
				int startIndex = dCoverQueueSuperTileFullM3Start[superTileId];
				int totlSize = dCoverQueueSuperTileFullM3CurInd[superTileId];
				for (int i = startIndex; i < totlSize; i++) {
					const auto proposal = dCoverQueueSuperTileFullM3BufferFinal[i];
					zPrePass(proposal);
				}
			}

			// Bin Level
			if constexpr (true) {
				int startIndex = dCoverQueueFullM2Start[binId];
				int totlSize = dCoverQueueFullM2CurInd[binId];
				for (int i = startIndex; i < totlSize; i++) {
					const auto proposal = dCoverQueueFullM2Buffer[i];
					zPrePass(proposal);
				}
			}

			// Pixel Level
			if constexpr (true) {
				auto curTileX = tileX * CU_TILE_WIDTH;
				auto curTileY = tileY * CU_TILE_WIDTH;
				constexpr auto curTileWid = CU_TILE_WIDTH;
				constexpr int numSubtilesX = CU_TILE_WIDTH / CU_EXPERIMENTAL_SUBTILE_WIDTH;
				int inTileX = pixelXS - curTileX;
				int inTileY = pixelYS - curTileY;
				int inSubTileX = inTileX / CU_EXPERIMENTAL_SUBTILE_WIDTH;
				int inSubTileY = inTileY / CU_EXPERIMENTAL_SUBTILE_WIDTH;
				int inSubTileId = inSubTileY * numSubtilesX + inSubTileX;

				int dwX = inTileX % CU_EXPERIMENTAL_SUBTILE_WIDTH;
				int dwY = inTileY % CU_EXPERIMENTAL_SUBTILE_WIDTH;
				int dwId = dwY * CU_EXPERIMENTAL_SUBTILE_WIDTH + dwX;
				int dwMask = (1 << dwId);

				int startIndex = dSecondBinnerFinerBufferStart[tileId * CU_MAX_SUBTILES_PER_TILE + inSubTileId];
				int totlSize = dSecondBinnerFinerBufferCurInd[tileId * CU_MAX_SUBTILES_PER_TILE + inSubTileId];
				for (int i = startIndex; i < totlSize; i++) {
					const auto proposal = dSecondBinnerFinerBuffer[i];
					if ((proposal.x & dwMask)) {
						zPrePass(proposal.y);
					}
					else {
						if constexpr (CU_PROFILER_OVERDRAW) {
							atomicAdd(&dIrrelevantPixel, 1);
						}
					}
					if constexpr (CU_PROFILER_OVERDRAW) {
						atomicAdd(&dTotalBinnedPixel, 1);
					}
				}
			}
			if constexpr (tpDepthFunc == IF_COMPARE_OP_ALWAYS) {
				if (candidatePrim != -1) {
					dDepthBuffer[pixelYS * frameWidth + pixelXS] = localDepthBuffer;
				}
			}
			else {
				if (candidatePrim != -1 && localDepthBuffer < compareDepth) {
					dDepthBuffer[pixelYS * frameWidth + pixelXS] = localDepthBuffer;
				}
			}
			int quadIdX = pixelXS >> 1;
			int quadOfX = pixelXS & 1;
			int quadIdY = pixelYS >> 1;
			int quadOfY = pixelYS & 1;
			int quadId = (quadIdY * CU_MAX_FRAMEBUFFER_WIDTH + quadIdX) * 4 + (quadOfY * 2 + quadOfX);
			dPixelShaderTags[quadId] = candidatePrim;
			
		}

		template <int geometryShaderEnabled>
		IFRIT_KERNEL void pixelShadingExecKernel(
			FragmentShader* fragmentShader,
			int* IFRIT_RESTRICT_CUDA dIndexBuffer,
			const float4* IFRIT_RESTRICT_CUDA dVaryingBuffer,
			ifloat4** IFRIT_RESTRICT_CUDA dColorBuffer,
			float* IFRIT_RESTRICT_CUDA dDepthBuffer
		) {
			IFRIT_SHARED float colorOutputSingle[256 * 4];
			IFRIT_SHARED float interpolatedVaryings[256 * 4 * CU_MAX_VARYINGS];
			float candidateBary[3];

			const int pixelXS = (threadIdx.x & 1) + (threadIdx.y + blockDim.y * blockIdx.y) * 2;
			const int pixelYS = (threadIdx.x >> 1) + (threadIdx.z + blockDim.z * blockIdx.z) * 2;

			if (pixelXS >= csFrameWidth || pixelYS >= csFrameHeight)return;
			float pDx = 1.0f * pixelXS, pDy = 1.0f * pixelYS;
			constexpr auto vertexStride = CU_TRIANGLE_STRIDE;
			auto varyingCount = csVaryingCounts;
			
			const auto threadId = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

			auto shadingPass = [&](const int pId, const bool isHelperInvocation) {
				float4 f1o = dAtriInterpolBase1[pId];
				float4 f2o = dAtriInterpolBase2[pId];
				float f3o = dAtriInterpolBase3[pId];

				float3 f1 = { f1o.x,f1o.y,f1o.z };
				float3 f2 = { f1o.w,f2o.x,f2o.y };
				float3 f3 = { f2o.z,f2o.w,f3o };

				float4 b12 = dAtriBaryCenter12[pId];
				float2 b3 = dAtriBaryCenter3[pId];
				int orgPrimId = dAtriOriginalPrimId[pId];
				if constexpr (geometryShaderEnabled) {
					orgPrimId = dAtriOriginalPrimIdGs[pId];
				}

				candidateBary[0] = (f1.x * pDx + f1.y * pDy + f1.z);
				candidateBary[1] = (f2.x * pDx + f2.y * pDy + f2.z);
				candidateBary[2] = (f3.x * pDx + f3.y * pDy + f3.z);
				float zCorr = 1.0f / (candidateBary[0] + candidateBary[1] + candidateBary[2]);
				candidateBary[0] *= zCorr;
				candidateBary[1] *= zCorr;
				candidateBary[2] *= zCorr;

				float desiredBary[3];
				desiredBary[0] = candidateBary[0] * b12.x + candidateBary[1] * b12.z + candidateBary[2] * b3.x;
				desiredBary[1] = candidateBary[0] * b12.y + candidateBary[1] * b12.w + candidateBary[2] * b3.y;
				desiredBary[2] = 1.0f - (desiredBary[0] + desiredBary[1]);
				auto addr = dIndexBuffer + orgPrimId * vertexStride;

				for (int k = 0; k < varyingCount; k++) {
					const auto va = dVaryingBuffer;
					float4 vd;
					vd = { 0,0,0,0 };
					for (int j = 0; j < 3; j++) {
						float4 vaf4;
						if constexpr (geometryShaderEnabled) {
							vaf4 = dGeometryShaderOutVaryings[orgPrimId * 3 * varyingCount + k + j * varyingCount];
						}
						else {
							vaf4 = va[addr[j] * varyingCount + k];
						}
						vd.x += vaf4.x * desiredBary[j];
						vd.y += vaf4.y * desiredBary[j];
						vd.z += vaf4.z * desiredBary[j];
						vd.w += vaf4.w * desiredBary[j];
					}
					auto dest = (ifloat4s256*)(interpolatedVaryings + 1024 * k + threadId);
					dest->x = vd.x;
					dest->y = vd.y;
					dest->z = vd.z;
					dest->w = vd.w;
				}
				if (isHelperInvocation) {
					return;
				}
				fragmentShader->execute(interpolatedVaryings + threadId, colorOutputSingle + threadId);
				auto col0 = static_cast<ifloat4*>(__builtin_assume_aligned(dColorBuffer[0], 16));
				ifloat4 finalRgba;
				ifloat4s256 midOutput = ((ifloat4s256*)(colorOutputSingle + threadId))[0];
				finalRgba.x = midOutput.x;
				finalRgba.y = midOutput.y;
				finalRgba.z = midOutput.z;
				finalRgba.w = midOutput.w;

				col0[pixelYS * csFrameWidth + pixelXS] = finalRgba;
				if constexpr (CU_PROFILER_OVERDRAW) {
					atomicAdd(&dOverDrawCounter, 1);
				}
				};

			//Process Quad
			int dw[4];
			int quadInternal = threadIdx.x;
			int quadX = (threadIdx.y + blockDim.y * blockIdx.y);
			int quadY = (threadIdx.z + blockDim.z * blockIdx.z);
			int quadIdx = (quadY * CU_MAX_FRAMEBUFFER_WIDTH + quadX) * 4;
			int selfV = dPixelShaderTags[quadIdx + quadInternal];
			if constexpr (CU_OPT_SHADER_DERIVATIVES) {
				for (int i = 0; i < 4; i++) {
					int tagId = dPixelShaderTags[quadIdx + i];
					dw[i] = tagId;
					if (tagId == -1)continue;
					bool flag = false;
					for (int j = 0; j < i; j++) {
						if (tagId == dw[j])flag = true;
					}
					if (flag)continue;
					shadingPass(tagId, tagId != selfV);
				}
			}
			else {
				if (selfV == -1)return;
				shadingPass(selfV, false);
			}

		}
	}
	namespace TriangleMiscStage {
		IFRIT_KERNEL void integratedResetKernel() {
			const auto globalInvocation = blockIdx.x * blockDim.x + threadIdx.x;

			dSecondBinnerFinerBufferStart[globalInvocation] = 0;
			dSecondBinnerFinerBufferSize[globalInvocation] = 0;

			if (globalInvocation < CU_MAX_LARGE_BIN_X * CU_MAX_LARGE_BIN_X) {
				dCoverQueueSuperTileFullM3Start[globalInvocation] = 0;
				dCoverQueueSuperTileFullM3Size[globalInvocation] = 0;
			}
			if (globalInvocation < CU_MAX_BIN_X * CU_MAX_BIN_X) {
				//dCoverQueueFullM2[globalInvocation].clear();
				dCoverQueueFullM2Size[globalInvocation] = 0;
			}
			if (globalInvocation == 0) {
				dAssembledTriangleCounterM2 = 0;
				dGeometryShaderOutSize = 0;
				dRasterQueueWorklistCounter = 0;
				dSmallTriangleCount = 0;
				dCoverPrimsCounter = 0;
				dCoverPrimsLargeTileCounter = 0;
				dSecondBinnerFinerBufferGlobalSize = 0;
				dSecondBinnerCandCounter = 0;
				dCoverQueueSuperTileFullM3GlobalSize = 0;
				dCoverQueueSuperTileFullM3TotlCands = 0;
				dCoverQueueFullM2GlobalSize = 0;
			}
			if constexpr (CU_PROFILER_OVERDRAW) {
				if (blockIdx.x == 0 && threadIdx.x == 0) {
					printf("Overdraw Rate:%f (%d)\n", dOverDrawCounter * 1.0f / 2048.0f / 2048.0f, dOverDrawCounter);
					printf("ZTest Rate:%f (%d)\n", dOverZTestCounter * 1.0f / 2048.0f / 2048.0f, dOverZTestCounter);
					printf("Irrelevant Rate:%f (%d/%d)\n", 1.0f * dIrrelevantPixel / dTotalBinnedPixel, dIrrelevantPixel, dTotalBinnedPixel);

					dOverDrawCounter = 0;
					dOverZTestCounter = 0;
					dIrrelevantPixel = 0;
					dTotalBinnedPixel = 0;
				}
			}
		}
		IFRIT_KERNEL void integratedInitKernel() {
			const auto globalInvocation = blockIdx.x * blockDim.x + threadIdx.x;
			if (globalInvocation == 0) {
				dAssembledTriangleCounterM2 = 0;
				dSecondBinnerActiveReqs = 0;
				dSecondBinnerTotalReqs = 0;
				dSecondBinnerEmptyTiles = 0;
				dRasterQueueWorklistCounter = 0;
				dSecondBinnerCandCounter = 0;
				dCoverQueueSuperTileFullM3GlobalSize = 0;
				dCoverQueueSuperTileFullM3TotlCands = 0;
			}
		}

		IFRIT_KERNEL void resetLargeTileKernel(bool resetTriangle) {
			const auto globalInvocation = blockIdx.x * blockDim.x + threadIdx.x;

			dSecondBinnerFinerBufferStart[globalInvocation] = 0;
			dSecondBinnerFinerBufferSize[globalInvocation] = 0;

			if (globalInvocation < CU_MAX_LARGE_BIN_X * CU_MAX_LARGE_BIN_X) {
				dCoverQueueSuperTileFullM3Start[globalInvocation] = 0;
				dCoverQueueSuperTileFullM3Size[globalInvocation] = 0;

			}
			if (globalInvocation < CU_MAX_BIN_X * CU_MAX_BIN_X) {
				dCoverQueueFullM2Size[globalInvocation] = 0;
			}

			if constexpr (CU_PROFILER_SECOND_BINNER_UTILIZATION) {
				if (globalInvocation == 0) {
					//printf("Tile Empty Rate: %f (%d/%d)\n",
					//	1.0f * dSecondBinnerEmptyTiles / (CU_TILE_SIZE * CU_TILE_SIZE), dSecondBinnerEmptyTiles, CU_TILE_SIZE * CU_TILE_SIZE);
					printf("Second Binner Actual Utilization: %f (%d/%d)\n",
						1.0f * dSecondBinnerActiveReqs / dSecondBinnerTotalReqs, dSecondBinnerActiveReqs, dSecondBinnerTotalReqs);
					dSecondBinnerActiveReqs = 0;
					dSecondBinnerTotalReqs = 0;
					dSecondBinnerEmptyTiles = 0;
				}
			}

			if (globalInvocation == 0) {
				if (resetTriangle) {
					dAssembledTriangleCounterM2 = 0;
					dGeometryShaderOutSize = 0;
					if constexpr (CU_PROFILER_TRIANGLE_SETUP) {
						printf("Primitive Queue Occupation: %f (%d/%d)\n",
							1.0f * dAssembledTriangleCounterM2 / (CU_SINGLE_TIME_TRIANGLE * 2), dAssembledTriangleCounterM2, CU_SINGLE_TIME_TRIANGLE * 2);
					}
				}
				dRasterQueueWorklistCounter = 0;
				dCoverPrimsCounter = 0;
				dCoverPrimsLargeTileCounter = 0;
				dSecondBinnerCandCounter = 0;
				dSecondBinnerFinerBufferGlobalSize = 0;
				dCoverQueueSuperTileFullM3GlobalSize = 0;
				dCoverQueueSuperTileFullM3TotlCands = 0;
				dCoverQueueFullM2GlobalSize = 0;
			}
		}
	}
	namespace TrianglePipeline {
		IFRIT_KERNEL void unifiedRasterEngineStageIIFilledTriangleKernel(
			int* dIndexBuffer,
			const float4 const* dVaryingBuffer,
			ifloat4** dColorBuffer,
			float* dDepthBuffer,
			GeometryShader* dGeometryShader,
			FragmentShader* dFragmentShader,
			bool isTailCall
		) {
			if constexpr (CU_OPT_II_SKIP_ON_FEW_GEOMETRIES) {
				if (!isTailCall && dAssembledTriangleCounterM2 < CU_EXPERIMENTAL_II_FEW_GEOMETRIES_LIMIT) {
					return;
				}
			}
			int totalTms = dAssembledTriangleCounterM2 / CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER;
			int curTime = -1;
			auto dispatchBlocks = dAssembledTriangleCounterM2 / CU_EXPERIMENTAL_GEOMETRY_POSTPROC_THREADS + (dAssembledTriangleCounterM2 % CU_EXPERIMENTAL_GEOMETRY_POSTPROC_THREADS != 0);
			Impl::TriangleGeometryStage::geometryParamPostprocKernel CU_KARG2(dispatchBlocks, CU_EXPERIMENTAL_GEOMETRY_POSTPROC_THREADS) (dAssembledTriangleCounterM2);

			for (int sI = 0; sI < dAssembledTriangleCounterM2; sI += CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER) {
				// Process sizes
				curTime++;
				int length = min(dAssembledTriangleCounterM2 - sI, CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER);
				int start = sI;
				bool isLast = curTime == totalTms;
				if (curTime == totalTms - 1) {
					if (dAssembledTriangleCounterM2 % CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER < CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER * 1 / 2) {
						length = dAssembledTriangleCounterM2 - sI;
						sI += CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER;
						isLast = true;
					}
				}
				// First binner
				Impl::TriangleRasterizationStage::firstBinnerRasterizerEntryKernel CU_KARG2(1, 1)(start, length);
				Impl::TriangleRasterizationStage::firstBinnerGatherAllocKernel CU_KARG2(CU_MAX_BIN_X, CU_MAX_BIN_X)();
				Impl::TriangleRasterizationStage::firstBinnerGatherEntryKernel CU_KARG2(1, 1)();
				
				// Second binner
				Impl::TriangleRasterizationStage::secondBinnerWorklistRasterizerEntryKernel CU_KARG2(1, 1)();
				int dispatchSfbaBlocks = IFRIT_InvoGetThreadBlocks(CU_MAX_TILE_X * CU_MAX_TILE_X * CU_MAX_SUBTILES_PER_TILE, 128);
				Impl::TriangleRasterizationStage::secondFinerBinnerAllocationKernel CU_KARG2(dispatchSfbaBlocks, 128) ();
				Impl::TriangleRasterizationStage::secondFinerBinnerRasterizationEntryKernel CU_KARG2(1, 1)();
				int numTileX = (csFrameWidth / CU_TILE_WIDTH) + (csFrameWidth % CU_TILE_WIDTH != 0);
				int numTileY = (csFrameHeight / CU_TILE_WIDTH) + (csFrameHeight % CU_TILE_WIDTH != 0);
				int dispZ = (CU_TILE_WIDTH / CU_EXPERIMENTAL_SUBTILE_WIDTH) + (CU_TILE_WIDTH % CU_EXPERIMENTAL_SUBTILE_WIDTH != 0);
				
				// Sort
				if constexpr (CU_OPT_FORCE_DETERMINISTIC_BEHAVIOR) {
					Impl::TriangleSortStage::sortEntryKernel CU_KARG2(1, 1)();
				}
				else {
					bool forceSort = csBlendState.blendEnable;
					if (forceSort) {
						Impl::TriangleSortStage::sortEntryKernel CU_KARG2(1, 1)();
					}
				}
				
				// Fragment Shader
				int quadX = IFRIT_InvoGetThreadBlocks(csFrameWidth >>1, 8);
				int quadY = IFRIT_InvoGetThreadBlocks(csFrameHeight >>1, 8);
				if (csBlendState.blendEnable) {

#define PIXEL_BLEND_FUNC_SIGN(cond,gsState) Impl::TriangleFragmentStage::pixelShadingAlphaBlendKernel<cond,gsState>
#define PIXEL_BLEND_FUNC(cond,gsState) PIXEL_BLEND_FUNC_SIGN(cond,gsState) CU_KARG2(dim3(1, quadX, quadY), dim3(4, 8, 8)) ( \
					dFragmentShader, dIndexBuffer, dVaryingBuffer, dColorBuffer, dDepthBuffer, dGlobalBlendCoefs, \
						dGlobalBlendCoefsAlpha);
#define PIXEL_BLEND_FUNC_COND(cond,gsState) if (Impl::csDepthFunc==(cond)){PIXEL_BLEND_FUNC(cond,gsState)}
#define PIXEL_BLEND_FUNC_COND_COLLECTION(gsState) \
	PIXEL_BLEND_FUNC_COND(IF_COMPARE_OP_ALWAYS,gsState) \
	PIXEL_BLEND_FUNC_COND(IF_COMPARE_OP_LESS,gsState) \
	PIXEL_BLEND_FUNC_COND(IF_COMPARE_OP_EQUAL,gsState) \
	PIXEL_BLEND_FUNC_COND(IF_COMPARE_OP_GREATER,gsState) \
	PIXEL_BLEND_FUNC_COND(IF_COMPARE_OP_LESS_OR_EQUAL,gsState) \
	PIXEL_BLEND_FUNC_COND(IF_COMPARE_OP_GREATER_OR_EQUAL,gsState) \
	PIXEL_BLEND_FUNC_COND(IF_COMPARE_OP_NOT_EQUAL,gsState) \
	PIXEL_BLEND_FUNC_COND(IF_COMPARE_OP_NEVER,gsState) 
					
					if (dGeometryShader == nullptr) {
						PIXEL_BLEND_FUNC_COND_COLLECTION(false);
					}
					else {
						PIXEL_BLEND_FUNC_COND_COLLECTION(true);
					}

#undef PIXEL_BLEND_FUNC_COND_COLLECTION
#undef PIXEL_BLEND_FUNC_COND
#undef PIXEL_BLEND_FUNC
#undef PIXEL_BLEND_FUNC_SIGN
				}
				else {
#define PIXEL_TAG_FUNC_SIGN(cond) Impl::TriangleFragmentStage::pixelTaggingExecKernel<cond>
#define PIXEL_TAG_FUNC(cond) PIXEL_TAG_FUNC_SIGN(cond) CU_KARG2(dim3(numTileX, numTileY, 1), dim3(CU_EXPERIMENTAL_SUBTILE_WIDTH, CU_TILE_WIDTH, dispZ)) ( \
					dFragmentShader, dIndexBuffer, dVaryingBuffer, dColorBuffer, dDepthBuffer);
#define PIXEL_TAG_FUNC_COND(cond) if (Impl::csDepthFunc==(cond)){PIXEL_TAG_FUNC(cond)}
#define PIXEL_TAG_FUNC_COND_COLLECTION \
	PIXEL_TAG_FUNC_COND(IF_COMPARE_OP_ALWAYS) \
	PIXEL_TAG_FUNC_COND(IF_COMPARE_OP_LESS) \
	PIXEL_TAG_FUNC_COND(IF_COMPARE_OP_EQUAL) \
	PIXEL_TAG_FUNC_COND(IF_COMPARE_OP_GREATER) \
	PIXEL_TAG_FUNC_COND(IF_COMPARE_OP_LESS_OR_EQUAL) \
	PIXEL_TAG_FUNC_COND(IF_COMPARE_OP_GREATER_OR_EQUAL) \
	PIXEL_TAG_FUNC_COND(IF_COMPARE_OP_NOT_EQUAL) \
	PIXEL_TAG_FUNC_COND(IF_COMPARE_OP_NEVER) 
					if (dGeometryShader == nullptr) {
						PIXEL_TAG_FUNC_COND_COLLECTION;
						Impl::TriangleFragmentStage::pixelShadingExecKernel<0> CU_KARG2(dim3(1, quadX, quadY), dim3(4, 8, 8)) (
							dFragmentShader, dIndexBuffer, dVaryingBuffer, dColorBuffer, dDepthBuffer);
					}
					else {
						PIXEL_TAG_FUNC_COND_COLLECTION;
						Impl::TriangleFragmentStage::pixelShadingExecKernel<1> CU_KARG2(dim3(1, quadX, quadY), dim3(4, 8, 8)) (
							dFragmentShader, dIndexBuffer, dVaryingBuffer, dColorBuffer, dDepthBuffer);
					}
#undef PIXEL_TAG_FUNC_COND_COLLECTION
#undef PIXEL_TAG_FUNC_COND
#undef PIXEL_TAG_FUNC
#undef PIXEL_TAG_FUNC_SIGN

				}
				Impl::TriangleMiscStage::resetLargeTileKernel CU_KARG2(CU_MAX_TILE_X * CU_MAX_SUBTILES_PER_TILE, CU_MAX_TILE_X)(isLast);
			}
		}

		IFRIT_KERNEL void unifiedFilledTriangleRasterEngineKernel(
			int totalIndices,
			ifloat4* dPositionBuffer,
			int* dIndexBuffer,
			const float4* dVaryingBuffer,
			ifloat4** dColorBuffer,
			float* dDepthBuffer,
			GeometryShader* dGeometryShader,
			FragmentShader* dFragmentShader
		) {
			if (totalIndices == -1) {
				totalIndices = dMeshShaderAggOffsetIndexGlobal;
				dPositionBuffer = reinterpret_cast<ifloat4*>(dMeshShaderPositionAfter);
				dIndexBuffer = dMeshShaderIndicesAfter;
				dVaryingBuffer = reinterpret_cast<float4*>(dMeshShaderVaryingsAfter);
			}
			for (int i = 0; i < totalIndices; i += CU_SINGLE_TIME_TRIANGLE * CU_TRIANGLE_STRIDE) {
				auto indexCount = min((int)(CU_SINGLE_TIME_TRIANGLE * CU_TRIANGLE_STRIDE), totalIndices - i);
				bool isTailCall = (i + CU_SINGLE_TIME_TRIANGLE * CU_TRIANGLE_STRIDE) >= totalIndices;
				if (dGeometryShader == nullptr) {
					int geometryExecutionBlocks = (indexCount / CU_TRIANGLE_STRIDE / CU_GEOMETRY_PROCESSING_THREADS) + ((indexCount / CU_TRIANGLE_STRIDE % CU_GEOMETRY_PROCESSING_THREADS) != 0);
#define invokeGeometryClippingKernel(tpCullMode) if(csCullMode == tpCullMode) Impl::TriangleGeometryStage::geometryClippingKernel<0,tpCullMode> CU_KARG2(geometryExecutionBlocks, CU_GEOMETRY_PROCESSING_THREADS)(dPositionBuffer, dIndexBuffer, i, indexCount);
					invokeGeometryClippingKernel(IF_CULL_MODE_BACK);
					invokeGeometryClippingKernel(IF_CULL_MODE_FRONT);
					invokeGeometryClippingKernel(IF_CULL_MODE_FRONT_AND_BACK);
					invokeGeometryClippingKernel(IF_CULL_MODE_NONE);
#undef invokeGeometryClippingKernel
				}
				else {
					GeneralGeometryStage::devCallGeometryShader(dGeometryShader, CU_TRIANGLE_STRIDE, i, indexCount,
						(float4*)dPositionBuffer, dIndexBuffer, (float4*)dVaryingBuffer);
					Impl::TriangleGeometryStage::geometryClippingKernelEntryWithGS CU_KARG2(1, 1)();
				}
				unifiedRasterEngineStageIIFilledTriangleKernel CU_KARG2(1, 1)(
					dIndexBuffer, dVaryingBuffer, dColorBuffer, dDepthBuffer,
					dGeometryShader, dFragmentShader, isTailCall);
			}
		}

		IFRIT_HOST void unifiedFilledTriangleRasterEngineProfileEntry(
			int totalIndices,
			ifloat4* dPositionBuffer,
			int* dIndexBuffer,
			const float4* dVaryingBuffer,
			ifloat4** dColorBuffer,
			float* dDepthBuffer,
			GeometryShader* dGeometryShader,
			FragmentShader* dFragmentShader,
			cudaStream_t& compStream
		) {
			for (int i = 0; i < totalIndices; i += CU_SINGLE_TIME_TRIANGLE * CU_TRIANGLE_STRIDE) {
				auto indexCount = min((int)(CU_SINGLE_TIME_TRIANGLE * CU_TRIANGLE_STRIDE), totalIndices - i);
				bool isTailCall = (i + CU_SINGLE_TIME_TRIANGLE * CU_TRIANGLE_STRIDE) >= totalIndices;
				if (dGeometryShader == nullptr) {
					int geometryExecutionBlocks = IFRIT_InvoGetThreadBlocks(indexCount / CU_TRIANGLE_STRIDE, CU_GEOMETRY_PROCESSING_THREADS);
#define invokeGeometryClippingKernel(tpCullMode) if(csCullMode == tpCullMode) Impl::TriangleGeometryStage::geometryClippingKernel<0,tpCullMode> CU_KARG2(geometryExecutionBlocks, CU_GEOMETRY_PROCESSING_THREADS)(dPositionBuffer, dIndexBuffer, i, indexCount);
					invokeGeometryClippingKernel(IF_CULL_MODE_BACK);
					invokeGeometryClippingKernel(IF_CULL_MODE_FRONT);
					invokeGeometryClippingKernel(IF_CULL_MODE_FRONT_AND_BACK);
					invokeGeometryClippingKernel(IF_CULL_MODE_NONE);
#undef invokeGeometryClippingKernel
				}
				else {
					int geometryShaderBlocks = IFRIT_InvoGetThreadBlocks(indexCount / CU_TRIANGLE_STRIDE, CU_GEOMETRY_SHADER_THREADS);
					Impl::GeneralGeometryStage::geometryShadingKernel CU_KARG2(geometryShaderBlocks, CU_GEOMETRY_SHADER_THREADS) (i, indexCount, dGeometryShader, (float4*)dPositionBuffer,
						dIndexBuffer, dVaryingBuffer);
					Impl::TriangleGeometryStage::geometryClippingKernelEntryWithGS CU_KARG2(1, 1)();
				}

				uint32_t dAssembledTriangleCounterM2Host = 0;
				cudaDeviceSynchronize();
				cudaMemcpyFromSymbol(&dAssembledTriangleCounterM2Host, dAssembledTriangleCounterM2, sizeof(uint32_t));
				if constexpr (CU_OPT_II_SKIP_ON_FEW_GEOMETRIES) {
					if (!isTailCall && dAssembledTriangleCounterM2Host < CU_EXPERIMENTAL_II_FEW_GEOMETRIES_LIMIT) {
						continue;
					}
				}
				int totalTms = dAssembledTriangleCounterM2Host / CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER;
				int curTime = -1;
				auto dispatchBlocks = IFRIT_InvoGetThreadBlocks(dAssembledTriangleCounterM2Host, CU_EXPERIMENTAL_GEOMETRY_POSTPROC_THREADS);
				Impl::TriangleGeometryStage::geometryParamPostprocKernel CU_KARG2(dispatchBlocks, CU_EXPERIMENTAL_GEOMETRY_POSTPROC_THREADS) (dAssembledTriangleCounterM2Host);
				for (int sI = 0; sI < dAssembledTriangleCounterM2Host; sI += CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER) {
					curTime++;
					int length = min(dAssembledTriangleCounterM2Host - sI, CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER);
					int start = sI;
					bool isLast = (curTime == totalTms);
					if (curTime == totalTms - 1) {
						if (dAssembledTriangleCounterM2 % CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER < CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER * 2 / 5) {
							length = dAssembledTriangleCounterM2Host - sI;
							sI += CU_SINGLE_TIME_TRIANGLE_FIRST_BINNER;
							isLast = true;
						}
					}
					Impl::TriangleRasterizationStage::firstBinnerRasterizerEntryProfileKernel(start, length);
					auto totalCoverPrims = 0;
					cudaMemcpyFromSymbol(&totalCoverPrims, dCoverPrimsCounter, sizeof(int));
					int dispatchGatherBlocks = totalCoverPrims / CU_FIRST_RASTERIZATION_GATHER_THREADS + (totalCoverPrims % CU_FIRST_RASTERIZATION_GATHER_THREADS != 0);
					if (dispatchGatherBlocks != 0) {
						Impl::TriangleRasterizationStage::firstBinnerGatherKernel CU_KARG2(dispatchGatherBlocks, CU_FIRST_RASTERIZATION_GATHER_THREADS)(totalCoverPrims);
					}
					Impl::TriangleRasterizationStage::secondBinnerWorklistRasterizerEntryProfileKernel();

					int dispatchSfbaBlocks = IFRIT_InvoGetThreadBlocks(CU_MAX_TILE_X * CU_MAX_TILE_X * CU_MAX_SUBTILES_PER_TILE, 128);
					Impl::TriangleRasterizationStage::secondFinerBinnerAllocationKernel CU_KARG2(dispatchSfbaBlocks, 128) ();
					cudaDeviceSynchronize();
					int secondBinnerCands = 0;
					cudaMemcpyFromSymbol(&secondBinnerCands, dSecondBinnerCandCounter, sizeof(int));

					int dispatchSfbrBlocks = IFRIT_InvoGetThreadBlocks(secondBinnerCands, 128);
					if (secondBinnerCands != 0) {
						Impl::TriangleRasterizationStage::secondFinerBinnerRasterizationKernel CU_KARG2(dispatchSfbrBlocks, 128)(secondBinnerCands);
					}

					int numTileX = (hsFrameWidth / CU_TILE_WIDTH) + (hsFrameWidth % CU_TILE_WIDTH != 0);
					int numTileY = (hsFrameHeight / CU_TILE_WIDTH) + (hsFrameHeight % CU_TILE_WIDTH != 0);
					int dispZ = (CU_TILE_WIDTH / CU_EXPERIMENTAL_SUBTILE_WIDTH) + (CU_TILE_WIDTH % CU_EXPERIMENTAL_SUBTILE_WIDTH != 0);
					std::abort();
					if (dGeometryShader == nullptr) {
						//Impl::TriangleFragmentStage::pixelTaggingKernel<0> CU_KARG4(dim3(numTileX, numTileY, 1), dim3(CU_EXPERIMENTAL_SUBTILE_WIDTH, CU_TILE_WIDTH, dispZ), 0, compStream) (
						//	dFragmentShader, dIndexBuffer, dVaryingBuffer, dColorBuffer, dDepthBuffer
						//	);
					}
					else {
						//Impl::TriangleFragmentStage::pixelTaggingKernel<1> CU_KARG4(dim3(numTileX, numTileY, 1), dim3(CU_EXPERIMENTAL_SUBTILE_WIDTH, CU_TILE_WIDTH, dispZ), 0, compStream) (
						//	dFragmentShader, dIndexBuffer, dVaryingBuffer, dColorBuffer, dDepthBuffer
						//	);
					}
					Impl::TriangleMiscStage::resetLargeTileKernel CU_KARG4(CU_MAX_TILE_X * CU_MAX_SUBTILES_PER_TILE, CU_MAX_TILE_X, 0, compStream)(isLast);
				}
				if constexpr (CU_PROFILER_SMALL_TRIANGLE_OVERHEAD) {
					auto dSmallTriangleCountHost = 0;
					cudaMemcpyFromSymbol(&dSmallTriangleCountHost, dSmallTriangleCount, sizeof(uint32_t));
					printf("Small Triangle Rate:%f (%d/%d)\n", 1.0f * dSmallTriangleCountHost / dAssembledTriangleCounterM2Host, dSmallTriangleCountHost, dAssembledTriangleCounterM2Host);
					dSmallTriangleCount = 0;
				}
			}
			cudaDeviceSynchronize();
		}
	}

	namespace PointRasterizationStage {
		template <int geometryShaderEnabled>
		IFRIT_KERNEL void pointRasterizationInsertKernel(
			ifloat4* IFRIT_RESTRICT_CUDA dPosBuffer,
			int* IFRIT_RESTRICT_CUDA dIndexBuffer,
			uint32_t startingIndexId,
			uint32_t indexCount
		) {
			//TODO: Overdraw for points
			int globalInvo = threadIdx.x + blockDim.x * blockIdx.x;
			if (globalInvo >= indexCount)return;
			int index;
			float4 v;
			if constexpr (geometryShaderEnabled) {
				index = globalInvo;
				v = dGeometryShaderOutPos[index];
			}
			else {
				index = dIndexBuffer[globalInvo + startingIndexId];
				v = ((float4*)dPosBuffer)[index];
			}
			
			if (v.w < CU_EPS)return;
			v.x /= v.w;
			v.y /= v.w;
			v.z /= v.w;
			v.x = v.x*0.5+0.5;
			v.y = v.y*0.5+0.5;
			if (v.x < 0.0 || v.x>1.0 || v.y < 0.0 || v.y>1.0)return;
			int finalPx = int(v.x * (csFrameWidth - 1));
			int finalPy = int(v.y * (csFrameHeight - 1));
			int pixelId = finalPy * CU_MAX_FRAMEBUFFER_WIDTH + finalPx;
			atomicAdd(&dSecondBinnerFinerBufferSizePoint[pixelId], 1);
			int dv = atomicAdd(&dAssembledTriangleCounterM2, 1);
			dAtriBaryCenter12[dv] = v;
			dAtriPixelBelong[dv] = pixelId;
			dAtriOriginalPrimId[dv] = index;
		}
		IFRIT_KERNEL void pointRasterizationAllocKernel() {
			int globalInvo = threadIdx.x + blockDim.x * blockIdx.x;
			if (globalInvo >= csFrameHeight * csFrameWidth) return;
			int pixelX = globalInvo % csFrameWidth;
			int pixelY = globalInvo / csFrameWidth;
			int pixelId = pixelY * CU_MAX_FRAMEBUFFER_WIDTH + pixelX;
			dSecondBinnerFinerBufferStartPoint[pixelId] = atomicAdd(&dSecondBinnerFinerBufferGlobalSize, dSecondBinnerFinerBufferSizePoint[pixelId]);
			dSecondBinnerFinerBufferCurIndPoint[pixelId] = dSecondBinnerFinerBufferStartPoint[pixelId];
		}
		IFRIT_KERNEL void pointRasterizationPlaceKernel(int totalCount) {
			int globalInvo = threadIdx.x + blockDim.x * blockIdx.x;
			if (globalInvo > totalCount)return;
			int pixelId = dAtriPixelBelong[globalInvo];
			int dv = atomicAdd(&dSecondBinnerFinerBufferCurIndPoint[pixelId], 1);
			dSecondBinnerFinerBufferPoint[dv] = globalInvo;
		}
		IFRIT_KERNEL void pointRasterizationPlaceEntryKernel() {
			int totalPoints = dSecondBinnerFinerBufferGlobalSize;
			int dispatchBlocks = IFRIT_InvoGetThreadBlocks(totalPoints, CU_POINT_RASTERIZATION_PLACE_THREADS);
			if (totalPoints == 0)return;
			pointRasterizationPlaceKernel CU_KARG2(dispatchBlocks, CU_POINT_RASTERIZATION_PLACE_THREADS)(totalPoints);
		}
		IFRIT_KERNEL void pointGeometryKernelEntryWithGS() {
			int numTriangles = dGeometryShaderOutSize;
			int dispatchBlocks = IFRIT_InvoGetThreadBlocks(numTriangles, CU_GEOMETRY_PROCESSING_THREADS);
			pointRasterizationInsertKernel<1> CU_KARG2(dispatchBlocks, CU_GEOMETRY_PROCESSING_THREADS)(nullptr, nullptr, 0, numTriangles * 3);
		}
	}
	namespace PointFragmentStage {
		template <int geometryShaderEnabled>
		IFRIT_KERNEL void pixelShadingKernel(
			FragmentShader* fragmentShader,
			int* IFRIT_RESTRICT_CUDA dIndexBuffer,
			const float4* IFRIT_RESTRICT_CUDA dVaryingBuffer,
			ifloat4** IFRIT_RESTRICT_CUDA dColorBuffer,
			float* IFRIT_RESTRICT_CUDA dDepthBuffer
		) {
			//TODO: must be 256 threads
			auto globalInvo = threadIdx.x + blockIdx.x * blockDim.x;
			auto pixelX = globalInvo % csFrameWidth;
			auto pixelY = globalInvo / csFrameHeight;
			auto curDepth = dDepthBuffer[globalInvo];
			auto dPixelId = pixelY * CU_MAX_FRAMEBUFFER_WIDTH + pixelX;
			int actPrim = -1;
			int localDepth = 1e9;
			IFRIT_SHARED float colorOutputSingle[256 * 4];
			IFRIT_SHARED float interpolatedVaryings[256 * 4 * CU_MAX_VARYINGS];

			auto shadingPass = [&](int primId) {
				auto varyingCount = csVaryingCounts;
				int orgPrimId = dAtriOriginalPrimId[primId];
				if constexpr (geometryShaderEnabled) {
					orgPrimId = dAtriOriginalPrimIdGs[primId];
				}
				auto addr = dIndexBuffer + orgPrimId;
				for (int k = 0; k < varyingCount; k++) {
					float4 vd;
					vd = { 0,0,0,0 };
					float4 vaf4;
					if constexpr (geometryShaderEnabled) {
						vaf4 = dGeometryShaderOutVaryings[orgPrimId * varyingCount + k];
					}
					else {
						vaf4 = dVaryingBuffer[orgPrimId * varyingCount + k];
					}
					vd.x = vaf4.x;
					vd.y = vaf4.y;
					vd.z = vaf4.z;
					vd.w = vaf4.w;
					auto dest = (ifloat4s256*)(interpolatedVaryings + 1024 * k + threadIdx.x);
					dest->x = vd.x;
					dest->y = vd.y;
					dest->z = vd.z;
					dest->w = vd.w;
				}
				fragmentShader->execute(interpolatedVaryings + threadIdx.x, colorOutputSingle + threadIdx.x);
				auto col0 = static_cast<ifloat4*>(__builtin_assume_aligned(dColorBuffer[0], 16));
				ifloat4 finalRgba;
				ifloat4s256 midOutput = ((ifloat4s256*)(colorOutputSingle + threadIdx.x))[0];
				finalRgba.x = midOutput.x;
				finalRgba.y = midOutput.y;
				finalRgba.z = midOutput.z;
				finalRgba.w = midOutput.w;
				col0[globalInvo] = finalRgba;

				};
			auto zPrePass = [&](int primId) {
				auto z = dAtriBaryCenter12[primId].z;
				if (z < localDepth) {
					actPrim = primId;
					localDepth = z;
				}
			};
			int dStart = dSecondBinnerFinerBufferStartPoint[dPixelId];
			int dLast = dSecondBinnerFinerBufferCurIndPoint[dPixelId];

			for (int i = dStart; i < dLast; i++) {
				zPrePass(dSecondBinnerFinerBufferPoint[i]);
			}
			if (localDepth < curDepth && actPrim != -1) {
				shadingPass(actPrim);
				dDepthBuffer[globalInvo] = localDepth;
			}
		}
	}
	namespace PointMiscStage {
		IFRIT_KERNEL void resetPointRasterizerKernel() {
			int globalInvo = threadIdx.x + blockDim.x * blockIdx.x;
			dSecondBinnerFinerBufferSizePoint[globalInvo] = 0;
			if (globalInvo == 0) {
				dSecondBinnerFinerBufferGlobalSize = 0;
			}
		}
	}
	namespace PointPipeline {
		IFRIT_KERNEL void unifiedPointRasterEngineKernel(
			int totalIndices,
			ifloat4* dPositionBuffer,
			int* dIndexBuffer,
			const float4* dVaryingBuffer,
			ifloat4** dColorBuffer,
			float* dDepthBuffer,
			GeometryShader* dGeometryShader,
			FragmentShader* dFragmentShader
		) {
			for (int i = 0; i < totalIndices; i += CU_SINGLE_TIME_TRIANGLE) {
				auto indexCount = min((int)(CU_SINGLE_TIME_TRIANGLE), totalIndices - i);
				bool isTailCall = (i + CU_SINGLE_TIME_TRIANGLE) >= totalIndices;
				int geometryExecutionBlocks = IFRIT_InvoGetThreadBlocks(indexCount, CU_POINT_RASTERIZATION_FIRST_THREADS);
				if (geometryExecutionBlocks == 0)continue;
				if (dGeometryShader != nullptr) {
					GeneralGeometryStage::devCallGeometryShader(dGeometryShader, CU_TRIANGLE_STRIDE, i, indexCount,
						(float4*)dPositionBuffer, dIndexBuffer, (float4*)dVaryingBuffer);
					PointRasterizationStage::pointGeometryKernelEntryWithGS CU_KARG2(1, 1)();
				}
				else {
					PointRasterizationStage::pointRasterizationInsertKernel<0> CU_KARG2(geometryExecutionBlocks, CU_POINT_RASTERIZATION_FIRST_THREADS)(
						dPositionBuffer, dIndexBuffer, i, indexCount);
				}

				int allocBlocks = IFRIT_InvoGetThreadBlocks(csFrameHeight * csFrameWidth, 128);
				PointRasterizationStage::pointRasterizationAllocKernel CU_KARG2(allocBlocks, 128)();
				PointRasterizationStage::pointRasterizationPlaceEntryKernel CU_KARG2(1, 1) ();
				int psBlocks = IFRIT_InvoGetThreadBlocks(csFrameHeight * csFrameWidth, 256);
				if (dGeometryShader != nullptr) {
					PointFragmentStage::pixelShadingKernel<1> CU_KARG2(psBlocks, 256)(dFragmentShader, dIndexBuffer, dVaryingBuffer, dColorBuffer, dDepthBuffer);
				}
				else {
					PointFragmentStage::pixelShadingKernel<0> CU_KARG2(psBlocks, 256)(dFragmentShader, dIndexBuffer, dVaryingBuffer, dColorBuffer, dDepthBuffer);
				}

				Impl::TriangleMiscStage::resetLargeTileKernel CU_KARG2(CU_MAX_TILE_X * CU_MAX_SUBTILES_PER_TILE, CU_MAX_TILE_X)(true);

				int rsBlocks = IFRIT_InvoGetThreadBlocks(CU_MAX_FRAMEBUFFER_SIZE, 256);
				Impl::PointMiscStage::resetPointRasterizerKernel CU_KARG2(rsBlocks, 256)();
			}
		}
	}

	namespace LineGeometryStage {
		IFRIT_DEVICE void devLineCullClip(float4 v1, float4 v2, float4 bary1,float4 bary2, int primId) {
			if (v1.w < 0 && v2.w < 0)return;
			if (v1.w < 0) {
				float deno = v2.w - v1.w;
				float numo = -1.0 + v2.w;
				float t = numo / deno;
				using Ifrit::Engine::Math::ShaderOps::CUDA::lerp;
				float4 intersection = lerp(v2, v1, t);
				float4 barycenter = lerp(bary2, bary1, t);
				v1 = intersection;
				bary1 = barycenter;
			}
			if (v2.w < 0) {
				float deno = v1.w - v2.w;
				float numo = -1.0 + v1.w;
				float t = numo / deno;
				using Ifrit::Engine::Math::ShaderOps::CUDA::lerp;
				float4 intersection = lerp(v1, v2, t);
				float4 barycenter = lerp(bary1, bary2, t);
				v2 = intersection;
				bary2 = barycenter;
			}
			v1.x /= v1.w;
			v1.y /= v1.w;
			v1.z /= v1.w;
			v2.x /= v2.w;
			v2.y /= v2.w;
			v2.z /= v2.w;
			v1.x = v1.x * 0.5 + 0.5;
			v1.y = v1.y * 0.5 + 0.5;
			v2.x = v2.x * 0.5 + 0.5;
			v2.y = v2.y * 0.5 + 0.5;
			
			//WARNING & TODO: Simple Discard & Frustum Clip
			if (v1.x < 0.0 || v1.y < 0.0 || v1.x>1.0 || v1.y>1.0)return;
			if (v2.x < 0.0 || v2.y < 0.0 || v1.x>1.0 || v1.y>1.0)return;
			int ds = atomicAdd(&dAssembledTriangleCounterM2, 1);
			dAtriInterpolBase1[ds] = v1;
			dAtriInterpolBase2[ds] = v2;
			dAtriEdgeCoefs1[ds] = bary1;
			dAtriEdgeCoefs2[ds] = bary2;
			dAtriOriginalPrimId[ds] = primId;
			//printf("%f %f -> %f %f \n", v1.x, v1.y, v2.x, v2.y);
		}
		template <int geometryShaderEnabled>
		IFRIT_KERNEL void lineGeometryAssemblyKernel(
			ifloat4* IFRIT_RESTRICT_CUDA dPosBuffer,
			int* IFRIT_RESTRICT_CUDA dIndexBuffer,
			uint32_t startingIndexId,
			uint32_t indexCount
		) {
			float4 v1, v2, v3;
			int primId;
			const auto globalInvoIdx = blockIdx.x * blockDim.x + threadIdx.x;
			if (globalInvoIdx >= indexCount / CU_TRIANGLE_STRIDE) return;
			const auto indexStart = globalInvoIdx * CU_TRIANGLE_STRIDE + startingIndexId;

			if constexpr (geometryShaderEnabled) {
				if (globalInvoIdx >= indexCount / 3) return;
				auto dGSPosBufferAligned = static_cast<float4*>(__builtin_assume_aligned(dGeometryShaderOutPos, 16));
				v1 = dGSPosBufferAligned[globalInvoIdx * 3];
				v2 = dGSPosBufferAligned[globalInvoIdx * 3 + 1];
				v3 = dGSPosBufferAligned[globalInvoIdx * 3 + 2];
				primId = globalInvoIdx;
			}
			else {
				auto dPosBufferAligned = static_cast<float4*>(__builtin_assume_aligned(dPosBuffer, 16));
				v1 = dPosBufferAligned[dIndexBuffer[indexStart]];
				v2 = dPosBufferAligned[dIndexBuffer[indexStart + 1]];
				v3 = dPosBufferAligned[dIndexBuffer[indexStart + 2]];
			}
			
			primId = globalInvoIdx + startingIndexId / CU_TRIANGLE_STRIDE;
			devLineCullClip(v1, v2, { 1,0,0,0 }, { 0,1,0,0 }, primId * 3 + 0);
			devLineCullClip(v2, v3, { 0,1,0,0 }, { 0,0,1,0 }, primId * 3 + 1);
			devLineCullClip(v3, v1, { 0,0,1,0 }, { 1,0,0,0 }, primId * 3 + 2);
		}
		IFRIT_KERNEL void lineGeometryKernelEntryWithGS() {
			int numTriangles = dGeometryShaderOutSize;
			int dispatchBlocks = IFRIT_InvoGetThreadBlocks(numTriangles/ CU_TRIANGLE_STRIDE, CU_GEOMETRY_PROCESSING_THREADS);
			lineGeometryAssemblyKernel<1> CU_KARG2(dispatchBlocks, CU_GEOMETRY_PROCESSING_THREADS)(nullptr, nullptr, 0, numTriangles);
		}
	}
	namespace LineRasterizationStage {
		IFRIT_KERNEL void bresenhamRasterizationKernel(int totalCount) {
			//TODO: Excessive Global Atomics
			//TODO: Excessive Branch Divergence
			int globalInvo = threadIdx.x + blockDim.x * blockIdx.x;
			if (globalInvo >= totalCount)return;
			float4 v1 = dAtriInterpolBase1[globalInvo];
			float4 v2 = dAtriInterpolBase2[globalInvo];
			int v1x = int(v1.x * (csFrameWidth - 1));
			int v1y = int(v1.y * (csFrameHeight - 1));
			int v2x = int(v2.x * (csFrameWidth - 1));
			int v2y = int(v2.y * (csFrameHeight - 1));
			int deltaX = abs(v1x - v2x);
			int deltaY = abs(v1y - v2y);

			if (deltaX > deltaY) {
				if (v2x < v1x) {
					v2x ^= v1x;
					v1x ^= v2x;
					v2x ^= v1x;
					v2y ^= v1y;
					v1y ^= v2y;
					v2y ^= v1y;
				}
				int cx = v1x, cy = v1y;
				float dslope = abs(1.0f * (v2y - v1y) / (v2x - v1x));
				int ystep = ((v2y - v1y) > 0) ? 1 : -1;
				float d = 0.0f;
				int storeStart = atomicAdd(&dLineRasterOutSize, (v2x - v1x + 1));
				for (int i = 0; i <= v2x-v1x; i++) {
					int pixelId = cy * CU_MAX_FRAMEBUFFER_WIDTH + cx;
					dLineRasterOut[storeStart + i].x = globalInvo;
					dLineRasterOut[storeStart + i].y = pixelId;
					atomicAdd(&dSecondBinnerFinerBufferSizePoint[pixelId], 1);
					d += dslope;
					if (d > 0.5) {
						cy += ystep;
						d -= 1.0;
					}
					cx++;
				}
			}
			else {
				if (v2y < v1y) {
					v2x ^= v1x;
					v1x ^= v2x;
					v2x ^= v1x;
					v2y ^= v1y;
					v1y ^= v2y;
					v2y ^= v1y;
				}
				int cx = v1x, cy = v1y;
				float dslope = abs( 1.0f * (v2x - v1x) / (v2y - v1y));
				int ystep = ((v2x - v1x) > 0) ? 1 : -1;
				float d = 0.0f;
				int storeStart = atomicAdd(&dLineRasterOutSize, (v2y - v1y + 1));
				for (int i = 0; i <= v2y - v1y; i++) {
					int pixelId = cy * CU_MAX_FRAMEBUFFER_WIDTH + cx;
					dLineRasterOut[storeStart + i].x = globalInvo;
					dLineRasterOut[storeStart + i].y = pixelId;
					atomicAdd(&dSecondBinnerFinerBufferSizePoint[pixelId], 1);
					d += dslope;
					if (d > 0.5) {
						cx += ystep;
						d -= 1.0;
					}
					cy++;
				}
			}
		}
		IFRIT_KERNEL void lineRasterAllocKernel() {
			int globalInvo = threadIdx.x + blockIdx.x * blockDim.x;
			if (globalInvo >= csFrameHeight * csFrameWidth)return;
			int pixelX = globalInvo % csFrameWidth;
			int pixelY = globalInvo / csFrameWidth;
			int pixelId = pixelY * CU_MAX_FRAMEBUFFER_WIDTH + pixelX;
			dSecondBinnerFinerBufferStartPoint[pixelId] = atomicAdd(&dSecondBinnerFinerBufferGlobalSize,
				dSecondBinnerFinerBufferSizePoint[pixelId]);
			dSecondBinnerFinerBufferCurIndPoint[pixelId] = dSecondBinnerFinerBufferStartPoint[pixelId];
		}

		IFRIT_KERNEL void linerRasterPlaceKernel(int totalSize) {
			int globalInvo = threadIdx.x + blockIdx.x * blockDim.x;
			if (globalInvo >= totalSize)return;
			int prId = dLineRasterOut[globalInvo].x;
			int pixelId = dLineRasterOut[globalInvo].y;
			int orgPrimId = dAtriOriginalPrimId[prId];
			int ds = atomicAdd(&dSecondBinnerFinerBufferCurIndPoint[pixelId], 1);
			dSecondBinnerFinerBufferPoint[ds] = prId;
		}

		IFRIT_KERNEL void bresenhamRasterizationEntryKernel() {
			auto total = dAssembledTriangleCounterM2;
			auto dispatchBlocks = IFRIT_InvoGetThreadBlocks(total, CU_LINE_RASTERIZATION_FIRST_THREADS);
			bresenhamRasterizationKernel CU_KARG2(dispatchBlocks, CU_LINE_RASTERIZATION_FIRST_THREADS)(total);
		}

		IFRIT_KERNEL void linerRasterPlaceEntryKernel() {
			auto total = dSecondBinnerFinerBufferGlobalSize;
			auto dispatchBlocks = IFRIT_InvoGetThreadBlocks(total, CU_LINE_RASTERIZATION_PLACE_THREADS);
			linerRasterPlaceKernel CU_KARG2(dispatchBlocks, CU_LINE_RASTERIZATION_PLACE_THREADS)(total);
		}
	}
	namespace LineFragmentStage {

		template <int geometryShaderEnabled>
		IFRIT_KERNEL void pixelShadingKernel(
			FragmentShader* fragmentShader,
			int* IFRIT_RESTRICT_CUDA dIndexBuffer,
			const float4* IFRIT_RESTRICT_CUDA dVaryingBuffer,
			ifloat4** IFRIT_RESTRICT_CUDA dColorBuffer,
			float* IFRIT_RESTRICT_CUDA dDepthBuffer
		) {
			//TODO: Faster Perspective Interpolation
			auto globalInvo = threadIdx.x + blockIdx.x * blockDim.x;
			auto pixelX = globalInvo % csFrameWidth;
			auto pixelY = globalInvo / csFrameHeight;
			auto curDepth = dDepthBuffer[globalInvo];
			auto dPixelId = pixelY * CU_MAX_FRAMEBUFFER_WIDTH + pixelX;
			int actPrim = -1;
			int localDepth = 1e9;
			IFRIT_SHARED float colorOutputSingle[256 * 4];
			IFRIT_SHARED float interpolatedVaryings[256 * 4 * CU_MAX_VARYINGS];
			auto getPercent = [&](float4 v1, float4 v2) {
				v1.x *= csFrameWidth;
				v1.y *= csFrameHeight;
				v2.x *= csFrameWidth;
				v2.y *= csFrameHeight;
				float deltaX = abs(v2.x - v1.x);
				float deltaY = abs(v2.y - v1.y);
				//TODO: Branch Divergence
				if (deltaX >= deltaY) {
					float percent = (pixelX - v1.x) / (v2.x - v1.x);
					return percent;
				}
				else {
					float percent = (pixelY - v1.y) / (v2.y - v1.y);
					return percent;
				}
			};
			auto shadingPass = [&](int primId) {
				//TODO: Optimization for primitive
				float4 v1 = dAtriInterpolBase1[primId];
				float4 v2 = dAtriInterpolBase2[primId];
				auto percent = getPercent(v1, v2);
				float z1 = v1.w, z2 = v2.w;
				//zt = 1/(1/z1+s(1/z2-1/z1)
				//zt = z1z2/(z2+s(z1-z2))
				float zt = z1 * z2 / (z2 + percent * (z1 - z2));
				int orgPrimId = dAtriOriginalPrimId[primId];
				int intPrimId = orgPrimId % 3;
				orgPrimId /= 3;

				int idx1, idx2;
				if constexpr (geometryShaderEnabled) {
					idx1 = orgPrimId * CU_TRIANGLE_STRIDE + intPrimId;
					idx2 = orgPrimId * CU_TRIANGLE_STRIDE + (intPrimId + 1) % 3;
				}
				else {
					idx1 = dIndexBuffer[orgPrimId * CU_TRIANGLE_STRIDE + intPrimId];
					idx2 = dIndexBuffer[orgPrimId * CU_TRIANGLE_STRIDE + (intPrimId + 1) % 3];
				}
				
				for (int i = 0; i < csVaryingCounts; i++) {
					float4 vary1, vary2;
					if constexpr (geometryShaderEnabled) {
						vary1 = dGeometryShaderOutVaryings[idx1 * csVaryingCounts + i];
						vary2 = dGeometryShaderOutVaryings[idx2 * csVaryingCounts + i];
					}
					else {
						vary1 = dVaryingBuffer[idx1 * csVaryingCounts + i];
						vary2 = dVaryingBuffer[idx2 * csVaryingCounts + i];
					}
					vary1.x /= z1;
					vary1.y /= z1;
					vary1.z /= z1;
					vary1.w /= z1;
					vary2.x /= z2;
					vary2.y /= z2;
					vary2.z /= z2;
					vary2.w /= z2;

					float4 vd;
					vd.x = (percent * (vary2.x - vary1.x) + vary1.x) * zt;
					vd.y = (percent * (vary2.y - vary1.y) + vary1.y) * zt;
					vd.z = (percent * (vary2.z - vary1.z) + vary1.z) * zt;
					vd.w = (percent * (vary2.w - vary1.w) + vary1.w) * zt;

					auto dest = (ifloat4s256*)(interpolatedVaryings + 1024 * i + threadIdx.x);
					dest->x = vd.x;
					dest->y = vd.y;
					dest->z = vd.z;
					dest->w = vd.w;
				}
				fragmentShader->execute(interpolatedVaryings + threadIdx.x, colorOutputSingle + threadIdx.x);
				auto col0 = static_cast<ifloat4*>(__builtin_assume_aligned(dColorBuffer[0], 16));
				ifloat4 finalRgba;
				ifloat4s256 midOutput = ((ifloat4s256*)(colorOutputSingle + threadIdx.x))[0];
				finalRgba.x = midOutput.x;
				finalRgba.y = midOutput.y;
				finalRgba.z = midOutput.z;
				finalRgba.w = midOutput.w;
				col0[globalInvo] = finalRgba;
			};
			auto zPrePass = [&](int primId) {
				float4 v1 = dAtriInterpolBase1[primId];
				float4 v2 = dAtriInterpolBase2[primId];
				auto percent = getPercent(v1, v2);
				using Ifrit::Engine::Math::ShaderOps::CUDA::lerp;
				auto zVal = lerp(v1.z, v2.z, percent);
				if (zVal < localDepth) {
					actPrim = primId;
					localDepth = zVal;
				}
			};
			int dStart = dSecondBinnerFinerBufferStartPoint[dPixelId];
			int dLast = dSecondBinnerFinerBufferCurIndPoint[dPixelId];
			for (int i = dStart; i < dLast; i++) {
				zPrePass(dSecondBinnerFinerBufferPoint[i]);
			}
			if (localDepth < curDepth && actPrim != -1) {
				shadingPass(actPrim);
				dDepthBuffer[globalInvo] = localDepth;
			}
		}
	}
	namespace LineMiscStage {
		IFRIT_KERNEL void resetLineRasterizerKernel() {
			int globalInvo = threadIdx.x + blockDim.x * blockIdx.x;
			dSecondBinnerFinerBufferSizePoint[globalInvo] = 0;
			if (globalInvo == 0) {
				dSecondBinnerFinerBufferGlobalSize = 0;
				dLineRasterOutSize = 0;
			}
		}
	}
	namespace LinePipeline {
		IFRIT_KERNEL void unifiedLineRasterEngineKernel(int totalIndices,
			ifloat4* dPositionBuffer,
			int* dIndexBuffer,
			const float4* dVaryingBuffer,
			ifloat4** dColorBuffer,
			float* dDepthBuffer,
			GeometryShader* dGeometryShader,
			FragmentShader* dFragmentShader
		) {
			for (int i = 0; i < totalIndices; i += CU_SINGLE_TIME_TRIANGLE * CU_TRIANGLE_STRIDE) {
				auto indexCount = min((int)(CU_SINGLE_TIME_TRIANGLE * CU_TRIANGLE_STRIDE), totalIndices - i);
				bool isTailCall = (i + CU_SINGLE_TIME_TRIANGLE * CU_TRIANGLE_STRIDE) >= totalIndices;
				int geometryExecutionBlocks = IFRIT_InvoGetThreadBlocks(indexCount / CU_TRIANGLE_STRIDE, CU_LINE_GEOMETRY_THREADS);
				if (geometryExecutionBlocks == 0)continue;
				if (dGeometryShader != nullptr) {
					GeneralGeometryStage::devCallGeometryShader(dGeometryShader, CU_TRIANGLE_STRIDE, i, indexCount,
						(float4*)dPositionBuffer, dIndexBuffer, (float4*)dVaryingBuffer);
					LineGeometryStage::lineGeometryKernelEntryWithGS CU_KARG2(1, 1)();
				}
				else {
					LineGeometryStage::lineGeometryAssemblyKernel<0> CU_KARG2(geometryExecutionBlocks, CU_LINE_GEOMETRY_THREADS)(
						dPositionBuffer, dIndexBuffer, i, indexCount);
				}
				LineRasterizationStage::bresenhamRasterizationEntryKernel CU_KARG2(1, 1) ();
				int allocBlocks = IFRIT_InvoGetThreadBlocks(csFrameHeight * csFrameWidth, 128);
				LineRasterizationStage::lineRasterAllocKernel CU_KARG2(allocBlocks, 128)();
				LineRasterizationStage::linerRasterPlaceEntryKernel CU_KARG2(1, 1) ();
				int psBlocks = IFRIT_InvoGetThreadBlocks(csFrameHeight * csFrameWidth, 256);
				if (dGeometryShader != nullptr) {
					LineFragmentStage::pixelShadingKernel<1> CU_KARG2(psBlocks, 256)(dFragmentShader, dIndexBuffer, dVaryingBuffer, dColorBuffer, dDepthBuffer);
				}
				else {
					LineFragmentStage::pixelShadingKernel<0> CU_KARG2(psBlocks, 256)(dFragmentShader, dIndexBuffer, dVaryingBuffer, dColorBuffer, dDepthBuffer);
				}
				
				TriangleMiscStage::resetLargeTileKernel CU_KARG2(CU_MAX_TILE_X * CU_MAX_SUBTILES_PER_TILE, CU_MAX_TILE_X)(true);
				int rsBlocks = IFRIT_InvoGetThreadBlocks(CU_MAX_FRAMEBUFFER_SIZE, 256);
				LineMiscStage::resetLineRasterizerKernel CU_KARG2(rsBlocks, 256)();
			}
		}
	}

	namespace ImageOperator {
		IFRIT_KERNEL void imageResetFloat32MonoKernel(
			float* dBuffer,
			float value
		) {
			const auto invoX = blockIdx.x * blockDim.x + threadIdx.x;
			const auto invoY = blockIdx.y * blockDim.y + threadIdx.y;
			if (invoX >= csFrameWidth || invoY >= csFrameHeight) {
				return;
			}
			dBuffer[(invoY * csFrameWidth + invoX)] = value;
		}

		IFRIT_KERNEL void imageResetFloat32RGBAKernel(
			float* dBuffer,
			float4 value
		) {
			const auto invoX = blockIdx.x * blockDim.x + threadIdx.x;
			const auto invoY = blockIdx.y * blockDim.y + threadIdx.y;
			if (invoX >= csFrameWidth || invoY >= csFrameHeight) {
				return;
			}
			const auto dAlignedBuffer = static_cast<float*>(__builtin_assume_aligned(dBuffer, 16));
			dBuffer[(invoY * csFrameWidth + invoX) * 4 + 0] = value.x;
			dBuffer[(invoY * csFrameWidth + invoX) * 4 + 1] = value.y;
			dBuffer[(invoY * csFrameWidth + invoX) * 4 + 2] = value.z;
			dBuffer[(invoY * csFrameWidth + invoX) * 4 + 3] = value.w;
		}
	}
	namespace MiscKernels {
		IFRIT_KERNEL void updateFragmentShaderKernel(FragmentShader* fragmentShader) {
			for (int i = 0; i < CU_MAX_TEXTURE_SLOTS; i++) {
				fragmentShader->atTexture[i] = csTextures[i];
				fragmentShader->atTextureHei[i] = csTextureHeight[i];
				fragmentShader->atTextureWid[i] = csTextureWidth[i];
			}
			for (int i = 0; i < CU_MAX_SAMPLER_SLOTS; i++) {
				fragmentShader->atSamplerPtr[i] = csSamplers[i];
			}
		}
		IFRIT_KERNEL void updateBlendStateKernel() {
			auto bs = csBlendState;
			///SrcColor (x1/x[src.alpha]/x[dst.alpha]/x0)
			if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE) {
				dGlobalBlendCoefs.s = { 1,0,0,0 };
			}
			else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ZERO) {
				dGlobalBlendCoefs.s = { 0,0,0,1 };
			}
			else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_DST_ALPHA) {
				dGlobalBlendCoefs.s = { 0,0,1,0 };
			}
			else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_SRC_ALPHA) {
				dGlobalBlendCoefs.s = { 0,1,0,0 };
			}
			else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_DST_ALPHA) {
				dGlobalBlendCoefs.s = { 1,0,-1,0 };
			}
			else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
				dGlobalBlendCoefs.s = { 1,-1,0,0 };
			}
			else {
				printf("Unsupported blend factor");
				Impl::GeneralFunction::devAbort();
			}
			//SrcAlpha
			if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE) {
				dGlobalBlendCoefsAlpha.s = { 1,0,0,0 };
			}
			else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ZERO) {
				dGlobalBlendCoefsAlpha.s = { 0,0,0,1 };
			}
			else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_DST_ALPHA) {
				dGlobalBlendCoefsAlpha.s = { 0,0,1,0 };
			}
			else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_SRC_ALPHA) {
				dGlobalBlendCoefsAlpha.s = { 0,1,0,0 };
			}
			else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_DST_ALPHA) {
				dGlobalBlendCoefsAlpha.s = { 1,0,-1,0 };
			}
			else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
				dGlobalBlendCoefsAlpha.s = { 1,-1,0,0 };
			}
			else {
				printf("Unsupported blend factor");
				Impl::GeneralFunction::devAbort();
			}


			//DstColor
			if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE) {
				dGlobalBlendCoefs.d = { 1,0,0,0 };
			}
			else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ZERO) {
				dGlobalBlendCoefs.d = { 0,0,0,1 };
			}
			else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_DST_ALPHA) {
				dGlobalBlendCoefs.d = { 0,0,1,0 };
			}
			else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_SRC_ALPHA) {
				dGlobalBlendCoefs.d = { 0,1,0,0 };
			}
			else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_DST_ALPHA) {
				dGlobalBlendCoefs.d = { 1,0,-1,0 };
			}
			else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
				dGlobalBlendCoefs.d = { 1,-1,0,0 };
			}

			//DstAlpha
			if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE) {
				dGlobalBlendCoefsAlpha.d = { 1,0,0,0 };
			}
			else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ZERO) {
				dGlobalBlendCoefsAlpha.d = { 0,0,0,1 };
			}
			else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_DST_ALPHA) {
				dGlobalBlendCoefsAlpha.d = { 0,0,1,0 };
			}
			else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_SRC_ALPHA) {
				dGlobalBlendCoefsAlpha.d = { 0,1,0,0 };
			}
			else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_DST_ALPHA) {
				dGlobalBlendCoefsAlpha.d = { 1,0,-1,0 };
			}
			else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
				dGlobalBlendCoefsAlpha.d = { 1,-1,0,0 };
			}
			else {
				printf("Unsupported blend factor");
				Impl::GeneralFunction::devAbort();
			}
		}
	}
	
	namespace InitializationKernels {
		IFRIT_KERNEL void globalInitializationKernel() {
			csDepthFunc = IF_COMPARE_OP_LESS;
			csCullMode = IF_CULL_MODE_BACK;
		}
	}
}

namespace  Ifrit::Engine::TileRaster::CUDA::Invocation {

	char* deviceMalloc(uint32_t size) {
		char* ptr;
		cudaMalloc(&ptr, size);
		return ptr;
	}

	void deviceFree(char* ptr) {
		cudaFree(ptr);
	}

	void createTexture(uint32_t texId, const IfritImageCreateInfo& createInfo, float* data) {
		if (data != nullptr) {
			printf("Data should not be specified\n");
			std::abort();
		}
		void* devicePtr;
		auto texWid = createInfo.extent.width;
		auto texHeight = createInfo.extent.height;
		auto texArrLayers = createInfo.arrayLayers;
		auto texLodSizes = 0;
		auto texLodWid = texWid, texLodHeight = texHeight;
		for (int i = 0; i < createInfo.mipLevels; i++) {
			texLodWid = (texLodWid + 1) >> 1;
			texLodHeight = (texLodHeight + 1) >> 1;
			texLodSizes += (texLodWid * texLodHeight * texArrLayers);
		}
		cudaMalloc(&devicePtr, (texWid * texHeight * texArrLayers + texLodSizes) * 4 * sizeof(float));
		//cudaMemcpy(devicePtr, data, texWid * texHeight * 4 * sizeof(float), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		Impl::hsTextures[texId] = (float*)devicePtr;
		Impl::hsTextureHeight[texId] = texHeight;
		Impl::hsTextureWidth[texId] = texWid;
		Impl::hsTextureMipLevels[texId] = createInfo.mipLevels;
		Impl::hsTextureArrayLayers[texId] = createInfo.arrayLayers;

		cudaMemcpyToSymbol(Impl::csTextures, &Impl::hsTextures[texId], sizeof(float*), sizeof(float*) * texId);
		cudaMemcpyToSymbol(Impl::csTextureHeight, &Impl::hsTextureHeight[texId], sizeof(int), sizeof(int) * texId);
		cudaMemcpyToSymbol(Impl::csTextureMipLevels, &Impl::hsTextureMipLevels[texId], sizeof(int), sizeof(int) * texId);
		cudaMemcpyToSymbol(Impl::csTextureArrayLayers, &Impl::hsTextureArrayLayers[texId], sizeof(int), sizeof(int) * texId);
		cudaMemcpyToSymbol(Impl::csTextureWidth, &Impl::hsTextureWidth[texId], sizeof(int), sizeof(int)*texId);
	}
	void createSampler(uint32_t slotId, const IfritSamplerT& samplerState) {
		Impl::hsSampler[slotId] = samplerState;
		cudaMemcpyToSymbol(Impl::csSamplers, &samplerState, sizeof(samplerState), sizeof(samplerState) * slotId);
	}
	void createDeviceBuffer(uint32_t slotId, int bufferSize) {
		void* devicePtr;
		cudaMalloc(&devicePtr, bufferSize);
		Impl::hsGeneralBuffer[slotId] = (char*)devicePtr;
		Impl::hsGeneralBufferSize[slotId] = bufferSize;
		cudaMemcpyToSymbol(Impl::csGeneralBuffer, &Impl::hsGeneralBuffer[slotId], sizeof(void*), sizeof(void*) * slotId);
		cudaMemcpyToSymbol(Impl::csGeneralBufferSize, &Impl::hsGeneralBufferSize[slotId], sizeof(int), sizeof(int) * slotId);
	}
	void copyHostBufferToBuffer(void* srcBuffer, int dstSlot, int size) {
		cudaMemcpy(Impl::hsGeneralBuffer[dstSlot], srcBuffer, size, cudaMemcpyHostToDevice);
	}
	int* getIndexBufferDeviceAddr(const int* hIndexBuffer, uint32_t indexBufferSize, int* dOldIndexBuffer) {
		if(dOldIndexBuffer != nullptr) {
			cudaFree(dOldIndexBuffer);
		}
		int* dIndexBuffer;
		cudaMalloc(&dIndexBuffer, indexBufferSize * sizeof(int));
		cudaMemcpy(dIndexBuffer, hIndexBuffer, indexBufferSize * sizeof(int), cudaMemcpyHostToDevice);
		return dIndexBuffer;
	}

	char* getVertexBufferDeviceAddr(const char* hVertexBuffer, uint32_t bufferSize, char* dOldBuffer) {
		if(dOldBuffer != nullptr) {
			cudaFree(dOldBuffer);
		}
		char* dBuffer;
		cudaMalloc(&dBuffer, bufferSize);
		cudaMemcpy(dBuffer, hVertexBuffer, bufferSize, cudaMemcpyHostToDevice);
		return dBuffer;
	}

	TypeDescriptorEnum* getTypeDescriptorDeviceAddr(const TypeDescriptorEnum* hBuffer, uint32_t bufferSize, TypeDescriptorEnum* dOldBuffer) {
		if(dOldBuffer != nullptr) {
			cudaFree(dOldBuffer);
		}
		TypeDescriptorEnum* dBuffer;
		cudaMalloc(&dBuffer, bufferSize * sizeof(TypeDescriptorEnum));
		cudaMemcpy(dBuffer, hBuffer, bufferSize * sizeof(TypeDescriptorEnum), cudaMemcpyHostToDevice);
		return dBuffer;
	}

	float* getDepthBufferDeviceAddr(uint32_t bufferSize, float* dOldBuffer) {
		if(dOldBuffer != nullptr) {
			cudaFree(dOldBuffer);
		}
		float* dBuffer;
		cudaMalloc(&dBuffer, bufferSize * sizeof(float));
		return dBuffer;
	}

	ifloat4* getPositionBufferDeviceAddr(uint32_t bufferSize, ifloat4* dOldBuffer) {
		if(dOldBuffer != nullptr) {
			cudaFree(dOldBuffer);
		}
		ifloat4* dBuffer;
		cudaMalloc(&dBuffer, bufferSize * sizeof(ifloat4));
		return dBuffer;
	}

	void getColorBufferDeviceAddr(
		const std::vector<ifloat4*>& hColorBuffer,
		std::vector<ifloat4*>& dhColorBuffer,
		ifloat4**& dColorBuffer,
		uint32_t bufferSize,
		std::vector<ifloat4*>& dhOldColorBuffer,
		ifloat4** dOldBuffer
	) {
		//TODO: POTENTIAL BUGS & CUDA DEVICE MEMORY LEAK
		if (dOldBuffer != nullptr) {
			cudaFree(dOldBuffer);
		}
		dhColorBuffer.resize(hColorBuffer.size());

		cudaMalloc(&dColorBuffer, hColorBuffer.size() * sizeof(ifloat4*));
		for (int i = 0; i < hColorBuffer.size(); i++) {
			cudaMalloc(&dhColorBuffer[i], bufferSize * sizeof(ifloat4));
			cudaMemcpy(dhColorBuffer[i], hColorBuffer[i], bufferSize * sizeof(ifloat4), cudaMemcpyHostToDevice);
		}
		cudaMemcpy(dColorBuffer, dhColorBuffer.data(), hColorBuffer.size() * sizeof(ifloat4*), cudaMemcpyHostToDevice);
	}

	void updateFrameBufferConstants(uint32_t width,uint32_t height) {
		auto invWidth = 1.0f / width;
		auto invHeight = 1.0f / height;
		cudaMemcpyToSymbol(Impl::csFrameWidthInv, &invWidth, sizeof(float));
		cudaMemcpyToSymbol(Impl::csFrameHeightInv, &invHeight, sizeof(float));
		cudaMemcpyToSymbol(Impl::csFrameWidth, &width, sizeof(uint32_t));
		cudaMemcpyToSymbol(Impl::csFrameHeight, &height, sizeof(uint32_t));
		Impl::hsFrameHeight = height;
		Impl::hsFrameWidth = width;
	}

	void invokeFragmentShaderUpdate(FragmentShader* dFragmentShader) IFRIT_AP_NOTHROW {
		Impl::MiscKernels::updateFragmentShaderKernel CU_KARG2(1, 1)(dFragmentShader);
		cudaDeviceSynchronize();
		//printf("Shader init done \n");
	}

	void updateVarying(uint32_t varyingCounts) {
		Impl::hsVaryingCounts = varyingCounts;
		cudaMemcpyToSymbol(Impl::csVaryingCounts, &varyingCounts, sizeof(uint32_t));
	}

	void updateAttributes(uint32_t attributeCounts) {
		Impl::hsAttributeCounts = attributeCounts;
		cudaMemcpyToSymbol(Impl::csAttributeCounts, &attributeCounts, sizeof(uint32_t));
	}

	void updateVertexCount(uint32_t vertexCount) {
		Impl::hsVertexCounts = vertexCount;
		cudaMemcpyToSymbol(Impl::csVertexCount, &vertexCount, sizeof(uint32_t));
	}

	void setBlendFunc(IfritColorAttachmentBlendState blendState) {
		Impl::hsBlendState = blendState;
		cudaMemcpyToSymbol(Impl::csBlendState, &blendState, sizeof(blendState));
		if (!blendState.blendEnable) {
			return;
		}
		Impl::MiscKernels::updateBlendStateKernel CU_KARG2(1, 1) ();
		cudaDeviceSynchronize();
	}

	void setDepthFunc(IfritCompareOp depthFunc) {
		cudaMemcpyToSymbol(Impl::csDepthFunc, &depthFunc, sizeof(depthFunc));
	}
	void setCullMode(IfritCullMode cullMode) {
		cudaMemcpyToSymbol(Impl::csCullMode, &cullMode, sizeof(cullMode));
	}

	void initCudaRendering() {
		Impl::InitializationKernels::globalInitializationKernel CU_KARG2(1, 1) ();
		cudaDeviceSynchronize();
		cudaMemcpyToSymbol(Impl::csCounterClosewiseCull, &Impl::hsCounterClosewiseCull, sizeof(Impl::hsCounterClosewiseCull));
	}
	void updateVertexLayout(TypeDescriptorEnum* dVertexTypeDescriptor, int attrCounts) {
		Impl::hsTotalVertexOffsets = 0;
		for (int i = 0; i < attrCounts; i++) {
			int cof = 0;
			Impl::hsVertexOffsets[i] = Impl::hsTotalVertexOffsets;
			if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_FLOAT1) cof = sizeof(float);
			else if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_FLOAT2) cof = sizeof(ifloat2);
			else if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_FLOAT3) cof = sizeof(ifloat3);
			else if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_FLOAT4)cof = sizeof(ifloat4);
			else if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_INT1) cof = sizeof(int);
			else if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_INT2) cof = sizeof(iint2);
			else if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_INT3) cof = sizeof(iint3);
			else if (dVertexTypeDescriptor[i] == TypeDescriptorEnum::IFTP_INT4) cof = sizeof(iint4);
			Impl::hsTotalVertexOffsets += cof;
			cudaMemcpyToSymbol(Impl::csVertexOffsets, &Impl::hsVertexOffsets[i], sizeof(Impl::hsVertexOffsets[0]), sizeof(Impl::hsVertexOffsets[0]) * i);
		}
		cudaMemcpyToSymbol(Impl::csTotalVertexOffsets, &Impl::hsTotalVertexOffsets, sizeof(Impl::hsTotalVertexOffsets));
	}

	void invokeFilledTriangleProcessing(const RenderingInvocationArgumentSet& args, cudaStream_t& computeStream) {
		if constexpr (CU_PROFILER_II_CPU_NSIGHT) {
			Impl::TrianglePipeline::unifiedFilledTriangleRasterEngineProfileEntry(
				args.totalIndices, args.dPositionBuffer, args.dIndexBuffer, args.deviceContext->dVaryingBufferM2,
				args.dColorBuffer, args.dDepthBuffer, args.dGeometryShader, args.dFragmentShader, computeStream);
		}
		else {
			if (args.gGeometryPipelineType == IFCUINVO_GEOMETRY_GENERATION_MESHSHADER) {
				Impl::MeshShadingStage::meshShadingEntryPoint CU_KARG4(1, 1, 0, computeStream) (
					args.gMeshShaderLocalSize, args.gMeshShaderNumWorkGroups, args.dMeshShader, args.gMeshShaderAttributes);
				Impl::TrianglePipeline::unifiedFilledTriangleRasterEngineKernel CU_KARG4(1, 1, 0, computeStream)(
					-1,nullptr,nullptr,nullptr,args.dColorBuffer,args.dDepthBuffer,nullptr,args.dFragmentShader
				);
			}
			else {
				Impl::TrianglePipeline::unifiedFilledTriangleRasterEngineKernel CU_KARG4(1, 1, 0, computeStream)(
					args.totalIndices, args.dPositionBuffer, args.dIndexBuffer, args.deviceContext->dVaryingBufferM2,
					args.dColorBuffer, args.dDepthBuffer, args.dGeometryShader, args.dFragmentShader);
			}
			
		}
	}

	void invokePointProcessing(const RenderingInvocationArgumentSet& args, cudaStream_t& computeStream) {
		if constexpr (CU_PROFILER_II_CPU_NSIGHT) {
			printf("Profiler for point mode is not supported now\n");
			std::abort();
		}
		else {
			Impl::PointPipeline::unifiedPointRasterEngineKernel CU_KARG4(1, 1, 0, computeStream)(
				args.totalIndices, args.dPositionBuffer, args.dIndexBuffer, args.deviceContext->dVaryingBufferM2,
				args.dColorBuffer, args.dDepthBuffer, args.dGeometryShader, args.dFragmentShader);
		}
	}

	void invokeLineProcessing(const RenderingInvocationArgumentSet& args, cudaStream_t& computeStream) {
		if constexpr (CU_PROFILER_II_CPU_NSIGHT) {
			printf("Profiler for point mode is not supported now\n");
			std::abort();
		}
		else {
			Impl::LinePipeline::unifiedLineRasterEngineKernel CU_KARG4(1, 1, 0, computeStream)(
				args.totalIndices, args.dPositionBuffer, args.dIndexBuffer, args.deviceContext->dVaryingBufferM2,
				args.dColorBuffer, args.dDepthBuffer, args.dGeometryShader, args.dFragmentShader);
		}
	}

	void invokeCudaRenderingClear(const RenderingInvocationArgumentSet& args) IFRIT_AP_NOTHROW {
		constexpr int dispatchThreadsX = 8, dispatchThreadsY = 8;
		int dispatchBlocksX = (Impl::hsFrameWidth / dispatchThreadsX) + ((Impl::hsFrameWidth % dispatchThreadsX) != 0);
		int dispatchBlocksY = (Impl::hsFrameHeight / dispatchThreadsY) + ((Impl::hsFrameHeight % dispatchThreadsY) != 0);

		Impl::ImageOperator::imageResetFloat32MonoKernel CU_KARG2(dim3(dispatchBlocksX, dispatchBlocksY), dim3(dispatchThreadsX, dispatchThreadsY))(
			args.dDepthBuffer, args.hClearDepth
			);
		for (int i = 0; i < args.dHostColorBufferSize; i++) {
			Impl::ImageOperator::imageResetFloat32RGBAKernel CU_KARG2(dim3(dispatchBlocksX, dispatchBlocksY), dim3(dispatchThreadsX, dispatchThreadsY))(
				(float*)args.dHostColorBuffer[i], { args.hClearColors[i].x,  args.hClearColors[i].y,  args.hClearColors[i].z,  args.hClearColors[i].w }
			);
		}
	}

	void invokeCudaRendering(const RenderingInvocationArgumentSet& args) IFRIT_AP_NOTHROW {
		// Stream Preparation
		static int initFlag = 0;
		static int secondPass = 0;
		static cudaStream_t copyStream, computeStream;
		static cudaEvent_t  copyStart, copyEnd;

		if (secondPass == 1) {
			if constexpr (CU_OPT_CUDA_PROFILE) {
				cudaProfilerStart();
			}
			secondPass = 2;
		}
		if (initFlag < 2) {
			if (initFlag == 0) {
				initFlag = 1;
				cudaDeviceSetLimit(cudaLimitMallocHeapSize, CU_HEAP_MEMORY_SIZE);
				cudaStreamCreate(&copyStream);
				cudaStreamCreate(&computeStream);
				cudaEventCreate(&copyStart);
				cudaEventCreate(&copyEnd);
				Impl::TriangleMiscStage::integratedInitKernel CU_KARG4(CU_MAX_TILE_X, CU_MAX_TILE_X, 0, computeStream)();
				cudaDeviceSynchronize();
			}
			if ((++initFlag) == 2) {
				secondPass = 1;
			}
		}
		
		// Compute
		std::chrono::high_resolution_clock::time_point end1 = std::chrono::high_resolution_clock::now();
		constexpr int dispatchThreadsX = 8,dispatchThreadsY = 8;
		int dispatchBlocksX = (Impl::hsFrameWidth / dispatchThreadsX) + ((Impl::hsFrameWidth % dispatchThreadsX) != 0);
		int dispatchBlocksY = (Impl::hsFrameHeight / dispatchThreadsY) + ((Impl::hsFrameHeight % dispatchThreadsY) != 0);
		int vertexExecutionBlocks = (Impl::hsVertexCounts / CU_VERTEX_PROCESSING_THREADS) + ((Impl::hsVertexCounts % CU_VERTEX_PROCESSING_THREADS) != 0);
		Impl::VertexStage::vertexProcessingKernel CU_KARG4(vertexExecutionBlocks, CU_VERTEX_PROCESSING_THREADS, 0, computeStream)(
			args.dVertexShader, Impl::hsVertexCounts, args.dVertexBuffer, args.dVertexTypeDescriptor,
			args.deviceContext->dVaryingBufferM2, (float4*)args.dPositionBuffer
		);
		
		Impl::TriangleMiscStage::integratedResetKernel CU_KARG4(CU_MAX_TILE_X * CU_MAX_SUBTILES_PER_TILE, CU_MAX_TILE_X, 0, computeStream)();
		
		if (args.polygonMode == IF_POLYGON_MODE_FILL) {
			invokeFilledTriangleProcessing(args, computeStream);
		}
		else if (args.polygonMode == IF_POLYGON_MODE_POINT) {
			invokePointProcessing(args, computeStream);
		}
		else if (args.polygonMode == IF_POLYGON_MODE_LINE) {
			invokeLineProcessing(args, computeStream);
		}
		else {
			printf("Unsupported polygon mode\n");
			std::abort();
		}
		
		if (!args.doubleBuffering) {
			cudaDeviceSynchronize();
		}

		// Memory Copy
		std::chrono::high_resolution_clock::time_point end2 = std::chrono::high_resolution_clock::now();
		if (args.doubleBuffering) {
			for (int i = 0; i < args.dHostColorBufferSize; i++) {
				if constexpr (CU_PROFILER_ENABLE_MEMCPY) {
					cudaMemcpyAsync(args.hColorBuffer[i], args.dLastColorBuffer[i], Impl::hsFrameWidth * Impl::hsFrameHeight * sizeof(ifloat4), cudaMemcpyDeviceToHost, copyStream);
				}
			}
		}
		cudaStreamSynchronize(computeStream);
		if (args.doubleBuffering) {
			cudaStreamSynchronize(copyStream);
		}
		if (!args.doubleBuffering) {
			for (int i = 0; i < args.dHostColorBufferSize; i++) {
				cudaMemcpy(args.hColorBuffer[i], args.dHostColorBuffer[i], Impl::csFrameWidth * Impl::csFrameHeight * sizeof(ifloat4), cudaMemcpyDeviceToHost);
			}
		}
	}
}
#endif