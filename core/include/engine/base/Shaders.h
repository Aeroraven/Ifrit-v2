#pragma once

#include "engine/base/VaryingStore.h"
#include "engine/base/Structures.h"
#include "engine/base/VaryingDescriptor.h"

namespace Ifrit::Engine {
	enum GeometryShaderTopology {
		IGST_TRIANGLES = 0,
		IGST_LINES = 1,
		IGST_POINTS = 2
	};

	class IFRIT_APIDECL ShaderBase {
	public:
		float* atTexture[32];
		uint32_t atTextureWid[32];
		uint32_t atTextureHei[32];
		IfritSamplerT atSamplerPtr[32];
		char* atBuffer[32];
		bool isThreadSafe = true;
	};

	class IFRIT_APIDECL VertexShader :public ShaderBase {
	public:
		IFRIT_DUAL virtual void execute(
			const void* const* input,
			ifloat4* outPos,
			VaryingStore* const* outVaryings
		) = 0;
		IFRIT_HOST virtual VertexShader* getCudaClone() { return nullptr; };
		IFRIT_HOST virtual VertexShader* getThreadLocalCopy() { return nullptr; };
		IFRIT_HOST virtual void updateUniformData(int binding, int set, const void* pData) {}
		IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() { return{}; }
		IFRIT_HOST virtual VaryingDescriptor getVaryingDescriptor() { return{}; }
	};

	class IFRIT_APIDECL FragmentShader :public ShaderBase {
	public:
		bool allowDepthModification = false;
		IFRIT_DUAL virtual void execute(
			const void* varyings, 
			void* colorOutput,
			float* fragmentDepth
		) = 0;
		IFRIT_HOST virtual FragmentShader* getCudaClone() { return nullptr; };
		IFRIT_HOST virtual FragmentShader* getThreadLocalCopy() { return nullptr; };
		IFRIT_HOST virtual void updateUniformData(int binding, int set, const void* pData) {}
		IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() { return{}; }
	};

	class IFRIT_APIDECL GeometryShader :public ShaderBase {
	public:
		GeometryShaderTopology atTopology = IGST_TRIANGLES;
		uint32_t atMaxVertices = 4;
		IFRIT_DUAL virtual void execute(
			const ifloat4* const*  inPos,
			const VaryingStore* const*  inVaryings,
			ifloat4* outPos,
			VaryingStore* outVaryings,
			int* outSize
		) = 0;
		IFRIT_HOST virtual GeometryShader* getCudaClone() { return nullptr; };
	};

	class IFRIT_APIDECL HullShader :public ShaderBase {
	public:
		IFRIT_DUAL virtual void executeMain(
			const ifloat4** inputPos,
			const VaryingStore** inputVaryings,
			ifloat4* outPos,
			VaryingStore* outVaryings,
			int invocationId,
			int patchId
		) = 0;
		IFRIT_DUAL virtual void executePatchFunc(
			const ifloat4** inputPos,
			const VaryingStore** inputVaryings,
			int* outerTessLevels,
			int* innerTessLevels,
			int invocationId,
			int patchId
		) = 0;
		IFRIT_HOST virtual HullShader* getCudaClone() { return nullptr; };
	};

	class IFRIT_APIDECL MeshShader :public ShaderBase {
	public:
		IFRIT_DUAL virtual void execute(
			iint3 localInvocation, 
			int workGroupId,
			const void* inTaskShaderPayload,
			VaryingStore* outVaryings,
			ifloat4* outPos,
			int* outIndices,
			int& outNumVertices,
			int& outNumIndices
		) = 0;
		IFRIT_HOST virtual MeshShader* getCudaClone() { return nullptr; };
	};

	class IFRIT_APIDECL TaskShader : public ShaderBase {
	public:
		IFRIT_DUAL virtual void execute(
			int workGroupId,
			void* outTaskShaderPayload,
			iint3* outMeshWorkGroups,
			int& outNumMeshWorkGroups
		) = 0;
		IFRIT_HOST virtual TaskShader* getCudaClone() { return nullptr; };
	};

	/* Function Delegates & C-ABI Compatible */
	typedef void (*VertexShaderFunctionalPtr)(const void* const* input, ifloat4* outPos, VaryingStore* const* outVaryings);
	typedef void (*FragmentShaderFunctionalPtr)(const void* varyings, void* colorOutput, float* fragmentDepth);
	typedef void (*GeometryShaderFunctionalPtr)(const ifloat4* const* inPos, const VaryingStore* const* inVaryings,
		ifloat4* outPos, VaryingStore* outVaryings, int* outSize);
}