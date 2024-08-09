#pragma once
#ifdef IFRIT_FEATURE_CUDA
#include "MeshletDemoShaders.cuh"
#include "engine/base/Shaders.h"
#include "engine/math/ShaderOpsCuda.cuh"
#include "engine/math/ShaderBuiltin.h"
#include "engine/math/ShaderBuiltinCuda.cuh"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"

namespace Ifrit::Demo::MeshletDemo {
	void MeshletDemoCuMS::execute(iint3 localInvocation, int workGroupId, Ifrit::Engine::VaryingStore* outVaryings, ifloat4* outPos,
		int* outIndices, int& outNumVertices, int& outNumIndices) {

		using namespace Ifrit::Engine::Math::ShaderOps::CUDA;
		auto vertexData = reinterpret_cast<ifloat4*>(getBufferPtr(0));
		auto indexData = reinterpret_cast<int*>(getBufferPtr(1));
		auto vertOffsets = reinterpret_cast<int*>(getBufferPtr(2));
		auto indOffsets = reinterpret_cast<int*>(getBufferPtr(3));

		auto indStart = indOffsets[workGroupId], indEnd = indOffsets[workGroupId + 1];
		auto vertStart = vertOffsets[workGroupId], vertEnd = vertOffsets[workGroupId + 1];
		outNumIndices = indEnd - indStart;
		outNumVertices = vertEnd - vertStart;

		for (int i = 0; i < outNumVertices; i++) {
			outPos[i] = vertexData[vertStart * 2 + i * 2];
			outVaryings[i] = reinterpret_cast<Ifrit::Engine::VaryingStore*>(vertexData)[vertStart * 2 + i * 2 + 1];
		}
		for (int i = 0; i < outNumIndices; i++) {
			outIndices[i] = indexData[i];
		}
	}
	IFRIT_HOST Ifrit::Engine::MeshShader* MeshletDemoCuMS::getCudaClone() {
		return Ifrit::Core::CUDA::hostGetDeviceObjectCopy<MeshletDemoCuMS>(this);
	}

	IFRIT_DUAL void MeshletDemoCuFS::execute(const  void* varyings, void* colorOutput) {
		auto& co = isbcuReadPsColorOut(colorOutput, 0);
		co.x = 1.0f;
		co.y = 1.0f;
		co.z = 1.0f;
		co.w = 1.0f;
	}

	IFRIT_HOST Ifrit::Engine::FragmentShader* MeshletDemoCuFS::getCudaClone() {
		return Ifrit::Core::CUDA::hostGetDeviceObjectCopy<MeshletDemoCuFS>(this);
	}
}

#endif