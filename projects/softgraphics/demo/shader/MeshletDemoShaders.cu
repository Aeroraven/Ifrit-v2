
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
#ifdef IFRIT_FEATURE_CUDA
#include "MeshletDemoShaders.cuh"
#include "engine/base/Shaders.h"
#include "engine/math/ShaderOpsCuda.cuh"
#include "engine/math/ShaderBuiltin.h"
#include "engine/math/ShaderBuiltinCuda.cuh"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"

namespace Ifrit::Demo::MeshletDemo {
	void MeshletDemoCuMS::execute(Vector3i localInvocation, int workGroupId, const void* inTaskShaderPayload, Ifrit::SoftRenderer::VaryingStore* outVaryings, Vector4f* outPos,
		int* outIndices, int& outNumVertices, int& outNumIndices) {

		using namespace Ifrit::SoftRenderer::Math::ShaderOps::CUDA;
		Matrix4x4f view = (LookAt({ 0,0.1,0.25 }, { 0,0.1,0.0 }, { 0,1,0 }));
		Matrix4x4f proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 0.1, 1000));
		Matrix4x4f mvp = multiply(proj, view);

		auto vertexData = reinterpret_cast<Vector4f*>(GetBufferPtr(0));
		auto indexData = reinterpret_cast<int*>(GetBufferPtr(1));
		auto vertOffsets = reinterpret_cast<int*>(GetBufferPtr(2));
		auto indOffsets = reinterpret_cast<int*>(GetBufferPtr(3));

		auto rsWorkGroupId = *reinterpret_cast<const int*>(inTaskShaderPayload);

		auto indStart = indOffsets[rsWorkGroupId], indEnd = indOffsets[rsWorkGroupId + 1];
		auto vertStart = vertOffsets[rsWorkGroupId], vertEnd = vertOffsets[rsWorkGroupId + 1];
		outNumIndices = indEnd - indStart;
		outNumVertices = vertEnd - vertStart;

		for (int i = 0; i < outNumVertices; i++) {
			auto s = vertexData[vertStart * 2 + i * 2];
			s.w = 1.0f;
			auto p = multiply(mvp, s);
			outPos[i] = p;
			outVaryings[i] = reinterpret_cast<Ifrit::SoftRenderer::VaryingStore*>(vertexData)[vertStart * 2 + i * 2 + 1];
		}
		for (int i = 0; i < outNumIndices; i++) {
			outIndices[i] = indexData[indStart+i];
		}
	}
	IFRIT_HOST Ifrit::SoftRenderer::MeshShader* MeshletDemoCuMS::GetCudaClone() {
		return Ifrit::SoftRenderer::Core::CUDA::hostGetDeviceObjectCopy<MeshletDemoCuMS>(this);
	}

	IFRIT_DUAL void MeshletDemoCuTS::execute(int workGroupId,void* outTaskShaderPayload,
		Vector3i* outMeshWorkGroups,int& outNumMeshWorkGroups) {

		using namespace Ifrit::SoftRenderer::Math::ShaderOps::CUDA;
		outNumMeshWorkGroups = 0;
		emitMeshTask({ 1,1,1 }, outMeshWorkGroups, outNumMeshWorkGroups);
		int* baseId = reinterpret_cast<int*>(outTaskShaderPayload);
		*baseId = workGroupId;

		
	}

	IFRIT_HOST Ifrit::SoftRenderer::TaskShader* MeshletDemoCuTS::GetCudaClone() {
		return Ifrit::SoftRenderer::Core::CUDA::hostGetDeviceObjectCopy<MeshletDemoCuTS>(this);

	}

	IFRIT_DUAL void MeshletDemoCuFS::execute(const  void* varyings, void* colorOutput, float* fragmentDepth) {
		auto result = isbcuReadPsVarying(varyings, 0);
		auto& co = isbcuReadPsColorOut(colorOutput, 0);
		co.x = result.x;
		co.y = result.y;
		co.z = result.z;
		co.w = result.w;
	}

	IFRIT_HOST Ifrit::SoftRenderer::FragmentShader* MeshletDemoCuFS::GetCudaClone() {
		return Ifrit::SoftRenderer::Core::CUDA::hostGetDeviceObjectCopy<MeshletDemoCuFS>(this);
	}
}

#endif