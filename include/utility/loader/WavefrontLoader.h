#pragma once
#include "core/definition/CoreExports.h"

namespace Ifrit::Utility::Loader {
	class WavefrontLoader {
	public:
		void loadObject(const char* path, std::vector<float3>& vertices, std::vector<float3>& normals, std::vector<float2>& uvs, std::vector<uint32_t>& indices);
		std::vector<float3> remapNormals(std::vector<float3> normals, std::vector<uint32_t> indices, int numVertices);
	};
}