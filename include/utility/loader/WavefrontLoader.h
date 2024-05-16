#pragma once
#include "core/definition/CoreExports.h"

namespace Ifrit::Utility::Loader {
	class WavefrontLoader {
	public:
		void loadObject(const char* path, std::vector<ifloat3>& vertices, std::vector<ifloat3>& normals, std::vector<ifloat2>& uvs, std::vector<uint32_t>& indices);
		std::vector<ifloat3> remapNormals(std::vector<ifloat3> normals, std::vector<uint32_t> indices, int numVertices);
	};
}