#include "utility/loader/WavefrontLoader.h"
#include "engine/math/ShaderOps.h"
namespace Ifrit::Utility::Loader {
	std::vector<float3> WavefrontLoader::remapNormals(std::vector<float3> normals, std::vector<uint32_t> indices, int numVertices) {
		using namespace Ifrit::Engine::Math::ShaderOps;

		std::vector<float3> retNormals;
		std::vector<int> counters;
		retNormals.clear();
		counters.clear();
		retNormals.resize(numVertices);
		counters.resize(numVertices);
		for (int i = 0; i < numVertices; i++) {
			retNormals[i] = { 0,0,0 };
			counters[i] = 0;
		}
		for (int i = 0; i < indices.size(); i += 3) {
			auto faceNode = indices[i];
			auto normalNode = indices[i + 2];
			retNormals[faceNode].x += normals[normalNode].x;
			retNormals[faceNode].y += normals[normalNode].y;
			retNormals[faceNode].z += normals[normalNode].z;
			counters[faceNode]++;
		}
		for (int i = 0; i < numVertices; i++) {
			retNormals[i] = normalize(retNormals[i]);
		}
		return retNormals;
	}
	void WavefrontLoader::loadObject(const char* path, std::vector<float3>& vertices,
		std::vector<float3>& normals, std::vector<float2>& uvs, std::vector<uint32_t>& indices) {

		// This section is auto-generated from Copilot

		std::ifstream file(path);
		std::string line;

		while (std::getline(file, line)) {
			std::istringstream iss(line);
			std::string type;
			iss >> type;

			if (type == "v") {
				float3 vertex;
				iss >> vertex.x >> vertex.y >> vertex.z;
				vertices.push_back(vertex);
			}
			else if (type == "vn") {
				float3 normal;
				iss >> normal.x >> normal.y >> normal.z;
				normals.push_back(normal);
			}
			else if (type == "vt") {
				float2 uv;
				iss >> uv.x >> uv.y;
				uvs.push_back(uv);
			}
			else if (type == "f") {
				std::string vertex;
				for (int i = 0; i < 3; i++) {
					iss >> vertex;
					std::istringstream vss(vertex);
					std::string index;
					for (int j = 0; j < 3; j++) {
						std::getline(vss, index, '/');
						if (index.size() != 0) {
							indices.push_back(std::stoi(index) - 1);
						}
						else {
							indices.push_back(0);
						}
						
					}
				}
			}
		}
	}
}