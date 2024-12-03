
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


#include "ifrit/meshproc/engine/mesh/MeshletConeCull.h"
#include "ifrit/common/math/simd/SimdVectors.h"
#include <stdexcept>
namespace Ifrit::MeshProcLib::MeshProcess {

void MeshletConeCullProc::createNormalCones(
    const MeshDescriptor &meshDesc, const std::vector<iint4> &meshlets,
    const std::vector<uint32_t> &meshletVertices,
    const std::vector<uint8_t> &meshletTriangles,
    std::vector<ifloat4> &normalConeAxisCutoff,
    std::vector<ifloat4> &normalConeApex, std::vector<ifloat4> &boundSphere) {
  using namespace Ifrit::Math::SIMD;
  if (meshDesc.vertexData == nullptr || meshDesc.indexData == nullptr ||
      meshDesc.normalData == nullptr) {
    throw std::runtime_error("Invalid mesh descriptor");
    return;
  }

  for (uint32_t i = 0; i < meshlets.size(); i++) {
    const iint4 &meshlet = meshlets[i];
    auto vertexCount = meshlet.z;
    auto triangleCount = meshlet.w;
    auto vertexOffset = meshlet.x;
    auto triangleOffset = meshlet.y;

    meshopt_Bounds bounds;
    const auto meshletVertStart = meshletVertices.data() + vertexOffset;
    const auto meshletTriStart = meshletTriangles.data() + triangleOffset;
    bounds = meshopt_computeMeshletBounds(
        meshletVertStart, meshletTriStart, triangleCount,
        (float *)meshDesc.vertexData, meshDesc.vertexCount,
        meshDesc.vertexStride);

    normalConeAxisCutoff.push_back({bounds.cone_axis[0], bounds.cone_axis[1],
                                    bounds.cone_axis[2], bounds.cone_cutoff});
    normalConeApex.push_back(
        {bounds.cone_apex[0], bounds.cone_apex[1], bounds.cone_apex[2], 0.0f});
    boundSphere.push_back(
        {bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius});
    printf("Bound sphere: %f %f %f %f\n", bounds.center[0], bounds.center[1],
           bounds.center[2], bounds.radius);
  }

  /*
      if (vertexCount == 0 || triangleCount == 0) {
        normalConeAxisCutoff.push_back(ifloat4(0.0f, 0.0f, 0.0f, 0.0f));
        throw std::runtime_error("Invalid vertex index");
      }
      // Iterate over the vertices of the meshlet
      vfloat3 avgNormal = vfloat3(0.0f, 0.0f, 0.0f);
      std::vector<vfloat3> normals;
      for (uint32_t j = 0; j < vertexCount; j++) {
        uint32_t vertexIndex = meshletVertices[vertexOffset + j];
        if (vertexIndex >= meshDesc.vertexCount) {
          throw std::runtime_error("Invalid vertex index");
          return;
        }
        // Get the normal of the vertex
        float *normal =
            (float *)(meshDesc.normalData + vertexIndex *
    meshDesc.normalStride); vfloat3 normalVec = vfloat3(normal[0], normal[1],
    normal[2]); normals.push_back(normalVec); avgNormal += normalVec;
      }
      avgNormal = normalize(avgNormal);

      // Iterate again to find the span angle
      float minAngleDot = 1.0f;
      for (uint32_t j = 0; j < vertexCount; j++) {
        vfloat3 normalVec = normals[j];
        float angle = dot(normalVec, avgNormal);
        minAngleDot = std::min(minAngleDot, angle);
      }
      float maxAngle = std::acos(minAngleDot);
      normalConeAxisCutoff.push_back(
          ifloat4(avgNormal.x, avgNormal.y, avgNormal.z, maxAngle));
      printf("Normal cone: %f %f %f %f\n", avgNormal.x, avgNormal.y,
    avgNormal.z, maxAngle);
    }
*/
}

} // namespace Ifrit::MeshProcLib::MeshProcess