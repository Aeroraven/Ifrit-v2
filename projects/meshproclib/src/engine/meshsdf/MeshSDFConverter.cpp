
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
#include "ifrit/meshproc/engine/meshsdf/MeshSDFConverter.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/math/simd/SimdVectors.h"
#include "ifrit/core/algo/Parallel.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/core/math/VectorOps.h"
#include "ifrit/core/math/SphericalSampling.h"

#include <random>

// TODO: Use algo in following paper
// Reference: https://www2.imm.dtu.dk/pubdb/edoc/imm1289.pdf

using namespace Ifrit::Math::SIMD;
namespace Ifrit::MeshProcLib::MeshSDFProcess
{

    IF_CONSTEXPR u32 cMinBvhChilds = 32;

    struct BVHNode : public NonCopyableStruct
    {
        SVector3f     bboxMin;
        SVector3f     bboxMax;
        Uref<BVHNode> left;
        Uref<BVHNode> right;
        u32           startIdx;
        u32           endIdx;
    };

    struct Mesh2SDFTempData : public NonCopyableStruct
    {
        SVector3f      bboxMin;
        SVector3f      bboxMax;
        f32*           meshVxBuffer;
        u32*           meshIxBuffer;
        u32            meshVxStride;
        u32            meshNumVerices;
        u32            meshNumIndices;

        Vec<u32>       asTriIndices;
        Uref<BVHNode>  asRoot;
        Vec<SVector3f> asTriBboxMin;
        Vec<SVector3f> asTriBboxMax;
        Vec<SVector3f> asTriBboxMid;
        Vec<SVector3f> asTriNormals;
    };

    IF_FORCEINLINE void ComputeMeshBoundingBox(Mesh2SDFTempData& data)
    {
        const u32 vertexCount = data.meshNumVerices;
        for (u32 i = 0; i < vertexCount; i++)
        {
            f32       vX = data.meshVxBuffer[i * data.meshVxStride + 0];
            f32       vY = data.meshVxBuffer[i * data.meshVxStride + 1];
            f32       vZ = data.meshVxBuffer[i * data.meshVxStride + 2];
            SVector3f vP = SVector3f(vX, vY, vZ);
            data.bboxMin = Min(data.bboxMin, vP);
            data.bboxMax = Max(data.bboxMax, vP);
        }
    }

    IF_FORCEINLINE SVector3f getTriangleNormal(const SVector3f& a, const SVector3f& b, const SVector3f& c)
    {
        return Normalize(Cross(b - a, c - a));
    }

    IF_FORCEINLINE void ComputeTriangleBoundingBox(Mesh2SDFTempData& data)
    {
        const u32 indexCount = data.meshNumIndices;
        data.asTriBboxMin.resize(indexCount / 3);
        data.asTriBboxMax.resize(indexCount / 3);
        data.asTriBboxMid.resize(indexCount / 3);
        data.asTriNormals.resize(indexCount / 3);
        for (u32 i = 0; i < indexCount; i += 3)
        {
            u32       i0             = data.meshIxBuffer[i + 0];
            u32       i1             = data.meshIxBuffer[i + 1];
            u32       i2             = data.meshIxBuffer[i + 2];
            f32       v0X            = data.meshVxBuffer[i0 * data.meshVxStride + 0];
            f32       v0Y            = data.meshVxBuffer[i0 * data.meshVxStride + 1];
            f32       v0Z            = data.meshVxBuffer[i0 * data.meshVxStride + 2];
            f32       v1X            = data.meshVxBuffer[i1 * data.meshVxStride + 0];
            f32       v1Y            = data.meshVxBuffer[i1 * data.meshVxStride + 1];
            f32       v1Z            = data.meshVxBuffer[i1 * data.meshVxStride + 2];
            f32       v2X            = data.meshVxBuffer[i2 * data.meshVxStride + 0];
            f32       v2Y            = data.meshVxBuffer[i2 * data.meshVxStride + 1];
            f32       v2Z            = data.meshVxBuffer[i2 * data.meshVxStride + 2];
            SVector3f v0             = SVector3f(v0X, v0Y, v0Z);
            SVector3f v1             = SVector3f(v1X, v1Y, v1Z);
            SVector3f v2             = SVector3f(v2X, v2Y, v2Z);
            SVector3f triMin         = Min(Min(v0, v1), v2);
            SVector3f triMax         = Max(Max(v0, v1), v2);
            data.asTriBboxMin[i / 3] = triMin;
            data.asTriBboxMax[i / 3] = triMax;
            data.asTriBboxMid[i / 3] = (triMin + triMax) * 0.5f;
            data.asTriNormals[i / 3] = getTriangleNormal(v0, v1, v2);
        }
    }
    IF_FORCEINLINE SVector3f PointDistToTriangle(
        const SVector3f& p, const SVector3f& a, const SVector3f& b, const SVector3f& c)
    {
        // Code from: https://github.com/RenderKit/embree/blob/master/tutorials/common/math/closest_point.h
        const auto ab = b - a;
        const auto ac = c - a;
        const auto ap = p - a;

        const f32  d1 = Dot(ab, ap);
        const f32  d2 = Dot(ac, ap);
        if (d1 <= 0.f && d2 <= 0.f)
            return a;

        const auto bp = p - b;
        const f32  d3 = Dot(ab, bp);
        const f32  d4 = Dot(ac, bp);
        if (d3 >= 0.f && d4 <= d3)
            return b;

        const auto cp = p - c;
        const f32  d5 = Dot(ab, cp);
        const f32  d6 = Dot(ac, cp);
        if (d6 >= 0.f && d5 <= d6)
            return c;

        const f32 vc = d1 * d4 - d3 * d2;
        if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f)
        {
            const f32 v = d1 / (d1 - d3);
            return a + ab * v;
        }

        const f32 vb = d5 * d2 - d1 * d6;
        if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f)
        {
            const f32 v = d2 / (d2 - d6);
            return a + ac * v;
        }

        const f32 va = d3 * d6 - d5 * d4;
        if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f)
        {
            const f32 v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + (c - b) * v;
        }

        const f32 denom = 1.f / (va + vb + vc);
        const f32 v     = vb * denom;
        const f32 w     = vc * denom;
        return a + ab * v + ac * w;
    }

    IF_FORCEINLINE f32 RayTrianleIntersection(
        const Vector3f& rayO, const Vector3f& rayD, const Vector3f& a, const Vector3f& b, const Vector3f& c)
    {
        using namespace Ifrit::Math;

        const f32 EPSILON = 0.000001f;

        Vector3f  edge1 = b - a;
        Vector3f  edge2 = c - a;

        Vector3f  h   = Cross(rayD, edge2);
        f32       det = Dot(edge1, h);

        if (det > -EPSILON && det < EPSILON)
            return FLT_MAX;

        f32      invDet = 1.0f / det;

        // Calculate distance from vertex a to ray origin
        Vector3f s = rayO - a;

        f32      u = Dot(s, h) * invDet;
        if (u < 0.0f || u > 1.0f)
            return FLT_MAX;

        Vector3f q = Cross(s, edge1);
        f32      v = Dot(rayD, q) * invDet;
        if (v < 0.0f || u + v > 1.0f)
            return FLT_MAX;

        f32 t = Dot(edge2, q) * invDet;

        return t;
    }

    IF_FORCEINLINE bool RayBoxIntersect(const SVector3f& rayO, const SVector3f& rayD, const SVector3f& bboxMin,
        const SVector3f& bboxMax, f32 tmin, f32 tmax)
    {
        using namespace Ifrit::Math;
        using namespace Ifrit::Math::SIMD;

        SVector3f invDir = SVector3f(1.0f) / rayD;
        SVector3f t0     = (bboxMin - rayO) * invDir;
        SVector3f t1     = (bboxMax - rayO) * invDir;
        tmin             = std::max(std::max(std::min(t0.x, t1.x), std::min(t0.y, t1.y)), std::min(t0.z, t1.z));
        tmax             = std::min(std::min(std::max(t0.x, t1.x), std::max(t0.y, t1.y)), std::max(t0.z, t1.z));
        return tmax > tmin;
    }

    void CalculateAsChildBbox(const Mesh2SDFTempData& data, Uref<BVHNode>& node)
    {
        node->bboxMin = SVector3f(FLT_MAX);
        node->bboxMax = SVector3f(-FLT_MAX);
        for (u32 i = node->startIdx; i < node->endIdx; i++)
        {
            auto triId    = data.asTriIndices[i];
            node->bboxMin = Min(node->bboxMin, data.asTriBboxMin[triId]);
            node->bboxMax = Max(node->bboxMax, data.asTriBboxMax[triId]);
        }
    }

    void BuildAccelStructRecur(Mesh2SDFTempData& data, Uref<BVHNode>& node)
    {
        auto numChildren = node->endIdx - node->startIdx;
        if (numChildren <= cMinBvhChilds)
        {
            return;
        }
        auto longestAxis = 0;
        auto axisLength  = node->bboxMax.x - node->bboxMin.x;
        if (node->bboxMax.y - node->bboxMin.y > axisLength)
        {
            longestAxis = 1;
            axisLength  = node->bboxMax.y - node->bboxMin.y;
        }
        if (node->bboxMax.z - node->bboxMin.z > axisLength)
        {
            longestAxis = 2;
        }
        auto mid = (node->bboxMin + node->bboxMax) * 0.5f;

        // Inplace rearrange
        auto leftIdx  = node->startIdx;
        auto rightIdx = node->endIdx - 1;

        f32  totalMid = 0;
        for (u32 i = node->startIdx; i < node->endIdx; i++)
        {
            totalMid += ElementAt(data.asTriBboxMid[data.asTriIndices[i]], longestAxis);
        }
        f32 avgMid = totalMid / numChildren;

        while (leftIdx < rightIdx)
        {
            while (leftIdx < rightIdx && ElementAt(data.asTriBboxMid[data.asTriIndices[leftIdx]], longestAxis) < avgMid)
            {
                leftIdx++;
            }
            while (
                leftIdx < rightIdx && ElementAt(data.asTriBboxMid[data.asTriIndices[rightIdx]], longestAxis) >= avgMid)
            {
                rightIdx--;
            }
            if (leftIdx < rightIdx)
            {
                std::swap(data.asTriIndices[leftIdx], data.asTriIndices[rightIdx]);
                leftIdx++;
                rightIdx--;
            }
        }
        if (leftIdx == node->startIdx || leftIdx == node->endIdx || rightIdx == node->endIdx - 1)
        {
            if (node->endIdx - node->startIdx > 128)
            {
                iWarn("Large BVH node with {} children, leftIdx:{}, rightIdx:{} | st:{}, ed:{}",
                    node->endIdx - node->startIdx, leftIdx, rightIdx, node->startIdx, node->endIdx);
                iWarn("AvgMid: {}", avgMid);
            }
            return;
        }

        if (node->endIdx - node->startIdx < cMinBvhChilds)
        {
            return;
        }

        node->left           = std::make_unique<BVHNode>();
        node->left->startIdx = node->startIdx;
        node->left->endIdx   = leftIdx;
        CalculateAsChildBbox(data, node->left);

        node->right           = std::make_unique<BVHNode>();
        node->right->startIdx = leftIdx;
        node->right->endIdx   = node->endIdx;
        CalculateAsChildBbox(data, node->right);

        BuildAccelStructRecur(data, node->left);
        BuildAccelStructRecur(data, node->right);
    }

    void BuildAccelStruct(Mesh2SDFTempData& data)
    {
        data.asRoot           = std::make_unique<BVHNode>();
        data.asRoot->bboxMin  = data.bboxMin;
        data.asRoot->bboxMax  = data.bboxMax;
        data.asRoot->startIdx = 0;
        data.asRoot->endIdx   = data.asTriIndices.size();
        BuildAccelStructRecur(data, data.asRoot);
    }

    IF_FORCEINLINE f32 GetDistanceToBbox(const SVector3f& p, const SVector3f& bboxMin, const SVector3f& bboxMax)
    {
        SVector3f dxMin  = bboxMin - p;
        SVector3f dxMax  = p - bboxMax;
        SVector3f dxZero = SVector3f(0.0f);
        SVector3f dx     = Max(Max(dxMin, dxMax), dxZero);
        f32       dist   = Length(dx);
        return dist;
    }

    void GetSignedDistanceToMeshRecur(const Mesh2SDFTempData& data, const SVector3f& p, BVHNode* node, f32& tgtDist)
    {
        auto leftChild  = node->left.get();
        auto rightChild = node->right.get();
        if (leftChild && rightChild)
        {
            // Non child node, check distance to bbox
            auto leftChildDist  = GetDistanceToBbox(p, leftChild->bboxMin, leftChild->bboxMax);
            auto rightChildDist = GetDistanceToBbox(p, rightChild->bboxMin, rightChild->bboxMax);
            if (leftChildDist <= rightChildDist && leftChildDist < std::abs(tgtDist))
            {
                GetSignedDistanceToMeshRecur(data, p, leftChild, tgtDist);
                if (std::abs(tgtDist) > rightChildDist)
                {
                    GetSignedDistanceToMeshRecur(data, p, rightChild, tgtDist);
                }
            }
            else if (rightChildDist < leftChildDist && rightChildDist < std::abs(tgtDist))
            {
                GetSignedDistanceToMeshRecur(data, p, rightChild, tgtDist);
                if (std::abs(tgtDist) > leftChildDist)
                {
                    GetSignedDistanceToMeshRecur(data, p, leftChild, tgtDist);
                }
            }
        }
        else
        {
            // child node, check distance to triangle
            for (u32 i = node->startIdx; i < node->endIdx; i++)
            {
                auto triId = data.asTriIndices[i];
                auto i0    = data.meshIxBuffer[triId * 3 + 0];
                auto i1    = data.meshIxBuffer[triId * 3 + 1];
                auto i2    = data.meshIxBuffer[triId * 3 + 2];

                auto v0 = SVector3f(data.meshVxBuffer[i0 * data.meshVxStride + 0],
                    data.meshVxBuffer[i0 * data.meshVxStride + 1], data.meshVxBuffer[i0 * data.meshVxStride + 2]);
                auto v1 = SVector3f(data.meshVxBuffer[i1 * data.meshVxStride + 0],
                    data.meshVxBuffer[i1 * data.meshVxStride + 1], data.meshVxBuffer[i1 * data.meshVxStride + 2]);
                auto v2 = SVector3f(data.meshVxBuffer[i2 * data.meshVxStride + 0],
                    data.meshVxBuffer[i2 * data.meshVxStride + 1], data.meshVxBuffer[i2 * data.meshVxStride + 2]);

                auto vNormal   = data.asTriNormals[triId];
                auto nearestPt = PointDistToTriangle(p, v0, v1, v2);
                auto vDist     = p - nearestPt;
                auto sign      = Dot(vDist, vNormal) > 0.0f ? 1.0f : -1.0f;
                auto dist      = Length(vDist) * sign;
                if (std::abs(dist) < std::abs(tgtDist))
                {
                    tgtDist = dist;
                }
            }
        }
    }

    f32 GetSignedDistanceToMesh(const Mesh2SDFTempData& data, const SVector3f& p)
    {
        f32 tgtDist = FLT_MAX;
        GetSignedDistanceToMeshRecur(data, p, data.asRoot.get(), tgtDist);
        return tgtDist;
    }

    void GetSignedDistanceToMeshRayTraceSingleRayRecur(
        BVHNode* node, const Mesh2SDFTempData& data, const SVector3f& p, const SVector3f& dir, Vector4f& collResult)
    {
        // printf("Tracing: %p, L:%d, R:%d\n", node, node->startIdx, node->endIdx);
        auto leftChild  = node->left.get();
        auto rightChild = node->right.get();
        if (leftChild && rightChild)
        {
            // Non child node, check distance to bbox

            bool rayIntersectLeft  = RayBoxIntersect(p, dir, leftChild->bboxMin, leftChild->bboxMax, 0.0f, FLT_MAX);
            bool rayIntersectRight = RayBoxIntersect(p, dir, rightChild->bboxMin, rightChild->bboxMax, 0.0f, FLT_MAX);

            if (rayIntersectLeft)
            {
                GetSignedDistanceToMeshRayTraceSingleRayRecur(leftChild, data, p, dir, collResult);
            }
            if (rayIntersectRight)
            {
                GetSignedDistanceToMeshRayTraceSingleRayRecur(rightChild, data, p, dir, collResult);
            }
        }
        else
        {
            // child node, check distance to triangle
            for (u32 i = node->startIdx; i < node->endIdx; i++)
            {
                auto triId = data.asTriIndices[i];
                auto i0    = data.meshIxBuffer[triId * 3 + 0];
                auto i1    = data.meshIxBuffer[triId * 3 + 1];
                auto i2    = data.meshIxBuffer[triId * 3 + 2];

                auto v0 = SVector3f(data.meshVxBuffer[i0 * data.meshVxStride + 0],
                    data.meshVxBuffer[i0 * data.meshVxStride + 1], data.meshVxBuffer[i0 * data.meshVxStride + 2]);
                auto v1 = SVector3f(data.meshVxBuffer[i1 * data.meshVxStride + 0],
                    data.meshVxBuffer[i1 * data.meshVxStride + 1], data.meshVxBuffer[i1 * data.meshVxStride + 2]);
                auto v2 = SVector3f(data.meshVxBuffer[i2 * data.meshVxStride + 0],
                    data.meshVxBuffer[i2 * data.meshVxStride + 1], data.meshVxBuffer[i2 * data.meshVxStride + 2]);

                auto vNormal = data.asTriNormals[triId];

                auto tP   = Vector3f(p.x, p.y, p.z);
                auto tD   = Vector3f(dir.x, dir.y, dir.z);
                auto t0   = Vector3f(v0.x, v0.y, v0.z);
                auto t1   = Vector3f(v1.x, v1.y, v1.z);
                auto t2   = Vector3f(v2.x, v2.y, v2.z);
                auto dist = RayTrianleIntersection(tP, tD, t0, t1, t2);
                if (dist < 0.0f)
                {
                    continue;
                }
                // printf("Dist: %f\n", dist);
                auto svDot = Dot(vNormal, dir);
                if (svDot > 0.0f)
                {
                    dist = -dist;
                }

                if (std::abs(dist) < std::abs(collResult.w))
                {
                    collResult = Vector4f(vNormal.x, vNormal.y, vNormal.z, dist);
                }
            }
        }
    }

    f32 GetSignedDistanceToMeshRayTraceSingleRay(const Mesh2SDFTempData& data, const SVector3f& p, const SVector3f& dir)
    {
        Vector4f collResult = Vector4f(0.0f, 0.0f, 0.0f, FLT_MAX);
        auto     root       = data.asRoot.get();
        GetSignedDistanceToMeshRayTraceSingleRayRecur(root, data, p, dir, collResult);

        // printf("Coll: %f %f %f %f\n", collResult.x, collResult.y, collResult.z, collResult.w);
        return collResult.w;
    }

    f32 GetSignedDistanceToMeshRayTrace(const Mesh2SDFTempData& data, const SVector3f& p, const Vec<Vector3f>& samples)
    {
        auto optimalDistance = FLT_MAX;
        auto pT              = Vector3f(p.x, p.y, p.z);

        u32  hit        = 0;
        u32  numSamples = SizeCast<u32>(samples.size());

        // Follow unreal's Mesh DF implementation, it judges whether the ray hits the backside of the triangle to
        // determine whether the point is inside the mesh.
        u32  hitBackside = 0;

        // Check each sample
        for (auto& sample : samples)
        {
            using namespace Ifrit::Math;

            auto dir = Math::Normalize(sample);
            auto pV  = pT - dir * 1e-4f;

            auto dist    = GetSignedDistanceToMeshRayTraceSingleRay(data, p, SVector3f(dir.x, dir.y, dir.z));
            auto distAbs = std::abs(dist);
            if (distAbs < optimalDistance)
            {
                optimalDistance = distAbs;
            }
            if (dist < FLT_MAX)
            {
                hit++;
                if (dist < 0.0f)
                {
                    hitBackside++;
                }
            }
        }

        if (hit > 0 && hitBackside > 0.25f * numSamples)
        {
            // If all samples hit the backside of the triangle, the point is inside the mesh
            optimalDistance = -optimalDistance;
        }
        return optimalDistance;
    }

    IFRIT_MESHPROC_API void ConvertMeshToSDF(const MeshDescriptor& meshDesc, SignedDistanceField& sdf, u32 sdfWidth,
        u32 sdfHeight, u32 sdfDepth, SDFGenerateMethod method)
    {
        // TODO: this is a trivial implementation, that has worse performance O(NM), where N is the number of voxels and
        // M is the number of triangles, although AS approach is used. However, mesh df is generated offline. Better
        // approach like "Jump Flooding" should be considered later.

        iDebug("Converting mesh to SDF: V={} I={}", meshDesc.vertexCount, meshDesc.indexCount);

        Mesh2SDFTempData data;
        data.bboxMin        = SVector3f(FLT_MAX);
        data.bboxMax        = SVector3f(-FLT_MAX);
        data.meshVxBuffer   = reinterpret_cast<f32*>(meshDesc.vertexData);
        data.meshIxBuffer   = reinterpret_cast<u32*>(meshDesc.indexData);
        data.meshVxStride   = meshDesc.vertexStride / sizeof(f32);
        data.meshNumVerices = meshDesc.vertexCount;
        data.meshNumIndices = meshDesc.indexCount;
        ComputeMeshBoundingBox(data);
        ComputeTriangleBoundingBox(data);

        // dilate the bbox by a small amount, like 5%
        auto bboxDilate = (data.bboxMax - data.bboxMin) * 0.1f;
        data.bboxMin -= bboxDilate;
        data.bboxMax += bboxDilate;

        // build accel structure
        data.asTriIndices.resize(meshDesc.indexCount / 3);
        for (u32 i = 0; i < meshDesc.indexCount / 3; i++)
        {
            data.asTriIndices[i] = i;
        }
        BuildAccelStruct(data);

        // then, for each voxel, calculate the distance to the mesh
        sdf.width  = sdfWidth;
        sdf.height = sdfHeight;
        sdf.depth  = sdfDepth;
        sdf.sdfData.resize(sdfWidth * sdfHeight * sdfDepth);
        auto totalVoxels = sdfWidth * sdfHeight * sdfDepth;

        if (method == SDFGenerateMethod::Trivial)
        {
            UnorderedFor<u32>(0, totalVoxels, [&](u32 el) {
                auto      depth  = el / (sdfWidth * sdfHeight);
                auto      height = (el % (sdfWidth * sdfHeight)) / sdfWidth;
                auto      width  = (el % (sdfWidth * sdfHeight)) % sdfWidth;
                f32       x      = ((f32)width + 0.5f) / (f32)sdfWidth;
                f32       y      = ((f32)height + 0.5f) / (f32)sdfHeight;
                f32       z      = ((f32)depth + 0.5f) / (f32)sdfDepth;
                f32       lx     = std::lerp(data.bboxMin.x, data.bboxMax.x, x);
                f32       ly     = std::lerp(data.bboxMin.y, data.bboxMax.y, y);
                f32       lz     = std::lerp(data.bboxMin.z, data.bboxMax.z, z);
                SVector3f p      = SVector3f(lx, ly, lz);
                f32       dist   = GetSignedDistanceToMesh(data, p);
                sdf.sdfData[el]  = dist;
            });
            sdf.bboxMin = Vector3f(data.bboxMin.x, data.bboxMin.y, data.bboxMin.z);
            sdf.bboxMax = Vector3f(data.bboxMax.x, data.bboxMax.y, data.bboxMax.z);
        }
        else if (method == SDFGenerateMethod::RayTracing)
        {
            // random engine for sampling, uniform distribution for f32
            std::random_device                  rd;
            std::mt19937                        gen(rd());
            std::uniform_real_distribution<f32> dis(0.0f, 1.0f);

            Vec<Vector3f>                       samples;
            constexpr u32                       sqrtNumSamples = 7;
            constexpr u32                       numSamples     = sqrtNumSamples * sqrtNumSamples;
            samples.reserve(numSamples);
            for (u32 i = 0; i < numSamples; i++)
            {
                auto fractX = dis(gen);
                auto fractY = dis(gen);

                auto fX = i % sqrtNumSamples;
                auto fY = i / sqrtNumSamples;

                auto sX = (fX + fractX) / sqrtNumSamples;
                auto sY = (fY + fractY) / sqrtNumSamples;

                auto sample = Math::ConcentricOctahedralTransform(Vector2f(sX, sY));
                samples.push_back(Vector3f(sample.x, sample.y, sample.z));
            }

            auto minBound = std::min(data.bboxMax.x - data.bboxMin.x,
                std::min(data.bboxMax.y - data.bboxMin.y, data.bboxMax.z - data.bboxMin.z));

            UnorderedFor<u32>(0, totalVoxels, [&](u32 el) {
                auto      depth    = el / (sdfWidth * sdfHeight);
                auto      height   = (el % (sdfWidth * sdfHeight)) / sdfWidth;
                auto      width    = (el % (sdfWidth * sdfHeight)) % sdfWidth;
                f32       x        = ((f32)width + 0.5f) / (f32)sdfWidth;
                f32       y        = ((f32)height + 0.5f) / (f32)sdfHeight;
                f32       z        = ((f32)depth + 0.5f) / (f32)sdfDepth;
                f32       lx       = std::lerp(data.bboxMin.x, data.bboxMax.x, x);
                f32       ly       = std::lerp(data.bboxMin.y, data.bboxMax.y, y);
                f32       lz       = std::lerp(data.bboxMin.z, data.bboxMax.z, z);
                SVector3f p        = SVector3f(lx, ly, lz);
                f32       distSign = GetSignedDistanceToMeshRayTrace(data, p, samples);
                f32       dist     = std::abs(GetSignedDistanceToMesh(data, p));
                dist               = (distSign > 0.0f) ? dist : -dist;
                sdf.sdfData[el]    = dist;
            });
            sdf.bboxMin = Vector3f(data.bboxMin.x, data.bboxMin.y, data.bboxMin.z);
            sdf.bboxMax = Vector3f(data.bboxMax.x, data.bboxMax.y, data.bboxMax.z);
        }
    }

} // namespace Ifrit::MeshProcLib::MeshSDFProcess