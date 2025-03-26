
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

#include "ifrit/softgraphics/engine/meshletbuilder/MeshletBuilder.h"
#include "ifrit/common/util/TypingUtil.h"
using namespace Ifrit::Common::Utility;

namespace Ifrit::Graphics::SoftGraphics::MeshletBuilder::Impl
{
	struct MbTriangle
	{
		Vector3i ind;
		Vector3f normal;
		Vector3f centroid;
	};

	struct MbCurrentMeshlet
	{
		std::vector<int> isvertexUsed;
		std::vector<int> usedVertices;
		std::vector<int> triangleContained;
		Vector3f		 sumCentroid = { 0.0f, 0.0f, 0.0f };
		Vector3f		 sumNormal = { 0.0f, 0.0f, 0.0f };
		float			 invSumNormal = 0.0f;
	};

	struct MbContext
	{
		std::vector<Vector4f>				   vertices;
		std::vector<MbTriangle>				   triangles;
		std::vector<std::vector<int>>		   adjMaps;
		std::vector<int>					   remainActiveAdj;
		std::vector<int>					   triangleEmitted;
		std::vector<MbCurrentMeshlet>		   finishedMeshlets;
		std::vector<std::unique_ptr<Meshlet>>* generatedMeshlets;
		const int							   maxTriangles = 128;
		const int							   maxVertices = 256;
	};

	void initializeContext(MbContext* ctx, const VertexBuffer& vbuf, const std::vector<int>& ibuf, int posAttrId)
	{

		auto cross = [](Vector3f a, Vector3f b) {
			return Vector3f{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
		};
		auto centroid = [](Vector4f a, Vector4f b, Vector4f c) {
			return Vector3f{ (a.x + b.x + c.x) / 3, (a.y + b.y + c.y) / 3, (a.z + b.z + c.z) / 3 };
		};
		auto veclength = [](Vector3f a) { return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z); };

		auto totVerts = vbuf.getVertexCount();
		auto totTris = ibuf.size() / 3;
		ctx->vertices.resize(totVerts);
		ctx->triangles.resize(totTris);
		ctx->adjMaps.resize(totVerts);
		ctx->remainActiveAdj.resize(totVerts);
		ctx->triangleEmitted.resize(totTris);
		for (int i = 0; i < totVerts; i++)
		{
			auto v = vbuf.getValue<Vector4f>(i, posAttrId);
			ctx->vertices[i] = v;
			ctx->adjMaps[i].clear();
		}
		for (int i = 0; i < totTris; i++)
		{
			Vector3i ind = { ibuf[i * 3], ibuf[i * 3 + 1], ibuf[i * 3 + 2] };
			auto	 pa = ctx->vertices[ind.x], pb = ctx->vertices[ind.y], pc = ctx->vertices[ind.z];
			Vector3f papb = { pa.x - pb.x, pa.y - pb.y, pa.z - pb.z };
			Vector3f papc = { pa.x - pc.x, pa.y - pc.y, pa.z - pc.z };
			Vector3f normal = cross(papb, papc);
			float	 normalLen = veclength(normal);
			normal.x /= normalLen;
			normal.y /= normalLen;
			normal.z /= normalLen;
			Vector3f center = centroid(pa, pb, pc);

			ctx->triangles[i].centroid = center;
			ctx->triangles[i].ind = ind;
			ctx->triangles[i].normal = normal;
			ctx->triangleEmitted[i] = 0;
		}

		// Build adjacency maps
		for (int i = 0; i < totTris; i++)
		{
			ctx->adjMaps[ctx->triangles[i].ind.x].push_back(i);
			ctx->adjMaps[ctx->triangles[i].ind.y].push_back(i);
			ctx->adjMaps[ctx->triangles[i].ind.z].push_back(i);
		}
		for (int i = 0; i < totVerts; i++)
		{
			ctx->remainActiveAdj[i] = Ifrit::Common::Utility::SizeCast<int>(ctx->adjMaps[i].size());
		}
	}

	int meshletExpansion(const MbContext& ctx, const MbCurrentMeshlet& meshlet)
	{
		// Algo Process Reference:
		// https://github.com/zeux/meshoptimizer/blob/master/src/clusterizer.cpp
		Vector3f clusterNormal = { meshlet.sumNormal.x * meshlet.invSumNormal, meshlet.sumNormal.y * meshlet.invSumNormal,
			meshlet.sumNormal.z * meshlet.invSumNormal };
		Vector3f clusterCentroid = { meshlet.sumCentroid.x / meshlet.triangleContained.size(),
			meshlet.sumCentroid.y / meshlet.triangleContained.size(),
			meshlet.sumCentroid.z / meshlet.triangleContained.size() };

		auto	 scoreFunc = [&](int triId, float cweight) {
			auto& tri = ctx.triangles[triId];
			auto  spread = clusterNormal.x * tri.normal.x + clusterNormal.y * tri.normal.y + clusterNormal.z * tri.normal.z;
			auto  diffx = clusterCentroid.x - tri.centroid.x;
			auto  diffy = clusterCentroid.y - tri.centroid.y;
			auto  diffz = clusterCentroid.z - tri.centroid.z;
			auto  diff = sqrtf(diffx * diffx + diffy * diffy + diffz * diffz);

			auto  cone = std::max(1e-3f, 1.0f - spread * cweight);
			return (1 + diff * (1 - cweight)) * cone;
		};

		int ret = -1;
		if (meshlet.triangleContained.size() == 0)
		{
			for (int i = 0; i < ctx.triangleEmitted.size(); i++)
			{
				if (ctx.triangleEmitted[i] == 0)
					return i;
			}
		}
		for (int i = 0; i < meshlet.usedVertices.size(); i++)
		{
			auto	 v = meshlet.usedVertices[i];
			uint32_t bestPriority = 0xffffffff;
			float	 bestScore = 1e9;
			for (int j = 0; j < ctx.adjMaps[v].size(); j++)
			{
				if (ctx.adjMaps[v][j] == -1)
					continue;
				auto tri = ctx.triangles[ctx.adjMaps[v][j]];
				auto ua = meshlet.isvertexUsed[tri.ind.x] == 0, ub = meshlet.isvertexUsed[tri.ind.y] == 0,
					 uc = meshlet.isvertexUsed[tri.ind.z] == 0;
				auto la = ctx.remainActiveAdj[tri.ind.x], lb = ctx.remainActiveAdj[tri.ind.y],
					 lc = ctx.remainActiveAdj[tri.ind.z];
				auto prior = ua + ub + uc + 0u;
				if (prior != 0)
				{
					if (la == 1 && lb == 1 && lc == 1)
						prior = 0;
					prior++;
				}
				auto score = scoreFunc(ctx.adjMaps[v][j], 0.0f);
				if (prior > bestPriority)
					continue;
				if (prior < bestPriority || score < bestScore)
				{
					ret = ctx.adjMaps[v][j];
					bestPriority = prior;
					bestScore = score;
				}
			}
		}
		return ret;
	}

	void addTriangleToMeshlet(int triId, MbContext* ctx, MbCurrentMeshlet* meshlet)
	{
		auto tri = ctx->triangles[triId];
		ctx->triangleEmitted[triId] = 1;
		for (int i = 0; i < ctx->adjMaps[tri.ind.x].size(); i++)
		{
			if (ctx->adjMaps[tri.ind.x][i] == triId)
				ctx->adjMaps[tri.ind.x][i] = -1;
		}
		for (int i = 0; i < ctx->adjMaps[tri.ind.y].size(); i++)
		{
			if (ctx->adjMaps[tri.ind.y][i] == triId)
				ctx->adjMaps[tri.ind.y][i] = -1;
		}
		for (int i = 0; i < ctx->adjMaps[tri.ind.z].size(); i++)
		{
			if (ctx->adjMaps[tri.ind.z][i] == triId)
				ctx->adjMaps[tri.ind.z][i] = -1;
		}
		ctx->remainActiveAdj[tri.ind.x]--;
		ctx->remainActiveAdj[tri.ind.y]--;
		ctx->remainActiveAdj[tri.ind.z]--;

		if (!meshlet->isvertexUsed[tri.ind.x])
		{
			meshlet->usedVertices.push_back(tri.ind.x);
			meshlet->isvertexUsed[tri.ind.x] = 1;
		}
		if (!meshlet->isvertexUsed[tri.ind.y])
		{
			meshlet->usedVertices.push_back(tri.ind.y);
			meshlet->isvertexUsed[tri.ind.y] = 1;
		}
		if (!meshlet->isvertexUsed[tri.ind.z])
		{
			meshlet->usedVertices.push_back(tri.ind.z);
			meshlet->isvertexUsed[tri.ind.z] = 1;
		}
		meshlet->triangleContained.push_back(triId);
		meshlet->sumNormal.x += ctx->triangles[triId].normal.x;
		meshlet->sumNormal.y += ctx->triangles[triId].normal.y;
		meshlet->sumNormal.z += ctx->triangles[triId].normal.z;
		meshlet->sumCentroid.x += ctx->triangles[triId].centroid.x;
		meshlet->sumCentroid.y += ctx->triangles[triId].centroid.y;
		meshlet->sumCentroid.z += ctx->triangles[triId].centroid.z;

		auto normalLen = sqrt(meshlet->sumNormal.x * meshlet->sumNormal.x + meshlet->sumNormal.y * meshlet->sumNormal.y + meshlet->sumNormal.z * meshlet->sumNormal.z);
		meshlet->invSumNormal = 1.0f / normalLen;
	}

	void initializeCurrentMeshlet(const MbContext& ctx, MbCurrentMeshlet* meshlet)
	{
		meshlet->isvertexUsed.resize(ctx.vertices.size());
		for (int i = 0; i < ctx.vertices.size(); i++)
			meshlet->isvertexUsed[i] = 0;
	}

	void writeGenratedMeshlet(MbContext* ctx, const MbCurrentMeshlet& meshlet)
	{
		auto emitMeshlet = std::make_unique<Meshlet>();
		auto totalVerts = meshlet.usedVertices.size();
		emitMeshlet->vbufs.setVertexCount(SizeCast<int>(totalVerts));
		emitMeshlet->vbufs.setLayout({ TypeDescriptors.FLOAT4, TypeDescriptors.FLOAT4 });
		Vector4f dcolor;
		dcolor.x = 1.0f * rand() / RAND_MAX;
		dcolor.y = 1.0f * rand() / RAND_MAX;
		dcolor.z = 1.0f * rand() / RAND_MAX;
		dcolor.w = 1.0f * rand() / RAND_MAX;

		emitMeshlet->vbufs.allocateBuffer(totalVerts);
		std::unordered_map<int, int> vmap;
		std::vector<int>&			 indices = emitMeshlet->ibufs;
		for (int i = 0; i < meshlet.usedVertices.size(); i++)
		{
			vmap[meshlet.usedVertices[i]] = i;
		}
		for (int i = 0; i < meshlet.triangleContained.size(); i++)
		{
			auto& tri = ctx->triangles[meshlet.triangleContained[i]];
			indices.push_back(vmap[tri.ind.x]);
			indices.push_back(vmap[tri.ind.y]);
			indices.push_back(vmap[tri.ind.z]);
		}
		for (int i = 0; i < totalVerts; i++)
		{
			emitMeshlet->vbufs.setValue<Vector4f>(i, 0, ctx->vertices[meshlet.usedVertices[i]]);
			emitMeshlet->vbufs.setValue<Vector4f>(i, 1, dcolor);
		}
		ctx->generatedMeshlets->push_back(std::move(emitMeshlet));
	}
} // namespace
  // Ifrit::Graphics::SoftGraphics::MeshletBuilder::Impl

namespace Ifrit::Graphics::SoftGraphics::MeshletBuilder
{
	IFRIT_APIDECL void TrivialMeshletBuilder::bindVertexBuffer(const VertexBuffer& vbuf)
	{
		this->vbuffer = &vbuf;
	}
	IFRIT_APIDECL void TrivialMeshletBuilder::bindIndexBuffer(const std::vector<int>& ibuf)
	{
		this->ibuffer = &ibuf;
	}
	IFRIT_APIDECL void TrivialMeshletBuilder::buildMeshlet(int posAttrId, std::vector<std::unique_ptr<Meshlet>>& outData)
	{
		using namespace Impl;
		auto ctx = std::make_unique<MbContext>();
		initializeContext(ctx.get(), *vbuffer, *ibuffer, posAttrId);

		auto curMeshlet = MbCurrentMeshlet();
		initializeCurrentMeshlet(*ctx, &curMeshlet);
		int procTris = 0;
		while (procTris < ctx->triangles.size())
		{
			auto bestCandidate = meshletExpansion(*ctx, curMeshlet);
			auto meshletSize = curMeshlet.triangleContained.size();
			if (bestCandidate == -1 || meshletSize == ctx->maxTriangles)
			{
				// printf("Append Meshlet %d
				// %d\n",bestCandidate,curMeshlet.triangleContained.size());
				ctx->finishedMeshlets.push_back(curMeshlet);
				curMeshlet = MbCurrentMeshlet();
				initializeCurrentMeshlet(*ctx, &curMeshlet);
			}
			if (bestCandidate != -1)
			{
				addTriangleToMeshlet(bestCandidate, ctx.get(), &curMeshlet);
				procTris++;
			}
		}

		ctx->generatedMeshlets = &outData;
		for (int i = 0; i < ctx->finishedMeshlets.size(); i++)
		{
			writeGenratedMeshlet(ctx.get(), ctx->finishedMeshlets[i]);
		}
	}
	IFRIT_APIDECL void TrivialMeshletBuilder::mergeMeshlet(const std::vector<std::unique_ptr<Meshlet>>& meshlets,
		Meshlet& outData, std::vector<int>& outVertexOffset,
		std::vector<int>& outIndexOffset, bool autoIncre)
	{
		auto totalVerts = 0;
		auto totalIndices = 0;
		auto accuOffset = 0;
		outData.vbufs.setLayout({ TypeDescriptors.FLOAT4, TypeDescriptors.FLOAT4 });
		for (auto& m : meshlets)
		{
			outVertexOffset.push_back(totalVerts);
			outIndexOffset.push_back(totalIndices);
			totalVerts += m->vbufs.getVertexCount();
			totalIndices += Ifrit::Common::Utility::SizeCast<int>(m->ibufs.size());
		}
		outVertexOffset.push_back(totalVerts);
		outIndexOffset.push_back(totalIndices);

		outData.vbufs.setVertexCount(totalVerts);
		outData.vbufs.allocateBuffer(totalVerts);
		for (auto& m : meshlets)
		{
			for (int i = 0; i < m->vbufs.getVertexCount(); i++)
			{
				outData.vbufs.setValue(accuOffset + i, 0, m->vbufs.getValue<Vector4f>(i, 0));
				outData.vbufs.setValue(accuOffset + i, 1, m->vbufs.getValue<Vector4f>(i, 1));
			}
			for (int i = 0; i < m->ibufs.size(); i++)
			{
				outData.ibufs.push_back(m->ibufs[i] + accuOffset * autoIncre);
			}
			accuOffset += m->vbufs.getVertexCount();
		}
	}
} // namespace Ifrit::Graphics::SoftGraphics::MeshletBuilder