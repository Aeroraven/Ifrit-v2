
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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
#include "ifrit/meshproc/engine/tetrahedralization/MeshTetrahedralizer.h"
#include "tetgen.h"
#include "ifrit/core/logging/Logging.h"

namespace Ifrit::MeshProcLib::Tetrahedralization
{

    IFRIT_APIDECL FTetrahedralMeshData TetrahedralizeMesh(const MeshDescriptor& mesh)
    {
        // TODO : delete new allocations in the end
        u32*     indexData = reinterpret_cast<u32*>(mesh.indexData);
        tetgenio in;
        in.firstnumber    = 0; // 0-based indexing
        in.numberofpoints = mesh.vertexCount;
        in.pointlist      = new REAL[in.numberofpoints * 3];
        for (int i = 0; i < mesh.vertexCount; i++)
        {
            iAssertion(mesh.vertexStride == 12, "Vertex stride must be 12 bytes (3 floats)");
            auto vertexPtr = reinterpret_cast<Vector3f*>(mesh.vertexData + mesh.positionOffset + i * mesh.vertexStride);
            in.pointlist[i * 3 + 0] = vertexPtr->x;
            in.pointlist[i * 3 + 1] = vertexPtr->y;
            in.pointlist[i * 3 + 2] = vertexPtr->z;
        }

        in.numberoffacets = mesh.indexCount / 3;
        in.facetlist      = new tetgenio::facet[in.numberoffacets];
        for (int i = 0; i < in.numberoffacets; i++)
        {
            tetgenio::facet* f  = &in.facetlist[i];
            f->numberofpolygons = 1;
            f->polygonlist      = new tetgenio::polygon[f->numberofpolygons];
            f->numberofholes    = 0;
            f->holelist         = nullptr;

            tetgenio::polygon* p = &f->polygonlist[0];
            p->numberofvertices  = 3;
            p->vertexlist        = new int[p->numberofvertices];
            p->vertexlist[0]     = indexData[i * 3 + 0];
            p->vertexlist[1]     = indexData[i * 3 + 1];
            p->vertexlist[2]     = indexData[i * 3 + 2];
        }

        tetgenio       out;

        tetgenbehavior behavior;

        behavior.plc       = 1;
        behavior.minratio  = 1.414;
        behavior.maxvolume = 0.1;
        behavior.quiet     = 1;

        tetrahedralize(&behavior, &in, &out);

        FTetrahedralMeshData tetrahedralMeshData;
        tetrahedralMeshData.m_Vertices.resize(out.numberofpoints);
        for (int i = 0; i < out.numberofpoints; i++)
        {
            tetrahedralMeshData.m_Vertices[i] = Vector3f(static_cast<f32>(out.pointlist[i * 3 + 0]),
                static_cast<f32>(out.pointlist[i * 3 + 1]), static_cast<f32>(out.pointlist[i * 3 + 2]));
        }
        tetrahedralMeshData.m_Indices.resize(out.numberoftetrahedra * 4);
        for (int i = 0; i < out.numberoftetrahedra; i++)
        {
            tetrahedralMeshData.m_Indices[i * 4 + 0] = out.tetrahedronlist[i * 4 + 0];
            tetrahedralMeshData.m_Indices[i * 4 + 1] = out.tetrahedronlist[i * 4 + 1];
            tetrahedralMeshData.m_Indices[i * 4 + 2] = out.tetrahedronlist[i * 4 + 2];
            tetrahedralMeshData.m_Indices[i * 4 + 3] = out.tetrahedronlist[i * 4 + 3];
        }
        iDebug("Tetrahedralization complete: {} vertices, {} tetrahedra", out.numberofpoints, out.numberoftetrahedra);

        return tetrahedralMeshData;
    }

} // namespace Ifrit::MeshProcLib::Tetrahedralization