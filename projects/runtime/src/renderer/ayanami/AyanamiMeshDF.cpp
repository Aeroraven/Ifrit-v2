
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

#include "ifrit/runtime/renderer/ayanami/AyanamiMeshDF.h"
#include "ifrit/runtime/base/Mesh.h"

#define IFRIT_MESHPROC_IMPORT
#include "ifrit/meshproc/engine/base/MeshDesc.h"
#include "ifrit/meshproc/engine/meshsdf/MeshSDFConverter.h"
#undef IFRIT_MESHPROC_IMPORT

#include "ifrit/core/math/simd/SimdVectors.h"
#include "ifrit/core/math/VectorOps.h"
#include "ifrit/core/file/FileOps.h"
#include "ifrit/core/typing/Util.h"
#include <filesystem>

using namespace Ifrit::Math;

namespace Ifrit::Runtime::Ayanami
{

    IF_CONSTEXPR u32   cAyanamiMeshDFWidth = 64;

    IFRIT_APIDECL void AyanamiMeshDF::BuildMeshDF(const std::string_view& cachePath)
    {
        auto meshFilter = this->GetParentUnsafe()->GetComponentUnsafe<MeshFilter>();
        if (meshFilter == nullptr)
        {
            iError("AyanamiMeshDF::BuildMeshDF() requires mesh to be attached to a object");
            std::abort();
        }
        auto meshContainer = meshFilter->GetMesh();
        auto meshData      = meshContainer->LoadMesh();

        {
            using namespace Ifrit::MeshProcLib::MeshSDFProcess;
            using namespace Ifrit::MeshProcLib;
            using namespace Ifrit;
            MeshDescriptor meshDesc;
            meshDesc.indexCount     = SizeCast<int>(meshData->m_indices.size());
            meshDesc.indexData      = reinterpret_cast<i8*>(meshData->m_indices.data());
            meshDesc.positionOffset = 0;
            meshDesc.vertexCount    = SizeCast<int>(meshData->m_vertices.size());
            meshDesc.vertexData     = reinterpret_cast<i8*>(meshData->m_vertices.data());
            meshDesc.vertexStride   = sizeof(Vector3f);
            meshDesc.normalData     = reinterpret_cast<i8*>(meshData->m_normals.data());
            meshDesc.normalStride   = sizeof(Vector3f);

            auto serialMeshDFName  = "core.ayanami.meshdf_1." + meshData->identifier + ".cache";
            bool hasCachedDF       = false;
            bool shouldGenCachedDF = false;
            auto cachePathStr      = String(cachePath);

            if (!std::filesystem::exists(cachePathStr))
            {
                shouldGenCachedDF = true;
            }
            else
            {
                auto serialMeshDFPath = cachePathStr + serialMeshDFName;
                hasCachedDF           = std::filesystem::exists(serialMeshDFPath);
                // hasCachedDF           = false;
                if (hasCachedDF)
                {
                    shouldGenCachedDF = false;
                }
                else
                {
                    shouldGenCachedDF = true;
                }
            }

            SignedDistanceField sdf;
            if (hasCachedDF)
            {
                auto serialMeshDFPath = cachePathStr + serialMeshDFName;
                auto buffer           = ReadBinaryFile(serialMeshDFPath);
                Ifrit::Common::Serialization::DeserializeBinary(buffer, sdf);
            }
            else
            {
                iInfo("Building mesh distance field for {}", meshData->identifier);
                ConvertMeshToSDF(meshDesc, sdf, cAyanamiMeshDFWidth, cAyanamiMeshDFWidth, cAyanamiMeshDFWidth,
                    MeshProcLib::MeshSDFProcess::SDFGenerateMethod::RayTracing, false);

                auto serialMeshDFPath = cachePathStr + serialMeshDFName;
                if (shouldGenCachedDF)
                {
                    String buffer;
                    Ifrit::Common::Serialization::SerializeBinary(sdf, buffer);
                    WriteBinaryFile(serialMeshDFPath, buffer);
                }
            }
            m_sdfData  = std::move(sdf.sdfData);
            m_sdWidth  = sdf.width;
            m_sdHeight = sdf.height;
            m_sdDepth  = sdf.depth;
            m_sdBoxMin = Vector3f(sdf.bboxMin.x, sdf.bboxMin.y, sdf.bboxMin.z);
            m_sdBoxMax = Vector3f(sdf.bboxMax.x, sdf.bboxMax.y, sdf.bboxMax.z);
            m_isBuilt  = true;

            if (Any(Abs(m_sdBoxMax - m_sdBoxMin) < 1e-1f))
            {
                iWarn("Mesh SDF BBox is too small, please check the mesh data.");
            }
        }
    }

    IFRIT_APIDECL void AyanamiMeshDF::BuildGPUResource(Graphics::Rhi::RhiBackend* rhi, SharedRenderResource* sharedRes)
    {
        auto linearClampSampler = sharedRes->GetLinearClampSampler();
        if (m_gpuResource == nullptr)
        {
            if (m_isBuilt == false)
            {
                iError("AyanamiMeshDF::BuildGPUResource() requires mesh to be built first");
                std::abort();
            }
            m_gpuResource = std::make_unique<AyanamiMeshDFResource>();
            using namespace Ifrit::Graphics::Rhi;
            auto volumeSize   = m_sdWidth * m_sdHeight * m_sdDepth;
            auto deviceVolume = rhi->CreateBuffer("Ayanami_DFVolume", volumeSize * sizeof(f32),
                RhiBufferUsage::RhiBufferUsage_CopyDst | RhiBufferUsage::RhiBufferUsage_CopySrc, true, false);
            deviceVolume->MapMemory();
            deviceVolume->WriteBuffer(m_sdfData.data(), volumeSize * sizeof(f32), 0);
            deviceVolume->FlushBuffer();
            deviceVolume->UnmapMemory();

            m_gpuResource->sdfTexture = rhi->CreateTexture3D("Ayanami_DFTexture", m_sdWidth, m_sdHeight, m_sdDepth,
                RhiImageFormat::RhiImgFmt_R32_SFLOAT,
                RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_CopyDst
                    | RhiImageUsage::RhiImgUsage_UnorderedAccess,
                true);
            m_gpuResource->sdfTextureBindId =
                rhi->RegisterCombinedImageSampler(m_gpuResource->sdfTexture.get(), linearClampSampler.get());
            m_gpuResource->sdfMetaBuffer = rhi->CreateBuffer("Ayanami_DFMeta", sizeof(AyanamiMeshDFResource::SDFMeta),
                RhiBufferUsage::RhiBufferUsage_CopyDst | RhiBufferUsage::RhiBufferUsage_SSBO, true, true);

            auto stagedMetaBuffer = rhi->CreateStagedSingleBuffer(m_gpuResource->sdfMetaBuffer.get());
            AyanamiMeshDFResource::SDFMeta sdfMeta;
            sdfMeta.bboxMin = Vector4f(m_sdBoxMin.x, m_sdBoxMin.y, m_sdBoxMin.z, 0);
            sdfMeta.bboxMax = Vector4f(m_sdBoxMax.x, m_sdBoxMax.y, m_sdBoxMax.z, 0);
            sdfMeta.width   = m_sdWidth;
            sdfMeta.height  = m_sdHeight;
            sdfMeta.depth   = m_sdDepth;
            sdfMeta.sdfId   = m_gpuResource->sdfTextureBindId->GetActiveId();

            auto tq = rhi->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
            tq->RunSyncCommand([&](const RhiCommandList* cmd) {
                RhiResourceBarrier barrier;
                barrier.m_type                     = RhiBarrierType::Transition;
                barrier.m_transition.m_type        = RhiResourceType::Texture;
                barrier.m_transition.m_texture     = m_gpuResource->sdfTexture.get();
                barrier.m_transition.m_srcState    = RhiResourceState::AutoTraced;
                barrier.m_transition.m_dstState    = RhiResourceState::CopyDst;
                barrier.m_transition.m_subResource = { 0, 0, 1, 1 };

                cmd->AddResourceBarrier({ barrier });
                cmd->CopyBufferToImage(deviceVolume.get(), m_gpuResource->sdfTexture.get(), { 0, 0, 1, 1 });
                stagedMetaBuffer->CmdCopyToDevice(cmd, &sdfMeta, sizeof(AyanamiMeshDFResource::SDFMeta), 0);
            });
        }
    }

} // namespace Ifrit::Runtime::Ayanami