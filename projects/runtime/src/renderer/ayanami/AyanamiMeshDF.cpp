
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

#include "ifrit/imaging/compress/CompressedTextureUtil.h"

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
            using namespace Ifrit::Imaging::Compress;

            MeshDescriptor meshDesc;
            meshDesc.indexCount     = SizeCast<int>(meshData->m_indices.size());
            meshDesc.indexData      = reinterpret_cast<i8*>(meshData->m_indices.data());
            meshDesc.positionOffset = 0;
            meshDesc.vertexCount    = SizeCast<int>(meshData->m_vertices.size());
            meshDesc.vertexData     = reinterpret_cast<i8*>(meshData->m_vertices.data());
            meshDesc.vertexStride   = sizeof(Vector3f);
            meshDesc.normalData     = reinterpret_cast<i8*>(meshData->m_normals.data());
            meshDesc.normalStride   = sizeof(Vector3f);

            auto serialMeshDFName  = "core.ayanami.meshdf_2." + meshData->identifier + ".cache";
            bool hasCachedDF       = false;
            bool shouldGenCachedDF = false;
            auto cachePathStr      = String(cachePath);

            auto serialCompactMeshDFName = "core.ayanami.meshdf_2_u8." + meshData->identifier + ".cache";
            auto hasCachedCompactDF      = false;
            bool shouldGenCompactDF      = false;
            auto cacheCompactPathStr     = String(cachePathStr + serialCompactMeshDFName);

            auto bc4CompactMeshDFName   = "core.ayanami.meshdf_2_bc4." + meshData->identifier + ".cache";
            auto hasCachedBC4CompactDF  = false;
            bool shouldGenBC4CompactDF  = false;
            auto cacheBC4CompactPathStr = String(cachePathStr + bc4CompactMeshDFName);

            if (!std::filesystem::exists(cacheBC4CompactPathStr))
            {
                shouldGenBC4CompactDF = true;
            }
            else
            {
                hasCachedBC4CompactDF = std::filesystem::exists(cacheBC4CompactPathStr);
                if (hasCachedBC4CompactDF)
                {
                    shouldGenBC4CompactDF = false;
                }
                else
                {
                    shouldGenBC4CompactDF = true;
                }
            }
            if (!std::filesystem::exists(cacheCompactPathStr))
            {
                shouldGenCompactDF = true;
            }
            else
            {
                hasCachedCompactDF = std::filesystem::exists(cacheCompactPathStr);
                if (hasCachedCompactDF)
                {
                    shouldGenCompactDF = false;
                }
                else
                {
                    shouldGenCompactDF = true;
                }
            }
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

            SignedDistanceField        sdf;
            CompactSignedDistanceField compactSdf;
            RSizedBuffer               bc4CompressedSdf;

            if (hasCachedCompactDF)
            {
                auto serialCompactMeshDFPath = cacheCompactPathStr;
                auto buffer                  = ReadBinaryFile(serialCompactMeshDFPath);
                Ifrit::Common::Serialization::DeserializeBinary(buffer, compactSdf);
            }
            else if (hasCachedDF)
            {
                auto serialMeshDFPath = cachePathStr + serialMeshDFName;
                auto buffer           = ReadBinaryFile(serialMeshDFPath);
                Ifrit::Common::Serialization::DeserializeBinary(buffer, sdf);
            }
            else
            {
                iInfo("Building mesh distance field for {}", meshData->identifier);

                ConvertMeshToSDF(
                    meshDesc, sdf, ~0u, ~0u, ~0u, MeshProcLib::MeshSDFProcess::SDFGenerateMethod::RayTracing, false);

                auto serialMeshDFPath = cachePathStr + serialMeshDFName;
                if (shouldGenCachedDF)
                {
                    String buffer;
                    Ifrit::Common::Serialization::SerializeBinary(sdf, buffer);
                    WriteBinaryFile(serialMeshDFPath, buffer);
                }
            }
            if (shouldGenCompactDF)
            {
                iInfo("Building compact mesh distance field for {}", meshData->identifier);
                CompactSDF(sdf, compactSdf);
                auto   serialCompactMeshDFPath = cacheCompactPathStr;
                String buffer;
                Ifrit::Common::Serialization::SerializeBinary(compactSdf, buffer);
                WriteBinaryFile(serialCompactMeshDFPath, buffer);
            }
            if (shouldGenBC4CompactDF)
            {
                // TODO:
                iInfo("Building BC4 compact mesh distance field for {}", meshData->identifier);
                auto         serialCompactMeshDFPath = cacheBC4CompactPathStr;
                RSizedBuffer bufferIn(compactSdf.sdfData);
                WriteTex2DToBlockCompressedFile(bufferIn, cacheBC4CompactPathStr, TextureFormat::R8_UNORM,
                    compactSdf.width, compactSdf.height, compactSdf.depth, CompressionAlgo::BC4);
            }

            // Read Bc4 compressed sdf
            u32 uWidth, uHeight, uDepth;
            ReadBlockCompressedTex2DFromFile(bc4CompressedSdf, cacheBC4CompactPathStr, uWidth, uHeight, uDepth);

            m_CompactSDFData = bc4CompressedSdf.ToByteVector<u8>(); //

            // m_CompactSDFData = std::move(compactSdf.sdfData);
            m_sdWidth  = uWidth;
            m_sdHeight = uHeight;
            m_sdDepth  = uDepth;
            m_sdBoxMin = Vector3f(compactSdf.bboxMin);
            m_sdBoxMax = Vector3f(compactSdf.bboxMax);
            m_isBuilt  = true;
            m_SdfMin   = compactSdf.m_SdfMin;
            m_SdfMax   = compactSdf.m_SdfMax;

            auto bboxSize = m_sdBoxMax - m_sdBoxMin;
            // iInfo("Mbox Extent: {} {} {}", bboxSize.x, bboxSize.y, bboxSize.z);

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
            auto volumeSize   = m_CompactSDFData.size();
            auto deviceVolume = rhi->CreateBuffer("Ayanami_DFVolume", volumeSize,
                RhiBufferUsage::RhiBufferUsage_CopyDst | RhiBufferUsage::RhiBufferUsage_CopySrc, true, false);
            deviceVolume->MapMemory();
            deviceVolume->WriteBuffer(m_CompactSDFData.data(), volumeSize, 0);
            deviceVolume->FlushBuffer();
            deviceVolume->UnmapMemory();

            m_gpuResource->sdfTexture = rhi->CreateTexture3D("Ayanami_DFTexture", m_sdWidth, m_sdHeight, m_sdDepth,
                RhiImageFormat::RhiImgFmt_BC4_UNORM_BLOCK,
                RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_CopyDst, false);
            m_gpuResource->sdfTextureBindId =
                rhi->RegisterCombinedImageSampler(m_gpuResource->sdfTexture.get(), linearClampSampler.get());
            m_gpuResource->sdfMetaBuffer = rhi->CreateBuffer("Ayanami_DFMeta", sizeof(AyanamiMeshDFResource::SDFMeta),
                RhiBufferUsage::RhiBufferUsage_CopyDst | RhiBufferUsage::RhiBufferUsage_SSBO, true, true);

            auto stagedMetaBuffer = rhi->CreateStagedSingleBuffer(m_gpuResource->sdfMetaBuffer.get());
            AyanamiMeshDFResource::SDFMeta sdfMeta;

            sdfMeta.bboxMin      = Vector4f(m_sdBoxMin.x, m_sdBoxMin.y, m_sdBoxMin.z, m_SdfMin);
            sdfMeta.bboxMax      = Vector4f(m_sdBoxMax.x, m_sdBoxMax.y, m_sdBoxMax.z, m_SdfMax);
            sdfMeta.sdfId        = m_gpuResource->sdfTextureBindId->GetActiveId();
            sdfMeta.m_IsTwoSided = m_IsDoubleSided ? 1 : 0;

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