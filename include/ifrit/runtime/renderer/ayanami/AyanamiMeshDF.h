
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
#include "ifrit/runtime/base/Component.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/renderer/SharedRenderResource.h"

namespace Ifrit::Runtime::Ayanami
{

    struct AyanamiMeshDFResource
    {
        struct SDFMeta
        {
            Vector4f bboxMin;
            Vector4f bboxMax;
            u32      sdfId;
            u32      m_IsTwoSided;
        };
        using GPUTexture = Graphics::Rhi::RhiTextureRef;
        using GPUBuffer  = Graphics::Rhi::RhiBufferRef;
        using SRVDesc    = Graphics::Rhi::RhiSRVDesc;

        GPUTexture sdfTexture;
        SRVDesc    sdfTextureBindId;
        GPUBuffer  sdfMetaBuffer;

        // GPUSampler     sdfSampler; // this design is not a good idea, should be removed in the future
        // Yes, it's removed now
    };

    // AyanamiMeshDF stores mesh-level signed distance field data
    // This is used for mesh-based raymarching
    class IFRIT_APIDECL AyanamiMeshDF : public Component
    {
    private:
        Vec<u8>                      m_CompactSDFData;
        u32                          m_sdWidth;
        u32                          m_sdHeight;
        u32                          m_sdDepth;
        Vector3f                     m_sdBoxMin;
        Vector3f                     m_sdBoxMax;
        bool                         m_isBuilt        = false;
        bool                         m_IsDoubleSided  = false;
        f32                          m_SdfMin         = 0.0f;
        f32                          m_SdfMax         = 0.0f;
        bool                         m_UseCompression = false;

        Owner<AyanamiMeshDFResource> m_gpuResource = nullptr;

    public:
        AyanamiMeshDF() {}
        AyanamiMeshDF(std::shared_ptr<GameObject> owner) : Component(owner) {}
        virtual ~AyanamiMeshDF() = default;

        inline std::string Serialize() override { return ""; }
        inline void        Deserialize() override {}

    public:
        void            BuildMeshDF(const std::string_view& cachePath);
        void            BuildGPUResource(Graphics::Rhi::RhiBackend* rhi, SharedRenderResource* sharedRes);
        inline u32      GetMetaBufferId() const { return m_gpuResource->sdfMetaBuffer->GetDescId(); }
        inline Vector3f GetBoxMin() const { return m_sdBoxMin; }
        inline Vector3f GetBoxMax() const { return m_sdBoxMax; }
        inline void     SetDoubleSided(bool isDoubleSided) { m_IsDoubleSided = isDoubleSided; }
        inline bool     IsDoubleSided() const { return m_IsDoubleSided; }
        inline void     SetCompression(bool useCompression) { m_UseCompression = useCompression; }
        IFRIT_COMPONENT_SERIALIZE(m_sdWidth, m_sdHeight, m_sdDepth, m_sdBoxMin, m_sdBoxMax);
    };

} // namespace Ifrit::Runtime::Ayanami

IFRIT_COMPONENT_REGISTER(Ifrit::Runtime::Ayanami::AyanamiMeshDF)
