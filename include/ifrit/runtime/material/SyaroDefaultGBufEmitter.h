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
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/forwarding/FwdBase.h"
#include "ifrit/runtime/base/Material.h"
#include "ifrit/runtime/asset/MaterialAsset.h"
namespace Ifrit::Runtime
{

    struct SyaroDefaultGBufEmitterData
    {
        u32 m_albedoId;
        u32 m_normalMapId;
    };

    class IFRIT_APIDECL SyaroDefaultGBufEmitter : public Material
    {
    private:
        SyaroDefaultGBufEmitterData m_materialData;
        static RHI::RhiShader*      m_shader;
        static ShaderEffect         m_shaderEffect;

    public:
        SyaroDefaultGBufEmitter(IApplication* app);
        ~SyaroDefaultGBufEmitter() = default;

        void        BuildMaterial();

        inline void SetAlbedoId(u32 id) { m_materialData.m_albedoId = id; }
        inline void SetNormalMapId(u32 id) { m_materialData.m_normalMapId = id; }
        inline u32  GetAlbedoId() const { return m_materialData.m_albedoId; }
        inline u32  GetNormalMapId() const { return m_materialData.m_normalMapId; }
    };

    using DefaultMaterial = SyaroDefaultGBufEmitter;

    class IFRIT_RUNTIME_API IF_CLASS() DefaultMaterialAsset : public MaterialAsset
    {
    public:
        using MaterialAsset::MaterialAsset;
        virtual Material* GetMaterial() override;

    private:
        Owner<SyaroDefaultGBufEmitter> mMaterial;
    };
} // namespace Ifrit::Runtime