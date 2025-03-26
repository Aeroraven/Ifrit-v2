
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

#pragma once
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/math/VectorDefs.h"
#include "ifrit/core/base/Material.h"
#include "ifrit/core/base/Mesh.h"
#include "ifrit/core/base/Object.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <unordered_map>
#include <vector>

namespace Ifrit::Core
{

    // The design is inappropriate for the current project.
    // It's renderer-specific and should be moved to the renderer.

    struct PerFramePerViewData
    {
        Matrix4x4f m_worldToView;
        Matrix4x4f m_perspective;
        Matrix4x4f m_worldToClip;
        Matrix4x4f m_inversePerspective;
        Matrix4x4f m_clipToWorld;
        Matrix4x4f m_viewToWorld;
        Vector4f   m_cameraPosition;
        Vector4f   m_cameraFront;
        f32        m_renderWidthf;
        f32        m_renderHeightf;
        f32        m_cameraNear;
        f32        m_cameraFar;
        f32        m_cameraFovX;
        f32        m_cameraFovY;
        f32        m_cameraAspect;
        f32        m_cameraOrthoSize;
        f32        m_hizLods;
        f32        m_viewCameraType; // 0: perspective, 1: ortho

        f32        m_cullCamOrthoSizeX;
        f32        m_cullCamOrthoSizeY;
    };

    struct PerObjectData
    {
        u32 transformRef     = 0;
        u32 objectDataRef    = 0;
        u32 instanceDataRef  = 0;
        u32 transformRefLast = 0;
        u32 materialId       = 0;
    };

    struct PerShaderEffectData
    {
        Vec<Material*>                           m_materials;
        Vec<Mesh*>                               m_meshes;
        Vec<Transform*>                          m_transforms;
        Vec<MeshInstance*>                       m_instances;

        // Data to GPUs
        u32                                      m_lastObjectCount = ~0u;
        Vec<PerObjectData>                       m_objectData;
        Ref<Graphics::Rhi::RhiMultiBuffer>       m_batchedObjectData = nullptr;
        Graphics::Rhi::RhiBindlessDescriptorRef* m_batchedObjBufRef  = nullptr;
    };

    struct PerFrameRenderTargets
    {
        Graphics::Rhi::RhiTextureRef                  m_colorRT;
        Ref<Graphics::Rhi::RhiDescHandleLegacy>       m_colorRTIdSRV;
        Graphics::Rhi::RhiTexture*                    m_depthRT;

        Ref<Graphics::Rhi::RhiColorAttachment>        m_colorRTRef;
        Ref<Graphics::Rhi::RhiDepthStencilAttachment> m_depthRTRef;
        Ref<Graphics::Rhi::RhiRenderTargets>          m_rts;
        u32                                           m_width = 0, m_height = 0;
    };

    struct ShadowMappingData
    {
        using GPUBuffer        = Graphics::Rhi::RhiBuffer;
        using GPUUniformBuffer = Graphics::Rhi::RhiMultiBuffer;
        using GPUBindId        = Graphics::Rhi::RhiDescHandleLegacy;

        struct SingleShadowView
        {
            Array<u32, 4> m_viewRef;
            Array<u32, 4> m_texRef;

            Array<u32, 4> m_viewMapping;
            Array<f32, 4> m_csmStart;
            Array<f32, 4> m_csmEnd;
            u32           m_csmSplits;
        };
        Vec<SingleShadowView> m_shadowViews;
        u32                   m_enabledShadowMaps = 0;

        Ref<GPUUniformBuffer> m_allShadowData = nullptr;
        Ref<GPUBindId>        m_allShadowDataId;
    };

    struct PerFrameData
    {
        using GPUUniformBuffer = Graphics::Rhi::RhiMultiBuffer;
        using GPUBuffer        = Graphics::Rhi::RhiBufferRef;
        using GPUBindlessRef   = Graphics::Rhi::RhiBindlessDescriptorRef;
        using GPUBindId        = Graphics::Rhi::RhiDescHandleLegacy;
        using GPUTexture       = Graphics::Rhi::RhiTextureRef;
        using GPUColorRT       = Graphics::Rhi::RhiColorAttachment;
        using GPUDepthRT       = Graphics::Rhi::RhiDepthStencilAttachment;
        using GPURTs           = Graphics::Rhi::RhiRenderTargets;
        using GPUSampler       = Graphics::Rhi::RhiSampler;
        using GPUBarrier       = Graphics::Rhi::RhiResourceBarrier;

        enum class ViewType
        {
            Invisible,
            Primary,
            Display,
            Shadow
        };

        struct GBufferDesc
        {
            u32 m_albedo_materialFlags;
            u32 m_specular_occlusion;
            u32 m_normal_smoothness;
            u32 m_emissive;
            u32 m_shadowMask;
        };

        struct GBuffer
        {
            GPUTexture      m_albedo_materialFlags;
            GPUTexture      m_specular_occlusion;
            GPUTexture      m_specular_occlusion_intermediate;
            GPUTexture      m_normal_smoothness;
            GPUTexture      m_emissive;
            GPUTexture      m_shadowMask;

            u32             m_rtWidth   = 0;
            u32             m_rtHeight  = 0;
            u32             m_rtCreated = 0;

            Ref<GPUBindId>  m_albedo_materialFlags_sampId;
            Ref<GPUBindId>  m_specular_occlusion_sampId;
            Ref<GPUBindId>  m_specular_occlusion_intermediate_sampId;
            Ref<GPUBindId>  m_normal_smoothness_sampId;

            Ref<GPUColorRT> m_specular_occlusion_colorRT;
            Ref<GPURTs>     m_specular_occlusion_RTs;

            Ref<GPUBindId>  m_depth_sampId;

            GPUBarrier      m_normal_smoothnessBarrier;
            GPUBarrier      m_specular_occlusionBarrier;

            GPUBuffer       m_gbufferRefs = nullptr;
            GPUBindlessRef* m_gbufferDesc = nullptr;

            Vec<GPUBarrier> m_gbufferBarrier;
        };

        struct SinglePassHiZData
        {
            GPUTexture      m_hizTexture = nullptr;
            Vec<u32>        m_hizRefs;
            GPUBuffer       m_hizRefBuffer = nullptr;
            GPUBuffer       m_hizAtomics   = nullptr;
            GPUBindlessRef* m_hizDesc      = nullptr;
            u32             m_hizIters     = 0;
            u32             m_hizWidth     = 0;
            u32             m_hizHeight    = 0;
        };

        struct PerViewData
        {
            ViewType              m_viewType = ViewType::Invisible;

            PerFramePerViewData   m_viewData;
            PerFramePerViewData   m_viewDataOld;
            Ref<GPUUniformBuffer> m_viewBuffer      = nullptr;
            Ref<GPUUniformBuffer> m_viewBufferLast  = nullptr;
            GPUBindlessRef*       m_viewBindlessRef = nullptr;
            Ref<GPUBindId>        m_viewBufferId    = nullptr;

            // Non-gpu data
            u32                   m_renderWidth;
            u32                   m_renderHeight;
            bool                  m_camMoved;

            // visibility buffer
            GPUTexture            m_visibilityBuffer_HW = nullptr;
            GPUTexture            m_visPassDepth_HW     = nullptr;

            Ref<GPUColorRT>       m_visColorRT_HW    = nullptr;
            Ref<GPUDepthRT>       m_visDepthRT_HW    = nullptr;
            Ref<GPURTs>           m_visRTs_HW        = nullptr;
            Ref<GPUBindId>        m_visDepthIdSRV_HW = nullptr;

            // visibility buffer software. It's compute shader, so
            // not repeated decl required
            GPUTexture            m_visibilityBuffer_SW    = nullptr;
            GPUBuffer             m_visPassDepth_SW        = nullptr;
            GPUBuffer             m_visPassDepthCASLock_SW = nullptr;

            // combined visibility buffer is required
            GPUTexture            m_visibilityBuffer_Combined = nullptr;
            GPUTexture            m_visibilityDepth_Combined  = nullptr;

            Ref<GPUBindId>        m_visibilityBufferIdSRV_Combined = nullptr;
            Ref<GPUBindId>        m_visibilityDepthIdSRV_Combined  = nullptr;

            // visibility buffer for 2nd pass, reference to the same texture, but
            // without clearing
            Ref<GPUColorRT>       m_visColorRT2_HW = nullptr;
            Ref<GPUDepthRT>       m_visDepthRT2_HW = nullptr;
            Ref<GPURTs>           m_visRTs2_HW     = nullptr;

            // all visible clusters
            GPUBuffer             m_allFilteredMeshletsAllCount = nullptr;

            GPUBuffer             m_allFilteredMeshletsHW = nullptr;
            GPUBuffer             m_allFilteredMeshletsSW = nullptr;

            u32                   m_allFilteredMeshlets_SWOffset = 0;

            u32                   m_allFilteredMeshletsMaxCount = 0;
            u32                   m_requireMaxFilteredMeshlets  = 0;
            GPUBindlessRef*       m_allFilteredMeshletsDesc     = nullptr;

            // SPD HiZ
            SinglePassHiZData     m_spHiZData;
            SinglePassHiZData     m_spHiZDataMin;

            // Instance culling
            GPUBuffer             m_instCullDiscardObj          = nullptr;
            GPUBuffer             m_instCullPassedObj           = nullptr;
            GPUBuffer             m_persistCullIndirectDispatch = nullptr;
            GPUBindlessRef*       m_instCullDesc                = nullptr;
            u32                   m_maxSupportedInstances       = 0;

            // Inst-Persist barrier
            Vec<GPUBarrier>       m_persistCullBarrier;
            Vec<GPUBarrier>       m_visibilityBarrier;
        };

        struct FSR2ExtraData
        {
            GPUTexture     m_fsr2Output      = nullptr;
            Ref<GPUBindId> m_fsr2OutputSRVId = nullptr;
            u32            m_fsrFrameId      = 0;
        };

        IF_CONSTEXPR static Graphics::Rhi::RhiImageFormat c_visibilityFormat =
            Graphics::Rhi::RhiImageFormat::RhiImgFmt_R32_UINT;

        HashSet<u32>                                       m_enabledEffects;
        Vec<PerShaderEffectData>                           m_shaderEffectData;
        CustomHashMap<ShaderEffect, u32, ShaderEffectHash> m_shaderEffectMap;
        PerShaderEffectData                                m_allInstanceData;

        // Per view data. Here for simplicity, assume 0 is the primary view
        // Other views are for shadow maps, etc.
        Vec<PerViewData>                                   m_views;

        // GBuffer
        GBuffer                                            m_gbuffer;
        GPUBindlessRef*                                    m_gbufferDepthDesc;

        // Gbuffer desc
        GPUBindlessRef*                                    m_gbufferDescFrag = nullptr;

        // Emit depth targets
        GPUTexture                                         m_velocityMaterial     = nullptr;
        GPUTexture                                         m_motionVector         = nullptr;
        GPUBindlessRef*                                    m_velocityMaterialDesc = nullptr;

        // Material classify
        GPUBuffer                                          m_matClassCountBuffer            = nullptr;
        GPUBuffer                                          m_matClassIndirectDispatchBuffer = nullptr;
        GPUBuffer                                          m_matClassFinalBuffer            = nullptr;
        GPUBuffer                                          m_matClassPixelOffsetBuffer      = nullptr;
        GPUBindlessRef*                                    m_matClassDesc                   = nullptr;
        Vec<GPUBarrier>                                    m_matClassBarrier;
        u32                                                m_matClassSupportedNumMaterials = 0;
        u32                                                m_matClassSupportedNumPixels    = 0;

        // For history
        u32                                                m_frameId           = 0;
        f32                                                m_frameTimestamp[2] = { 0.0f, 0.0f };

        // TAA
        Vec<PerFrameRenderTargets>                         m_taaHistory;
        GPUTexture                                         m_taaUnresolved  = nullptr;
        GPUBindlessRef*                                    m_taaHistoryDesc = nullptr;
        f32                                                m_taaJitterX     = 0.0f;
        f32                                                m_taaJitterY     = 0.0f;

        // FSR2
        FSR2ExtraData                                      m_fsr2Data;

        // Atmosphere
        Ref<void>                                          m_atmosphereData = nullptr;
        GPUTexture                                         m_atmoOutput;
        Vector4f                                           m_sunDir;

        // Shadow mapping
        ShadowMappingData                                  m_shadowData2;
        GPUTexture                                         m_deferShadowMask = nullptr;
        Ref<GPUColorRT>                                    m_deferShadowMaskRT;
        Ref<GPURTs>                                        m_deferShadowMaskRTs;

        Ref<GPUBindId>                                     m_deferShadowMaskId;
    };

} // namespace Ifrit::Core