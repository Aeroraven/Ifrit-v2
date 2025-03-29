
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

#include "ifrit/core/renderer/RendererBase.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/util/Parallel.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/base/Light.h"
#include "ifrit/rhi/common/RhiLayer.h"

#include "ifrit/core/renderer/util/NoiseUtils.h"
#include "ifrit/core/renderer/util/RenderingUtils.h"

#include <mutex>

using Ifrit::Common::Utility::SizeCast;
namespace Ifrit::Core
{

    // ImmutableRendererResources RendererBase::m_immRes = {};

    IFRIT_APIDECL void RendererBase::PrepareImmutableResources()
    {
        if (m_immRes.m_initialized)
        {
            return;
        }
        m_immRes.m_initialized = true;
        std::lock_guard<std::mutex> lock(m_immRes.m_mutex);
        auto                        rhi = m_app->GetRhi();
        if (m_immRes.m_linearSampler == nullptr)
        {
            m_immRes.m_linearSampler  = rhi->CreateTrivialSampler();
            m_immRes.m_nearestSampler = rhi->CreateTrivialNearestSampler(false);
        }

        // Load blue noise texture
        if (m_immRes.m_blueNoise == nullptr)
        {
            m_immRes.m_blueNoise = RenderingUtil::loadBlueNoise(rhi);
            m_immRes.m_blueNoiseSRV =
                rhi->RegisterCombinedImageSampler(m_immRes.m_blueNoise.get(), m_immRes.m_linearSampler.get());
        }
    }

    IFRIT_APIDECL void RendererBase::CollectPerframeData(PerFrameData& perframeData, Scene* scene, Camera* camera,
        GraphicsShaderPassType passType, RenderTargets* renderTargets,
        const SceneCollectConfig& config)
    {
        using Ifrit::Common::Utility::SizeCast;

        iAssertion(m_config != nullptr, "Renderer config is not set");
        // Filling per frame data
        if (camera == nullptr)
        {
            camera = scene->GetMainCamera();
        }
        if (camera == nullptr)
        {
            throw std::runtime_error("No camera found in scene");
        }
        if (perframeData.m_views.size() == 0)
        {
            perframeData.m_views.resize(1);
        }
        if (true)
        {
            auto rtArea        = renderTargets->GetRenderArea();
            u32  actualRtWidth = 0, actualRtHeight = 0;
            GetSupersampledRenderArea(renderTargets, &actualRtWidth, &actualRtHeight);

            // Check if camera changed
            bool  camChanged = false;

            auto& viewData = perframeData.m_views[0];
            if (true)
            {
                auto& viewData        = perframeData.m_views[0];
                auto  worldToView     = Math::Transpose(camera->GetWorldToCameraMatrix());
                auto  worldToViewLast = viewData.m_viewData.m_worldToView;
                for (auto i = 0; i < 4; i++)
                {
                    for (auto j = 0; j < 4; j++)
                    {
                        if (worldToView[i][j] != worldToViewLast[i][j])
                        {
                            camChanged = true;
                            break;
                        }
                    }
                }
            }
            viewData.m_camMoved               = camChanged;
            viewData.m_viewType               = PerFrameData::ViewType::Primary;
            viewData.m_viewDataOld            = viewData.m_viewData;
            viewData.m_viewData.m_worldToView = Math::Transpose(camera->GetWorldToCameraMatrix());
            viewData.m_viewData.m_perspective = Math::Transpose(camera->GetProjectionMatrix());
            viewData.m_viewData.m_perspective[2][0] += config.projectionTranslateX;
            viewData.m_viewData.m_perspective[2][1] += config.projectionTranslateY;
            viewData.m_viewData.m_worldToClip = Math::Transpose(Math::MatMul(
                Math::Transpose(viewData.m_viewData.m_perspective), Math::Transpose(viewData.m_viewData.m_worldToView)));
            viewData.m_viewData.m_viewToWorld =
                Math::Transpose(Math::Inverse4(Math::Transpose(viewData.m_viewData.m_worldToView)));
            viewData.m_viewData.m_cameraAspect = camera->GetAspect();
            viewData.m_viewData.m_inversePerspective =
                Math::Transpose(Ifrit::Math::Inverse4(Math::Transpose(viewData.m_viewData.m_perspective)));
            viewData.m_viewData.m_clipToWorld =
                Math::Transpose(Math::Inverse4(Math::Transpose(viewData.m_viewData.m_worldToClip)));
            auto cameraTransform = camera->GetParent()->GetComponentUnsafe<Transform>();
            if (cameraTransform == nullptr)
            {
                throw std::runtime_error("Camera has no transform");
            }
            auto pos                              = cameraTransform->GetPosition();
            viewData.m_viewData.m_cameraPosition  = Vector4f{ pos.x, pos.y, pos.z, 1.0f };
            viewData.m_viewData.m_cameraFront     = camera->GetFront();
            viewData.m_viewData.m_cameraNear      = camera->GetNear();
            viewData.m_viewData.m_cameraFar       = camera->GetFar();
            viewData.m_viewData.m_cameraFovX      = camera->GetFov();
            viewData.m_viewData.m_cameraFovY      = camera->GetFov();
            viewData.m_viewData.m_cameraOrthoSize = camera->GetOrthoSpaceSize();
            viewData.m_viewData.m_viewCameraType  = camera->GetCameraType() == CameraType::Orthographic ? 1.0f : 0.0f;
        }

        // Find lights that represent the sun
        auto sunLights = scene->FilterObjectsUnsafe([](SceneObject* obj) {
            auto light = obj->GetComponentUnsafe<Light>();
            if (!light)
            {
                return false;
            }
            // printf("Light:%p\n", light);
            return light->GetAffectPbrSky();
        });
        if (sunLights.size() > 1)
        {
            iError("Multiple sun lights found");
            throw std::runtime_error("Multiple sun lights found");
        }
        if (sunLights.size() == 1)
        {
            auto     transform = sunLights[0]->GetComponentUnsafe<Transform>();
            Vector4f front     = { 0.0f, 0.0f, 1.0f, 0.0f };
            auto     rotation  = transform->GetRotation();
            auto     rotMatrix = Math::EulerAngleToMatrix(rotation);
            front              = Math::MatMul(rotMatrix, front);
            // front.z = -front.z;
            perframeData.m_sunDir = front;
        }

        // Insert light view data, if shadow maps are enabled
        auto lightWithShadow = scene->FilterObjectsUnsafe([](SceneObject* obj) -> bool {
            auto light = obj->GetComponentUnsafe<Light>();
            if (!light)
            {
                return false;
            }
            auto shadow = light->GetShadowMap();
            return shadow;
        });

        auto shadowViewCounts = SizeCast<u32>(lightWithShadow.size()) * m_config->m_shadowConfig.m_csmCount;

        if (perframeData.m_views.size() < 1 + shadowViewCounts)
        {
            perframeData.m_views.resize(1 + shadowViewCounts);
        }
        if (perframeData.m_shadowData2.m_shadowViews.size() < m_config->m_shadowConfig.k_maxShadowMaps)
        {
            perframeData.m_shadowData2.m_shadowViews.resize(m_config->m_shadowConfig.k_maxShadowMaps);
        }
        perframeData.m_shadowData2.m_enabledShadowMaps = SizeCast<u32>(lightWithShadow.size());
        for (auto di = 0, dj = 0; auto& lightObj : lightWithShadow)
        {
            // Temporarily, we assume that all lights are directional lights
            auto                 light          = lightObj->GetComponentUnsafe<Light>();
            auto                 lightTransform = lightObj->GetComponentUnsafe<Transform>();
            std::vector<float>   csmSplits(m_config->m_shadowConfig.m_csmSplits.begin(),
                  m_config->m_shadowConfig.m_csmSplits.end());
            std::vector<float>   csmBorders(m_config->m_shadowConfig.m_csmBorders.begin(),
                  m_config->m_shadowConfig.m_csmBorders.end());
            auto                 maxDist = m_config->m_shadowConfig.m_maxDistance;
            std::array<float, 4> splitStart, splitEnd;
            auto&                viewData = perframeData.m_views[0];
            auto                 csmViews = RenderingUtil::CascadeShadowMapping::fillCSMViews(
                viewData, *light, light->GetShadowMapResolution(), *lightTransform, m_config->m_shadowConfig.m_csmCount,
                maxDist, csmSplits, csmBorders, splitStart, splitEnd);

            perframeData.m_shadowData2.m_shadowViews[di].m_csmSplits = m_config->m_shadowConfig.m_csmCount;
            perframeData.m_shadowData2.m_shadowViews[di].m_csmStart  = splitStart;
            perframeData.m_shadowData2.m_shadowViews[di].m_csmEnd    = splitEnd;
            for (auto i = 0; auto& csmView : csmViews)
            {
                perframeData.m_views[1 + dj].m_viewData                       = csmView.m_viewData;
                perframeData.m_views[1 + dj].m_renderHeight                   = static_cast<u32>(csmView.m_viewData.m_renderHeightf);
                perframeData.m_views[1 + dj].m_renderWidth                    = static_cast<u32>(csmView.m_viewData.m_renderWidthf);
                perframeData.m_views[1 + dj].m_viewType                       = PerFrameData::ViewType::Shadow;
                perframeData.m_shadowData2.m_shadowViews[di].m_viewMapping[i] = dj + 1;
                dj++;
                i++;
            }
            di++;
        }
        // filling shadow data
        auto start = std::chrono::high_resolution_clock::now();
        // For each mesh renderer, get the material, mesh, and transform
        perframeData.m_enabledEffects.clear();
        for (auto& effect : perframeData.m_shaderEffectData)
        {
            effect.m_materials.clear();
            effect.m_meshes.clear();
            effect.m_transforms.clear();
            effect.m_instances.clear();
        }

        std::vector<Material*>     materials;
        std::vector<Mesh*>         meshes;
        std::vector<Transform*>    transforms;
        std::vector<MeshInstance*> instances;

        std::vector<SceneNode*>    nodes;
        nodes.push_back(scene->GetRootNode().get());
        while (!nodes.empty())
        {
            auto node = nodes.back();
            nodes.pop_back();
            for (auto& child : node->GetChildren())
            {
                nodes.push_back(child.get());
            }
            for (auto& obj : node->GetGameObjects())
            {
                auto meshRenderer = obj->GetComponentUnsafe<MeshRenderer>();
                auto meshFilter   = obj->GetComponentUnsafe<MeshFilter>();
                if (!meshRenderer || !meshFilter)
                {
                    continue;
                }
                auto transform = obj->GetComponent<Transform>();
                if (meshRenderer && meshFilter && transform)
                {
                    materials.push_back(meshRenderer->GetMaterial().get());
                    meshes.push_back(meshFilter->GetMesh().get());
                    transforms.push_back(transform.get());
                    instances.push_back(meshFilter->GetMeshInstance().get());
                }
                else
                {
                    throw std::runtime_error("MeshRenderer, MeshFilter, or Transform not found");
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        // Groups meshes with the same shader effect
        for (size_t i = 0; i < materials.size(); i++)
        {
            auto&        material  = materials[i];
            auto&        mesh      = meshes[i];
            auto&        transform = transforms[i];
            auto&        instance  = instances[i];

            ShaderEffect effect;
            effect.m_shaders = material->m_effectTemplates[passType].m_shaders;

            // TODO: Heavy copy operation, should be avoided
            effect.m_drawPasses  = material->m_effectTemplates[passType].m_drawPasses;
            effect.m_computePass = material->m_effectTemplates[passType].m_computePass;
            effect.m_type        = material->m_effectTemplates[passType].m_type;

            if (perframeData.m_shaderEffectMap.count(effect) == 0)
            {
                perframeData.m_shaderEffectMap[effect] = SizeCast<u32>(perframeData.m_shaderEffectData.size());
                perframeData.m_shaderEffectData.push_back(PerShaderEffectData{});
            }
            auto  id               = perframeData.m_shaderEffectMap[effect];
            auto& shaderEffectData = perframeData.m_shaderEffectData[id];
            perframeData.m_enabledEffects.insert(id);

            shaderEffectData.m_materials.push_back(material);
            shaderEffectData.m_meshes.push_back(mesh);
            shaderEffectData.m_transforms.push_back(transform);
            shaderEffectData.m_instances.push_back(instance);
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // iDebug("CPU time, Collecting per frame data, shader effects: {} ms",
        //       elapsed.count() / 1000.0f);
        return;
    }
    IFRIT_APIDECL void RendererBase::BuildPipelines(PerFrameData& perframeData, GraphicsShaderPassType passType,
        RenderTargets* renderTargets)
    {
        using namespace Ifrit::Graphics::Rhi;
        for (auto& shaderEffectId : perframeData.m_enabledEffects)
        {
            auto& shaderEffect = perframeData.m_shaderEffectData[shaderEffectId];
            auto  ref          = shaderEffect.m_materials[0]->m_effectTemplates[passType];

            if (ref.m_type == ShaderEffectType::Compute)
            {
                if (ref.m_computePass)
                {
                    continue;
                }
                auto rhi         = m_app->GetRhi();
                auto computePass = rhi->CreateComputePass();
                auto shader      = ref.m_shaders[0];
                computePass->SetComputeShader(shader);

                // TODO: obtain via reflection
                computePass->SetPushConstSize(64); // Test value
                computePass->SetNumBindlessDescriptorSets(3);
                for (auto& material : shaderEffect.m_materials)
                {
                    material->m_effectTemplates[passType].m_computePass = computePass;
                }
                continue;
            }
            auto                      rtFormats = renderTargets->GetFormat();
            PipelineAttachmentConfigs paConfig  = { rtFormats.m_depthFormat, rtFormats.m_colorFormats };

            if (ref.m_drawPasses.count(paConfig) > 0)
            {
                continue;
            }
            RhiShader *vertexShader = nullptr, *fragmentShader = nullptr, *taskShader = nullptr, *meshShader = nullptr;
            auto       rhi     = m_app->GetRhi();
            auto       pass    = rhi->CreateGraphicsPass();
            u32        maxSets = 0;
            for (auto& shader : ref.m_shaders)
            {
                if (shader->GetStage() == RhiShaderStage::Vertex)
                {
                    vertexShader = shader;
                    pass->SetVertexShader(shader);
                }
                else if (shader->GetStage() == RhiShaderStage::Fragment)
                {
                    fragmentShader = shader;
                    pass->SetPixelShader(shader);
                    maxSets = std::max(maxSets, shader->GetNumDescriptorSets());
                }
                else if (shader->GetStage() == RhiShaderStage::Task)
                {
                    taskShader = shader;
                }
                else if (shader->GetStage() == RhiShaderStage::Mesh)
                {
                    meshShader = shader;
                    pass->SetMeshShader(shader);
                    maxSets = std::max(maxSets, shader->GetNumDescriptorSets());
                }
            }
            if (maxSets == 0)
            {
                throw std::runtime_error("No descriptor sets found in shader");
            }
            pass->SetNumBindlessDescriptorSets(maxSets - 1);
            pass->SetRenderTargetFormat(rtFormats);
            for (auto& material : shaderEffect.m_materials)
            {
                material->m_effectTemplates[passType].m_drawPasses[paConfig] = pass;
            }
        }
    }

    IFRIT_APIDECL void RendererBase::RecreateGBuffers(PerFrameData& perframeData, RenderTargets* renderTargets)
    {
        using namespace Ifrit::Graphics::Rhi;
        auto rhi          = m_app->GetRhi();
        auto rtArea       = renderTargets->GetRenderArea();
        auto needRecreate = (perframeData.m_gbuffer.m_rtCreated == 0);

        u32  actualRtWidth = 0, actualRtHeight = 0;
        GetSupersampledRenderArea(renderTargets, &actualRtWidth, &actualRtHeight);

        if (!needRecreate)
        {
            needRecreate =
                (perframeData.m_gbuffer.m_rtWidth != actualRtWidth || perframeData.m_gbuffer.m_rtHeight != actualRtHeight);
        }
        if (needRecreate)
        {
            perframeData.m_gbuffer.m_rtCreated = 1;
            perframeData.m_gbuffer.m_rtWidth   = actualRtWidth;
            perframeData.m_gbuffer.m_rtHeight  = actualRtHeight;

            auto tarGetUsage = RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT | RhiImageUsage::RHI_IMAGE_USAGE_TRANSFER_DST_BIT;

            // auto targetFomrat = RhiImageFormat::RhiImgFmt_R8G8B8A8_UNORM;
            auto targetFomrat = RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT;
            perframeData.m_gbuffer.m_albedo_materialFlags =
                rhi->CreateTexture2D("Render_GAlbedo", actualRtWidth, actualRtHeight, targetFomrat, tarGetUsage, true);
            perframeData.m_gbuffer.m_emissive =
                rhi->CreateTexture2D("Render_GEmissive", actualRtWidth, actualRtHeight, targetFomrat, tarGetUsage, true);
            perframeData.m_gbuffer.m_normal_smoothness =
                rhi->CreateTexture2D("Render_GNormSmooth", actualRtWidth, actualRtHeight, targetFomrat, tarGetUsage, true);
            perframeData.m_gbuffer.m_specular_occlusion =
                rhi->CreateTexture2D("Render_GSpecOccl", actualRtWidth, actualRtHeight, targetFomrat,
                    tarGetUsage | RhiImageUsage::RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, true);
            perframeData.m_gbuffer.m_specular_occlusion_intermediate =
                rhi->CreateTexture2D("Render_GSpecOccl_Intm", actualRtWidth, actualRtHeight, targetFomrat,
                    tarGetUsage | RhiImageUsage::RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, true);
            perframeData.m_gbuffer.m_shadowMask =
                rhi->CreateTexture2D("Render_GShadowMask", actualRtWidth, actualRtHeight, targetFomrat, tarGetUsage, true);

            // sampler
            perframeData.m_gbuffer.m_albedo_materialFlags_sampId = rhi->RegisterCombinedImageSampler(
                perframeData.m_gbuffer.m_albedo_materialFlags.get(), m_immRes.m_linearSampler.get());
            perframeData.m_gbuffer.m_normal_smoothness_sampId = rhi->RegisterCombinedImageSampler(
                perframeData.m_gbuffer.m_normal_smoothness.get(), m_immRes.m_linearSampler.get());
            perframeData.m_gbuffer.m_specular_occlusion_sampId = rhi->RegisterCombinedImageSampler(
                perframeData.m_gbuffer.m_specular_occlusion.get(), m_immRes.m_linearSampler.get());
            perframeData.m_gbuffer.m_specular_occlusion_intermediate_sampId = rhi->RegisterCombinedImageSampler(
                perframeData.m_gbuffer.m_specular_occlusion_intermediate.get(), m_immRes.m_linearSampler.get());

            // color rts
            RenderingUtil::warpRenderTargets(rhi, perframeData.m_gbuffer.m_specular_occlusion.get(),
                perframeData.m_gbuffer.m_specular_occlusion_colorRT,
                perframeData.m_gbuffer.m_specular_occlusion_RTs);

            // barriers
            perframeData.m_gbuffer.m_normal_smoothnessBarrier.m_type = RhiBarrierType::UAVAccess;
            perframeData.m_gbuffer.m_normal_smoothnessBarrier.m_uav.m_texture =
                perframeData.m_gbuffer.m_normal_smoothness.get();
            perframeData.m_gbuffer.m_normal_smoothnessBarrier.m_uav.m_type = RhiResourceType::Texture;

            perframeData.m_gbuffer.m_specular_occlusionBarrier.m_type = RhiBarrierType::UAVAccess;
            perframeData.m_gbuffer.m_specular_occlusionBarrier.m_uav.m_texture =
                perframeData.m_gbuffer.m_specular_occlusion.get();
            perframeData.m_gbuffer.m_specular_occlusionBarrier.m_uav.m_type = RhiResourceType::Texture;

            // Then gbuffer refs
            PerFrameData::GBufferDesc gbufferDesc;
            gbufferDesc.m_albedo_materialFlags = perframeData.m_gbuffer.m_albedo_materialFlags->GetDescId();
            gbufferDesc.m_emissive             = perframeData.m_gbuffer.m_emissive->GetDescId();
            gbufferDesc.m_normal_smoothness    = perframeData.m_gbuffer.m_normal_smoothness->GetDescId();
            gbufferDesc.m_specular_occlusion   = perframeData.m_gbuffer.m_specular_occlusion->GetDescId();
            gbufferDesc.m_shadowMask           = perframeData.m_gbuffer.m_shadowMask->GetDescId();

            // Then gbuffer desc
            perframeData.m_gbuffer.m_gbufferRefs =
                rhi->CreateBufferDevice("Render_GbufferRef", sizeof(PerFrameData::GBufferDesc),
                    RhiBufferUsage::RhiBufferUsage_CopyDst | RhiBufferUsage::RhiBufferUsage_SSBO, true);
            auto stagedBuf = rhi->CreateStagedSingleBuffer(perframeData.m_gbuffer.m_gbufferRefs.get());
            auto tq        = rhi->GetQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
            tq->RunSyncCommand([&](const RhiCommandList* cmd) {
                stagedBuf->CmdCopyToDevice(cmd, &gbufferDesc, sizeof(PerFrameData::GBufferDesc), 0);
            });
            perframeData.m_gbuffer.m_gbufferDesc = rhi->createBindlessDescriptorRef();
            perframeData.m_gbuffer.m_gbufferDesc->AddStorageBuffer(perframeData.m_gbuffer.m_gbufferRefs.get(), 0);

            // Then gbuffer desc for pixel shader
            perframeData.m_gbufferDescFrag = rhi->createBindlessDescriptorRef();
            perframeData.m_gbufferDescFrag->AddCombinedImageSampler(perframeData.m_gbuffer.m_albedo_materialFlags.get(),
                m_immRes.m_linearSampler.get(), 0);
            perframeData.m_gbufferDescFrag->AddCombinedImageSampler(perframeData.m_gbuffer.m_specular_occlusion.get(),
                m_immRes.m_linearSampler.get(), 1);
            perframeData.m_gbufferDescFrag->AddCombinedImageSampler(perframeData.m_gbuffer.m_normal_smoothness.get(),
                m_immRes.m_linearSampler.get(), 2);
            perframeData.m_gbufferDescFrag->AddCombinedImageSampler(perframeData.m_gbuffer.m_emissive.get(),
                m_immRes.m_linearSampler.get(), 3);
            perframeData.m_gbufferDescFrag->AddCombinedImageSampler(perframeData.m_gbuffer.m_shadowMask.get(),
                m_immRes.m_linearSampler.get(), 4);

            // Create a uav barrier for the gbuffer
            perframeData.m_gbuffer.m_gbufferBarrier.clear();
            RhiResourceBarrier bAlbedo;
            bAlbedo.m_type          = RhiBarrierType::UAVAccess;
            bAlbedo.m_uav.m_texture = perframeData.m_gbuffer.m_albedo_materialFlags.get();
            bAlbedo.m_uav.m_type    = RhiResourceType::Texture;

            RhiResourceBarrier bEmissive;
            bEmissive.m_type          = RhiBarrierType::UAVAccess;
            bEmissive.m_uav.m_texture = perframeData.m_gbuffer.m_emissive.get();
            bEmissive.m_uav.m_type    = RhiResourceType::Texture;

            RhiResourceBarrier bNormal;
            bNormal.m_type          = RhiBarrierType::UAVAccess;
            bNormal.m_uav.m_texture = perframeData.m_gbuffer.m_normal_smoothness.get();
            bNormal.m_uav.m_type    = RhiResourceType::Texture;

            RhiResourceBarrier bSpecular;
            bSpecular.m_type          = RhiBarrierType::UAVAccess;
            bSpecular.m_uav.m_texture = perframeData.m_gbuffer.m_specular_occlusion.get();
            bSpecular.m_uav.m_type    = RhiResourceType::Texture;

            RhiResourceBarrier bShadow;
            bShadow.m_type          = RhiBarrierType::UAVAccess;
            bShadow.m_uav.m_texture = perframeData.m_gbuffer.m_shadowMask.get();
            bShadow.m_uav.m_type    = RhiResourceType::Texture;

            perframeData.m_gbuffer.m_gbufferBarrier.push_back(bAlbedo);
            perframeData.m_gbuffer.m_gbufferBarrier.push_back(bEmissive);
            perframeData.m_gbuffer.m_gbufferBarrier.push_back(bNormal);
            perframeData.m_gbuffer.m_gbufferBarrier.push_back(bSpecular);
            perframeData.m_gbuffer.m_gbufferBarrier.push_back(bShadow);
        }
    }

    IFRIT_APIDECL void RendererBase::PrepareDeviceResources(PerFrameData& perframeData, RenderTargets* renderTargets)
    {
        using namespace Ifrit::Graphics::Rhi;
        auto rhi        = m_app->GetRhi();
        auto renderArea = renderTargets->GetRenderArea();

        u32  actualRenderWidth  = 0;
        u32  actualRenderHeight = 0;
        GetSupersampledRenderArea(renderTargets, &actualRenderWidth, &actualRenderHeight);

        auto& primaryView                      = perframeData.m_views[0];
        primaryView.m_viewType                 = PerFrameData::ViewType::Primary;
        primaryView.m_viewData.m_renderHeightf = static_cast<float>(actualRenderHeight);
        primaryView.m_viewData.m_renderWidthf  = static_cast<float>(actualRenderWidth);
        primaryView.m_renderHeight             = actualRenderHeight;
        primaryView.m_renderWidth              = actualRenderWidth;
        primaryView.m_viewData.m_hizLods =
            std::floor(std::log2(static_cast<float>(std::max(actualRenderWidth, actualRenderHeight)))) + 1.0f;

        // Shadow views data has been filled in CollectPerframeData

        std::vector<std::shared_ptr<RhiStagedSingleBuffer>> stagedBuffers;
        std::vector<void*>                                  pendingVertexBuffers;
        std::vector<u32>                                    pendingVertexBufferSizes;

        for (u32 i = 0; i < perframeData.m_views.size(); i++)
        {
            bool  initLastFrameMatrix = false;
            auto& curView             = perframeData.m_views[i];
            if (curView.m_viewBindlessRef == nullptr)
            {
                curView.m_viewBuffer =
                    rhi->CreateBufferCoherent(sizeof(PerFramePerViewData), RhiBufferUsage::RhiBufferUsage_Uniform);
                curView.m_viewBindlessRef = rhi->createBindlessDescriptorRef();
                curView.m_viewBindlessRef->AddUniformBuffer(curView.m_viewBuffer.get(), 0);
                curView.m_viewBufferLast =
                    rhi->CreateBufferCoherent(sizeof(PerFramePerViewData), RhiBufferUsage::RhiBufferUsage_Uniform);

                curView.m_viewBindlessRef->AddUniformBuffer(curView.m_viewBufferLast.get(), 1);
                initLastFrameMatrix    = true;
                curView.m_viewBufferId = rhi->RegisterUniformBuffer(curView.m_viewBuffer.get());
            }

            // Update view buffer

            auto viewBuffer    = curView.m_viewBuffer;
            auto viewBufferAct = viewBuffer->GetActiveBuffer();
            viewBufferAct->MapMemory();
            viewBufferAct->WriteBuffer(&curView.m_viewData, sizeof(PerFramePerViewData), 0);
            viewBufferAct->FlushBuffer();
            viewBufferAct->UnmapMemory();

            // Init last's frame matrix for the first frame
            if (initLastFrameMatrix)
            {
                curView.m_viewDataOld = curView.m_viewData;
            }
            auto viewBufferLast    = curView.m_viewBufferLast;
            auto viewBufferLastAct = viewBufferLast->GetActiveBuffer();
            viewBufferLastAct->MapMemory();
            viewBufferLastAct->WriteBuffer(&curView.m_viewDataOld, sizeof(PerFramePerViewData), 0);
            viewBufferLastAct->FlushBuffer();
            viewBufferLastAct->UnmapMemory();
        }

        // Per effect data
        u32 curShaderMaterialId = 0;
        for (auto& shaderEffectId : perframeData.m_enabledEffects)
        {
            auto& shaderEffect = perframeData.m_shaderEffectData[shaderEffectId];
            // find whether batched object data should be recreated
            auto  lastObjectCount = shaderEffect.m_lastObjectCount;
            auto  objectCount     = SizeCast<u32>(shaderEffect.m_materials.size());
            if (lastObjectCount != objectCount || lastObjectCount == ~0u)
            {
                // TODO/EMERGENCY: release old buffer
                shaderEffect.m_lastObjectCount = objectCount;
                shaderEffect.m_objectData.resize(objectCount);
                shaderEffect.m_batchedObjectData =
                    rhi->CreateBufferCoherent(sizeof(PerObjectData) * objectCount, RhiBufferUsage::RhiBufferUsage_SSBO);

                // TODO: update instead of recreate
                shaderEffect.m_batchedObjBufRef = rhi->createBindlessDescriptorRef();
                shaderEffect.m_batchedObjBufRef->AddStorageBuffer(shaderEffect.m_batchedObjectData.get(), 0);
            }

            // preloading meshes
            Ifrit::Common::Utility::UnorderedFor<size_t>(0, shaderEffect.m_materials.size(),
                [&](size_t x) { auto mesh = shaderEffect.m_meshes[x]; });

            // load meshes
            for (int i = 0; i < shaderEffect.m_materials.size(); i++)
            {
                // Setup transform buffers
                auto                                 transform           = shaderEffect.m_transforms[i];
                std::shared_ptr<RhiMultiBuffer>      transformBuffer     = nullptr;
                std::shared_ptr<RhiMultiBuffer>      transformBufferLast = nullptr;
                std::shared_ptr<RhiDescHandleLegacy> bindlessRef         = nullptr;
                std::shared_ptr<RhiDescHandleLegacy> bindlessRefLast     = nullptr;
                transform->GetGPUResource(transformBuffer, transformBufferLast, bindlessRef, bindlessRefLast);
                bool initLastFrameMatrix = false;
                if (transformBuffer == nullptr)
                {
                    transformBuffer =
                        rhi->CreateBufferCoherent(sizeof(MeshInstanceTransform), RhiBufferUsage::RhiBufferUsage_Uniform);
                    bindlessRef = rhi->RegisterUniformBuffer(transformBuffer.get());
                    transform->SetGPUResource(transformBuffer, transformBufferLast, bindlessRef, bindlessRefLast);
                    initLastFrameMatrix = true;
                    transformBufferLast =
                        rhi->CreateBufferCoherent(sizeof(MeshInstanceTransform), RhiBufferUsage::RhiBufferUsage_Uniform);
                    bindlessRefLast = rhi->RegisterUniformBuffer(transformBufferLast.get());
                    transform->SetGPUResource(transformBuffer, transformBufferLast, bindlessRef, bindlessRefLast);
                }

                // update uniform buffer, TODO: dirty flag
                auto transformDirty = transform->GetDirtyFlag();
                if (transformDirty.changed || transformDirty.lastChanged)
                {
                    MeshInstanceTransform model;
                    // Transpose is required because glsl uses column major matrices
                    model.model    = Math::Transpose(transform->GetModelToWorldMatrix());
                    model.invModel = Math::Transpose(Math::Inverse4(transform->GetModelToWorldMatrix()));
                    auto scale     = transform->GetScale();
                    model.maxScale = Vector4f(scale.x, scale.y, scale.z, 0.0); // std::max(scale.x, std::max(scale.y, scale.z));

                    auto buf = transformBuffer->GetActiveBuffer();
                    buf->MapMemory();
                    buf->WriteBuffer(&model, sizeof(MeshInstanceTransform), 0);
                    buf->FlushBuffer();
                    buf->UnmapMemory();
                    shaderEffect.m_objectData[i].transformRef     = bindlessRef->GetActiveId();
                    shaderEffect.m_objectData[i].transformRefLast = bindlessRefLast->GetActiveId();
                }

                if (initLastFrameMatrix)
                {
                    // transform->OnFrameCollecting();
                }

                if (initLastFrameMatrix || transformDirty.lastChanged)
                {
                    MeshInstanceTransform modelLast;
                    modelLast.model    = Math::Transpose(transform->GetModelToWorldMatrix());
                    modelLast.invModel = Math::Transpose(Math::Inverse4(transform->GetModelToWorldMatrix()));
                    auto lastScale     = transform->GetScaleLast();
                    modelLast.maxScale = Vector4f(lastScale.x, lastScale.y, lastScale.z, 0.0); // std::max(lastScale.x, std::max(lastScale.y, lastScale.z));
                    auto bufLast       = transformBufferLast->GetActiveBuffer();
                    bufLast->MapMemory();
                    bufLast->WriteBuffer(&modelLast, sizeof(MeshInstanceTransform), 0);
                    bufLast->FlushBuffer();
                    bufLast->UnmapMemory();
                    transform->OnFrameCollecting();
                }

                // Setup mesh buffers
                auto              mesh        = shaderEffect.m_meshes[i];
                auto              meshDataRef = mesh->LoadMesh();
                Mesh::GPUResource meshResource;
                bool              requireUpdate    = false;
                bool              haveMaterialData = false;

                mesh->GetGPUResource(meshResource);
                if (meshResource.objectBuffer == nullptr || mesh->m_resourceDirty)
                {
                    requireUpdate                             = true;
                    mesh->m_resourceDirty                     = false;
                    meshDataRef->m_cpCounter.totalBvhNodes    = SizeCast<u32>(meshDataRef->m_bvhNodes.size());
                    meshDataRef->m_cpCounter.totalLods        = meshDataRef->m_maxLod;
                    meshDataRef->m_cpCounter.totalNumClusters = SizeCast<u32>(meshDataRef->m_clusterGroups.size());

                    auto tmpUsage             = RhiBufferUsage_CopyDst | RhiBufferUsage_SSBO;
                    meshResource.vertexBuffer = rhi->CreateBufferDevice(
                        "Mesh_Vertex", SizeCast<u32>(meshDataRef->m_verticesAligned.size() * sizeof(Vector4f)), tmpUsage, true);
                    meshResource.normalBuffer = rhi->CreateBufferDevice(
                        "Mesh_Normal", SizeCast<u32>(meshDataRef->m_normalsAligned.size() * sizeof(Vector4f)), tmpUsage, true);
                    meshResource.uvBuffer = rhi->CreateBufferDevice(
                        "Mesh_UV", SizeCast<u32>(meshDataRef->m_uvs.size() * sizeof(Vector2f)), tmpUsage, true);
                    meshResource.bvhNodeBuffer = rhi->CreateBufferDevice(
                        "Mesh_BVHNode",
                        SizeCast<u32>(meshDataRef->m_bvhNodes.size() * sizeof(MeshProcLib::MeshProcess::FlattenedBVHNode)),
                        tmpUsage, true);
                    meshResource.clusterGroupBuffer = rhi->CreateBufferDevice(
                        "Mesh_ClusterGroup",
                        SizeCast<u32>(meshDataRef->m_clusterGroups.size() * sizeof(MeshProcLib::MeshProcess::ClusterGroup)),
                        tmpUsage, true);
                    meshResource.meshletBuffer = rhi->CreateBufferDevice(
                        "Mesh_Cluster", SizeCast<u32>(meshDataRef->m_meshlets.size() * sizeof(MeshData::MeshletData)), tmpUsage,
                        true);
                    meshResource.meshletVertexBuffer = rhi->CreateBufferDevice(
                        "Mesh_ClusterVertex", SizeCast<u32>(meshDataRef->m_meshletVertices.size() * sizeof(u32)), tmpUsage, true);
                    meshResource.meshletIndexBuffer = rhi->CreateBufferDevice(
                        "Mesh_ClusterIndex", SizeCast<u32>(meshDataRef->m_meshletTriangles.size() * sizeof(u32)), tmpUsage, true);
                    meshResource.meshletInClusterBuffer = rhi->CreateBufferDevice(
                        "Mesh_ClusterInGroups", SizeCast<u32>(meshDataRef->m_meshletInClusterGroup.size() * sizeof(u32)), tmpUsage,
                        true);
                    meshResource.cpCounterBuffer =
                        rhi->CreateBufferDevice("Mesh_CpCounter", SizeCast<u32>(sizeof(MeshData::GPUCPCounter)), tmpUsage, true);
                    meshResource.tangentBuffer = rhi->CreateBufferDevice(
                        "Mesh_Tangent", SizeCast<u32>(sizeof(Vector4f) * meshDataRef->m_tangents.size()), tmpUsage, true);
                    meshResource.indexBuffer = rhi->CreateBufferDevice(
                        "Mesh_Index", SizeCast<u32>(sizeof(u32) * meshDataRef->m_indices.size()), tmpUsage, true);

                    auto  materialDataSize = 0;
                    auto& materialRef      = shaderEffect.m_materials[i];
                    if (materialRef->m_data.size() > 0)
                    {
                        haveMaterialData                = true;
                        materialDataSize                = SizeCast<u32>(materialRef->m_data[0].size());
                        meshResource.materialDataBuffer = rhi->CreateBufferDevice(
                            "MaterialData", SizeCast<u32>(shaderEffect.m_materials[i]->m_data[0].size()), tmpUsage, true);
                    }
                    else
                    {
                        meshResource.materialDataBuffer = nullptr;
                        iWarn("Material data not found for mesh {}", i);
                    }

                    // Indices in bindless descriptors
                    meshResource.haveMaterialData = haveMaterialData;
                    // if (haveMaterialData) {
                    //   meshResource.materialDataBufferId = rhi->RegisterStorageBuffer(meshResource.materialDataBuffer.get());
                    // }

                    // Here, we assume that no double bufferring is allowed
                    // meaning no CPU-GPU data transfer is allowed for mesh data after
                    // initialization
                    Mesh::GPUObjectBuffer& objectBuffer   = meshResource.objectData;
                    objectBuffer.vertexBufferId           = meshResource.vertexBuffer->GetDescId();
                    objectBuffer.normalBufferId           = meshResource.normalBuffer->GetDescId();
                    objectBuffer.uvBufferId               = meshResource.uvBuffer->GetDescId();
                    objectBuffer.bvhNodeBufferId          = meshResource.bvhNodeBuffer->GetDescId();
                    objectBuffer.clusterGroupBufferId     = meshResource.clusterGroupBuffer->GetDescId();
                    objectBuffer.meshletBufferId          = meshResource.meshletBuffer->GetDescId();
                    objectBuffer.meshletVertexBufferId    = meshResource.meshletVertexBuffer->GetDescId();
                    objectBuffer.meshletIndexBufferId     = meshResource.meshletIndexBuffer->GetDescId();
                    objectBuffer.meshletInClusterBufferId = meshResource.meshletInClusterBuffer->GetDescId();
                    objectBuffer.cpCounterBufferId        = meshResource.cpCounterBuffer->GetDescId();
                    objectBuffer.boundingSphere           = mesh->GetBoundingSphere(meshDataRef->m_vertices);
                    objectBuffer.materialDataId           = haveMaterialData ? meshResource.materialDataBuffer->GetDescId() : 0;
                    objectBuffer.tangentBufferId          = meshResource.tangentBuffer->GetDescId();
                    objectBuffer.indexBufferId            = meshResource.indexBuffer->GetDescId();

                    // description for the whole mesh
                    meshResource.objectBuffer =
                        rhi->CreateBufferDevice("ObjectBuffer", sizeof(Mesh::GPUObjectBuffer), tmpUsage, true);

                    mesh->SetGPUResource(meshResource);
                }

                // Setup instance buffers
                auto                      meshInst = shaderEffect.m_instances[i];
                MeshInstance::GPUResource instanceResource;
                meshInst->GetGPUResource(instanceResource);
                auto& meshInstObjData = meshInst->m_resource.objectData;
                if (instanceResource.objectBuffer == nullptr)
                {

                    auto tmpUsage                  = RhiBufferUsage_CopyDst | RhiBufferUsage_SSBO;
                    requireUpdate                  = true;
                    instanceResource.cpQueueBuffer = rhi->CreateBufferDevice(
                        "Render_CpQueue", SizeCast<u32>(sizeof(u32) * meshDataRef->m_bvhNodes.size()), tmpUsage, true);

                    auto safeNumMeshlets = meshDataRef->m_numMeshletsEachLod[0];
                    if (meshDataRef->m_numMeshletsEachLod.size() > 1)
                        safeNumMeshlets += meshDataRef->m_numMeshletsEachLod[1];
                    instanceResource.filteredMeshlets =
                        rhi->CreateBufferDevice("Render_FilteredClsters", sizeof(u32) * safeNumMeshlets, tmpUsage, true);

                    instanceResource.objectData.cpQueueBufferId    = instanceResource.cpQueueBuffer->GetDescId();
                    instanceResource.objectData.filteredMeshletsId = instanceResource.filteredMeshlets->GetDescId();

                    instanceResource.objectBuffer =
                        rhi->CreateBufferDevice("Render_Objects", sizeof(MeshInstance::GPUObjectBuffer), tmpUsage, true);

                    meshInst->SetGPUResource(instanceResource);
                }

                shaderEffect.m_objectData[i].objectDataRef   = meshResource.objectBuffer->GetDescId();
                shaderEffect.m_objectData[i].instanceDataRef = meshInst->m_resource.objectBuffer->GetDescId();
                shaderEffect.m_objectData[i].materialId      = curShaderMaterialId;

                // update vertex buffer, TODO: dirty flag
                if (requireUpdate)
                {
                    auto funcEnqueueStagedBuffer = [&](std::shared_ptr<RhiStagedSingleBuffer> stagedBuffer, void* data, u32 size) {
                        stagedBuffers.push_back(stagedBuffer);
                        pendingVertexBuffers.push_back(data);
                        pendingVertexBufferSizes.push_back(size);
                    };
#define enqueueStagedBuffer(name, vecBuffer)                                    \
    auto staged##name = rhi->CreateStagedSingleBuffer(meshResource.name.get()); \
    funcEnqueueStagedBuffer(                                                    \
        staged##name, meshDataRef->vecBuffer.data(),                            \
        SizeCast<u32>(meshDataRef->vecBuffer.size() * sizeof(decltype(meshDataRef->vecBuffer)::value_type)))

                    enqueueStagedBuffer(vertexBuffer, m_verticesAligned);
                    enqueueStagedBuffer(normalBuffer, m_normalsAligned);
                    enqueueStagedBuffer(uvBuffer, m_uvs);
                    enqueueStagedBuffer(tangentBuffer, m_tangents);
                    enqueueStagedBuffer(indexBuffer, m_indices);
                    enqueueStagedBuffer(bvhNodeBuffer, m_bvhNodes);
                    enqueueStagedBuffer(clusterGroupBuffer, m_clusterGroups);
                    enqueueStagedBuffer(meshletBuffer, m_meshlets);
                    enqueueStagedBuffer(meshletVertexBuffer, m_meshletVertices);
                    enqueueStagedBuffer(meshletIndexBuffer, m_meshletTriangles);
                    enqueueStagedBuffer(meshletInClusterBuffer, m_meshletInClusterGroup);

                    auto stagedCPCounterBuffer = rhi->CreateStagedSingleBuffer(meshResource.cpCounterBuffer.get());
                    stagedBuffers.push_back(stagedCPCounterBuffer);
                    pendingVertexBuffers.push_back(&meshDataRef->m_cpCounter);
                    pendingVertexBufferSizes.push_back(sizeof(MeshData::GPUCPCounter));

                    std::shared_ptr<RhiStagedSingleBuffer> stagedMaterialDataBuffer = nullptr;
                    if (haveMaterialData)
                    {
                        stagedMaterialDataBuffer = rhi->CreateStagedSingleBuffer(meshResource.materialDataBuffer.get());
                        stagedBuffers.push_back(stagedMaterialDataBuffer);
                        pendingVertexBuffers.push_back(&shaderEffect.m_materials[i]->m_data[0][0]);
                        pendingVertexBufferSizes.push_back(shaderEffect.m_materials[i]->m_data[0].size());
                    }

                    auto stagedObjectBuffer = rhi->CreateStagedSingleBuffer(meshResource.objectBuffer.get());
                    stagedBuffers.push_back(stagedObjectBuffer);
                    pendingVertexBuffers.push_back(&mesh->m_resource.objectData);
                    pendingVertexBufferSizes.push_back(sizeof(Mesh::GPUObjectBuffer));

                    auto stagedInstanceObjectBuffer = rhi->CreateStagedSingleBuffer(instanceResource.objectBuffer.get());
                    stagedBuffers.push_back(stagedInstanceObjectBuffer);
                    pendingVertexBuffers.push_back(&meshInst->m_resource.objectData);
                    pendingVertexBufferSizes.push_back(sizeof(MeshInstance::GPUObjectBuffer));

#undef enqueueStagedBuffer
                }
            }

            // update batched object data
            auto batchedObjectData    = shaderEffect.m_batchedObjectData;
            auto batchedObjectDataAct = batchedObjectData->GetActiveBuffer();
            batchedObjectDataAct->MapMemory();
            batchedObjectDataAct->WriteBuffer(shaderEffect.m_objectData.data(), sizeof(PerObjectData) * objectCount, 0);
            batchedObjectDataAct->FlushBuffer();
            batchedObjectDataAct->UnmapMemory();

            curShaderMaterialId++;
        }
        // Issue a command buffer to copy data to GPU
        if (stagedBuffers.size() > 0)
        {
            auto queue = rhi->GetQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
            queue->RunSyncCommand([&](const RhiCommandList* cmd) {
                for (int i = 0; i < stagedBuffers.size(); i++)
                {
                    stagedBuffers[i]->CmdCopyToDevice(cmd, pendingVertexBuffers[i], pendingVertexBufferSizes[i], 0);
                }
            });
        }
    }

    IFRIT_APIDECL void RendererBase::UpdateLastFrameTransforms(PerFrameData& perframeData)
    {
        throw std::runtime_error("Deprecated");
    }

    IFRIT_APIDECL void RendererBase::EndFrame(const std::vector<GPUCommandSubmission*>& cmdToWait)
    {
        auto rhi = m_app->GetRhi();
        using namespace Ifrit::Graphics;
        auto drawq           = rhi->GetQueue(Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);
        auto swapchainImg    = rhi->GetSwapchainImage();
        auto sRenderComplete = rhi->GetSwapchainRenderDoneEventHandler();
        auto cmd             = drawq->RunAsyncCommand(
            [&](const Rhi::RhiCommandList* cmd) {
                Rhi::RhiTransitionBarrier barrier;
                barrier.m_texture     = swapchainImg;
                barrier.m_type        = Rhi::RhiResourceType::Texture;
                barrier.m_dstState    = Rhi::RhiResourceState::Present;
                barrier.m_subResource = { 0, 0, 1, 1 };

                Rhi::RhiResourceBarrier barrier2;
                barrier2.m_type       = Rhi::RhiBarrierType::Transition;
                barrier2.m_transition = barrier;

                cmd->AddResourceBarrier({ barrier2 });
            },
            cmdToWait, { sRenderComplete.get() });
        rhi->EndFrame();
    }

    IFRIT_APIDECL std::unique_ptr<RendererBase::GPUCommandSubmission> RendererBase::BeginFrame()
    {
        auto rhi = m_app->GetRhi();
        rhi->BeginFrame();
        return rhi->GetSwapchainFrameReadyEventHandler();
    }

} // namespace Ifrit::Core