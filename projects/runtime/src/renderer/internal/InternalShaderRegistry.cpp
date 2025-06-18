
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

#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.Ayanami.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.Neo.h"

#include "ifrit/runtime/simulation/internal/InternalShaderRegistry.Siro.h"

namespace Ifrit::Runtime::Internal
{
    IFRIT_APIDECL void RegisterRuntimeInternalShaders(ShaderRegistry* shaderRegistry)
    {
#define REG_SHADER(name, path, stage) shaderRegistry->RegisterShader(name, path, "main", stage)
#define REG_COMPUTE(name, path) REG_SHADER(name, path ".comp.glsl", Graphics::Rhi::RhiShaderStage::Compute)
#define REG_VERTEX(name, path) REG_SHADER(name, path ".vert.glsl", Graphics::Rhi::RhiShaderStage::Vertex)
#define REG_FRAGMENT(name, path) REG_SHADER(name, path ".frag.glsl", Graphics::Rhi::RhiShaderStage::Fragment)
#define REG_MESH(name, path) REG_SHADER(name, path ".mesh.glsl", Graphics::Rhi::RhiShaderStage::Mesh)

#define REG_SHADER_NEO(name, path, stage, entry) shaderRegistry->RegisterShader(name, path, entry, stage)
#define REG_COMPUTE_NEO(name, path, entry) \
    REG_SHADER_NEO(name, path ".comp.slang", Graphics::Rhi::RhiShaderStage::Compute, entry)
#define REG_VERTEX_NEO(name, path, entry) \
    REG_SHADER_NEO(name, path ".vert.slang", Graphics::Rhi::RhiShaderStage::Vertex, entry)
#define REG_FRAGMENT_NEO(name, path, entry) \
    REG_SHADER_NEO(name, path ".frag.slang", Graphics::Rhi::RhiShaderStage::Fragment, entry)
#define REG_MESH_NEO(name, path, entry) \
    REG_SHADER_NEO(name, path ".mesh.slang", Graphics::Rhi::RhiShaderStage::Mesh, entry)

        const auto& IST    = kIntShaderTable;
        const auto& ISTAya = kIntShaderTableAyanami;
        const auto& ISTNeo = kIntShaderTableNeo;

        const auto& ISTSiro = kIntShaderTableSiro;

        // GI & AO
        REG_COMPUTE(IST.GI.HBAOCS, "AmbientOcclusion/HBAO");
        REG_COMPUTE(IST.GI.SSGICS, "AmbientOcclusion/SSGI");

        // Atmosphere
        REG_COMPUTE(IST.Atmosphere.PASIndirectRadianceCS, "Atmosphere/PAS.ComputeIndirectIrradiance");
        REG_COMPUTE(IST.Atmosphere.PASIrradianceCS, "Atmosphere/PAS.ComputeIrradiance");
        REG_COMPUTE(IST.Atmosphere.PASMultipleScatteringCS, "Atmosphere/PAS.ComputeMultipleScattering");
        REG_COMPUTE(IST.Atmosphere.PASScatteringDensityCS, "Atmosphere/PAS.ComputeScatteringDensity");
        REG_COMPUTE(IST.Atmosphere.PASSingleScatteringCS, "Atmosphere/PAS.ComputeSingleScattering");
        REG_COMPUTE(IST.Atmosphere.PASTransmittanceCS, "Atmosphere/PAS.ComputeTransmittance");

        // Ayanami
        REG_FRAGMENT_NEO(ISTAya.CopyFS, "Ayanami/Ayanami.CopyTex", "CopyTexPS");
        REG_VERTEX_NEO(ISTAya.CopyVS, "Ayanami/Ayanami.CopyTex", "CopyTexVS");
        REG_COMPUTE(ISTAya.DirectShadowVisibilityCS, "Ayanami/Ayanami.DirectionalShadowVisibility");
        REG_COMPUTE_NEO(ISTAya.GlobalDFRayMarchCS, "Ayanami/Ayanami.Debug.GlobalDFRayMarch", "DebugGlobalDFRayMarchCS");
        REG_COMPUTE(ISTAya.RayMarchCS, "Ayanami/Ayanami.RayMarch");

        REG_COMPUTE_NEO(ISTAya.TrivialGlobalDFCompCS, "Ayanami/Ayanami.GlobalDF.Build", "GlobalDFBuildCS");
        REG_MESH(ISTAya.DFShadowTileCullingMS, "Ayanami/Ayanami.DFShadowTileCull");
        REG_FRAGMENT(ISTAya.DFShadowTileCullingFS, "Ayanami/Ayanami.DFShadowTileCull");
        REG_FRAGMENT(ISTAya.DFShadowFS, "Ayanami/Ayanami.DFShadow");
        REG_FRAGMENT_NEO(ISTAya.TestDeferShadingFS, "Ayanami/Ayanami.TestDeferShading", "TestDeferShadingPS");
        REG_COMPUTE(ISTAya.DFShadowVisibilityCS, "Ayanami/Ayanami.DFShadowVisibility");
        REG_COMPUTE(ISTAya.ObjectGridCompositionCS, "Ayanami/Ayanami.ObjectGridComposition");

        REG_COMPUTE(ISTAya.RadiosityTraceCS, "Ayanami/Ayanami.RadiosityTrace");
        REG_COMPUTE(ISTAya.RadiositySHConversionCS, "Ayanami/Ayanami.RadiositySHConversion");
        REG_COMPUTE_NEO(
            ISTAya.RadiositySHIntegrateCS, "Ayanami/Ayanami.Radiosity.SHIntegrate", "RadiositySHIntegrateCS");

        REG_COMPUTE_NEO(ISTAya.SurfaceCacheDirectLightCS, "Ayanami/Ayanami.SurfaceCache.DirectLighting",
            "SurfaceCacheDirectLightingCS");
        REG_COMPUTE_NEO(ISTAya.SurfaceCacheCombineLightCS, "Ayanami/Ayanami.SurfaceCache.CombineLighting",
            "SurfaceCacheCombineLightingCS");
        REG_FRAGMENT_NEO(ISTAya.SurfaceCacheGenFS, "Ayanami/Ayanami.SurfaceCache.Generate", "SurfaceCacheGeneratePS");
        REG_VERTEX_NEO(ISTAya.SurfaceCacheGenVS, "Ayanami/Ayanami.SurfaceCache.Generate", "SurfaceCacheGenerateVS");

        REG_COMPUTE(ISTAya.ScreenProbeAdaptivePlaceCS, "Ayanami/Ayanami.ScreenProbe.AdaptivePlace");
        REG_COMPUTE_NEO(ISTAya.ScreenProbeTraceScreenCS, "Ayanami/Ayanami.ScreenProbe.ScreenSpaceTrace",
            "ScreenProbeScreenSpaceTraceCS");
        REG_COMPUTE(ISTAya.ScreenProbeMDFCullPrepCS, "Ayanami/Ayanami.ScreenProbe.MDFCullMatPrep");
        REG_VERTEX(ISTAya.ScreenProbeMDFCullScatterVS, "Ayanami/Ayanami.ScreenProbe.CullMDFToGrids");
        REG_FRAGMENT(ISTAya.ScreenProbeMDFCullScatterFS, "Ayanami/Ayanami.ScreenProbe.CullMDFToGrids");
        REG_COMPUTE(ISTAya.ScreenProbeMDFTraceCS, "Ayanami/Ayanami.ScreenProbe.MDFTrace");
        REG_COMPUTE(ISTAya.ScreenProbeGDFTraceCS, "Ayanami/Ayanami.ScreenProbe.GDFTrace");
        REG_COMPUTE_NEO(
            ISTAya.ScreenProbeSHIntegrateCS, "Ayanami/Ayanami.ScreenProbe.IntegrateSH", "ScreenProbeIntegrateSHCS");
        REG_COMPUTE_NEO(
            ISTAya.ScreenProbePixelGatherCS, "Ayanami/Ayanami.ScreenProbe.PixelGather", "ScreenProbePixelGatherCS");
        REG_COMPUTE_NEO(ISTAya.ScreenProbeBorderFixCS, "Ayanami/Ayanami.ScreenProbe.OctMapBorderFix",
            "ScreenProbeOctMapBorderFixCS");

        REG_FRAGMENT_NEO(ISTAya.DeferredShadowFS, "Ayanami/Ayanami.FinalLighting.DirectShadow", "FinalDirectShadowPS");
        REG_FRAGMENT_NEO(
            ISTAya.DeferredLightingFS, "Ayanami/Ayanami.FinalLighting.DirectLighting", "FinalDirectLightingPS");
        REG_FRAGMENT_NEO(
            ISTAya.DeferredExpMixFS, "Ayanami/Ayanami.FinalLighting.ExperimentalMix", "FinalLightingMixPS");
        REG_COMPUTE_NEO(ISTAya.TemporalFilterIndirectCS, "Ayanami/Ayanami.FinalLighting.TemporalFilteringIndirect",
            "FinalLightingTemporalFilteringIndirectCS");

        REG_COMPUTE(ISTAya.DbgSampleObjectGridsCS, "Ayanami/Ayanami.Debug.SampleObjectGrids");
        REG_MESH(ISTAya.DbgVisObjGridsMS, "Ayanami/Ayanami.Debug.VisObjectGrids");
        REG_FRAGMENT_NEO(ISTAya.DbgVisObjGridsFS, "Ayanami/Ayanami.Debug.VisObjectGrids", "DebugVisObjectGridsPS");
        REG_COMPUTE_NEO(
            ISTAya.DbgVisAdaptiveProbeCS, "Ayanami/Ayanami.Debug.AdaptiveProbeLocate", "DebugAdaptiveProbeLocateCS");
        REG_COMPUTE_NEO(ISTAya.DbgVisScreenUniformProbeCS, "Ayanami/Ayanami.Debug.ScreenUniformProbeVis",
            "DebugScreenUniformProbeVisCS");
        REG_COMPUTE_NEO(ISTAya.DbgReconFromSurfaceCacheCS, "Ayanami/Ayanami.Debug.ReconFromSurfaceCache",
            "DebugReconFromSurfaceCacheCS");
        REG_COMPUTE_NEO(
            ISTAya.DbgSampleReconDepthCS, "Ayanami/Ayanami.Debug.SampleReconDepth", "DebugSampleReconDepthCS");

        // Common
        REG_VERTEX(IST.Common.FullScreenVS, "CommonPass/FullScreen");
        REG_COMPUTE(IST.Common.SinglePassHzbCS, "CommonPass/SinglePassHzb");

        // PostProcessing
        REG_FRAGMENT(IST.Postprocess.ACESFS, "Postprocess/ACESToneMapping");
        REG_COMPUTE(IST.Postprocess.FFTBloomCS, "Postprocess/FFTConv2d");
        REG_COMPUTE(IST.Postprocess.FFTBloomUpsampleCS, "Postprocess/FFTConv2d.Upsample");
        REG_FRAGMENT(IST.Postprocess.GaussianHoriFS, "Postprocess/GaussianHori");
        REG_FRAGMENT(IST.Postprocess.GaussianVertFS, "Postprocess/GaussianVert");
        REG_COMPUTE(IST.Postprocess.GaussianKernelGenerateCS, "Postprocess/GaussianKernelGenerate");
        REG_FRAGMENT(IST.Postprocess.GlobalFogFS, "Postprocess/GlobalFog");
        REG_FRAGMENT(IST.Postprocess.JointBilaterialFilterFS, "Postprocess/JointBilaterialFilter");
        REG_COMPUTE(IST.Postprocess.StockhamDFT2CS, "Postprocess/StockhamDFT2");

        // Postprocessing Vertex
        REG_VERTEX(IST.PostprocessVertex.CommonVS, "Postprocess/Postproc.Common");

        // Syaro
        REG_COMPUTE(IST.Syaro.ClassifyMaterialCountCS, "Syaro/Syaro.ClassifyMaterial.Count");
        REG_COMPUTE(IST.Syaro.ClassifyMaterialReserveCS, "Syaro/Syaro.ClassifyMaterial.Reserve");
        REG_COMPUTE(IST.Syaro.ClassifyMaterialScatterCS, "Syaro/Syaro.ClassifyMaterial.Scatter");
        REG_COMPUTE(IST.Syaro.CombineVisBufferCS, "Syaro/Syaro.CombineVisBuffer");
        REG_FRAGMENT(IST.Syaro.DeferredShadingFS, "Syaro/Syaro.DeferredShading");
        REG_VERTEX(IST.Syaro.DeferredShadingVS, "Syaro/Syaro.DeferredShading");
        REG_FRAGMENT(IST.Syaro.DeferredShadowingFS, "Syaro/Syaro.DeferredShadow");
        REG_VERTEX(IST.Syaro.DeferredShadowingVS, "Syaro/Syaro.DeferredShadow");
        REG_COMPUTE(IST.Syaro.EmitDepthTargetCS, "Syaro/Syaro.EmitDepthTarget");
        REG_COMPUTE(IST.Syaro.EmitGBufferCS, "Syaro/Syaro.EmitGBuffer.Default");
        REG_COMPUTE(IST.Syaro.InstanceCullingCS, "Syaro/Syaro.InstanceCulling");
        REG_COMPUTE(IST.Syaro.PersistentCullingCS, "Syaro/Syaro.PersistentCulling");
        REG_COMPUTE(IST.Syaro.PbrAtmoRenderCS, "Syaro/Syaro.PbrAtmoRender");
        REG_COMPUTE(IST.Syaro.SoftRasterizeCS, "Syaro/Syaro.SoftRasterize");
        REG_FRAGMENT(IST.Syaro.TAAFS, "Syaro/Syaro.TAA");
        REG_VERTEX(IST.Syaro.TAAVS, "Syaro/Syaro.TAA");
        REG_FRAGMENT(IST.Syaro.TriangleViewFS, "Syaro/Syaro.TriangleView");
        REG_VERTEX(IST.Syaro.TriangleViewVS, "Syaro/Syaro.TriangleView");
        REG_FRAGMENT(IST.Syaro.VisBufferFS, "Syaro/Syaro.VisBuffer");
        REG_MESH(IST.Syaro.VisBufferMS, "Syaro/Syaro.VisBuffer");
        REG_MESH(IST.Syaro.VisBufferDepthMS, "Syaro/Syaro.VisBufferDepth");

        // Neo
        REG_COMPUTE_NEO(ISTNeo.TestCS, "TestCS", "TestCS");

        // SIRO
        REG_COMPUTE_NEO(ISTSiro.TrivialPBDClothInit, "Siro/TrivialPBDCloth.Init", "TrivialPBDClothInit");

        iInfo("Internal: Compiling internal shaders...");
        shaderRegistry->WaitForShaderCompilations();
        iInfo("Internal: Internal shaders compiled.");

#undef REG_MESH
#undef REG_FRAGMENT
#undef REG_VERTEX
#undef REG_COMPUTE
#undef REG_SHADER

#undef REG_MESH_NEO
#undef REG_FRAGMENT_NEO
#undef REG_VERTEX_NEO
#undef REG_COMPUTE_NEO
    }
} // namespace Ifrit::Runtime::Internal