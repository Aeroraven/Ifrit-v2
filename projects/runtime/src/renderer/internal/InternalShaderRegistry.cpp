
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

namespace Ifrit::Runtime::Internal
{
    IFRIT_APIDECL void RegisterRuntimeInternalShaders(ShaderRegistry* shaderRegistry)
    {
#define REG_SHADER(name, path, stage) shaderRegistry->RegisterShader(name, path, "main", stage)
#define REG_COMPUTE(name, path) REG_SHADER(name, path ".comp.glsl", Graphics::Rhi::RhiShaderStage::Compute)
#define REG_VERTEX(name, path) REG_SHADER(name, path ".vert.glsl", Graphics::Rhi::RhiShaderStage::Vertex)
#define REG_FRAGMENT(name, path) REG_SHADER(name, path ".frag.glsl", Graphics::Rhi::RhiShaderStage::Fragment)
#define REG_MESH(name, path) REG_SHADER(name, path ".mesh.glsl", Graphics::Rhi::RhiShaderStage::Mesh)

        const auto& IST    = kIntShaderTable;
        const auto& ISTAya = kIntShaderTableAyanami;
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
        REG_FRAGMENT(ISTAya.CopyFS, "Ayanami/Ayanami.CopyPass");
        REG_VERTEX(ISTAya.CopyVS, "Ayanami/Ayanami.CopyPass");
        REG_COMPUTE(ISTAya.DirectRadianceInjectionCS, "Ayanami/Ayanami.DirectionalRadianceInjection");
        REG_COMPUTE(ISTAya.GlobalDFRayMarchCS, "Ayanami/Ayanami.GlobalDFRayMarch");
        REG_COMPUTE(ISTAya.RayMarchCS, "Ayanami/Ayanami.RayMarch");
        REG_FRAGMENT(ISTAya.SurfaceCacheGenFS, "Ayanami/Ayanami.SurfaceCacheGen");
        REG_VERTEX(ISTAya.SurfaceCacheGenVS, "Ayanami/Ayanami.SurfaceCacheGen");
        REG_COMPUTE(ISTAya.TrivialGlobalDFCompCS, "Ayanami/Ayanami.TrivialGlobalDFComposite");
        REG_MESH(ISTAya.DFShadowTileCullingMS, "Ayanami/Ayanami.DFShadowTileCull");
        REG_FRAGMENT(ISTAya.DFShadowTileCullingFS, "Ayanami/Ayanami.DFShadowTileCull");
        REG_FRAGMENT(ISTAya.DFShadowFS, "Ayanami/Ayanami.DFShadow");
        REG_FRAGMENT(ISTAya.TestDeferShadingFS, "Ayanami/Ayanami.TestDeferShading");
        REG_COMPUTE(ISTAya.DFRadianceInjectionCS, "Ayanami/Ayanami.DFRadianceInjection");
        REG_COMPUTE(ISTAya.ObjectGridCompositionCS, "Ayanami/Ayanami.ObjectGridComposition");
        REG_COMPUTE(ISTAya.DbgReconFromSurfaceCacheCS, "Ayanami/Ayanami.Debug.ReconFromSurfaceCache");
        REG_COMPUTE(ISTAya.DbgSampleReconDepthCS, "Ayanami/Ayanami.Debug.SampleReconDepth");
        REG_COMPUTE(ISTAya.RadiosityTraceCS, "Ayanami/Ayanami.RadiosityTrace");

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

        iInfo("Internal: Compiling internal shaders...");
        shaderRegistry->WaitForShaderCompilations();
        iInfo("Internal: Internal shaders compiled.");

#undef REG_MESH
#undef REG_FRAGMENT
#undef REG_VERTEX
#undef REG_COMPUTE
#undef REG_SHADER
    }
} // namespace Ifrit::Runtime::Internal