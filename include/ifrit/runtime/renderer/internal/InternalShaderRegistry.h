
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
#include "ifrit/runtime/material/ShaderRegistry.h"
#include "ifrit/runtime/base/Base.h"

namespace Ifrit::Runtime::Internal
{
    // I found that the boot time is mostly used for shader compilation
    // So I decided to put the shader compilation here.
    // For unreal, shaders are mostly classes, but I dont think I can afford such compilation time
    // MULTITHREADING, START!
    IFRIT_RUNTIME_API void RegisterRuntimeInternalShaders(ShaderRegistry* shaderRegistry);

#define DECLARE_VS(name) name "/VS"
#define DECLARE_FS(name) name "/FS"
#define DECLARE_CS(name) name "/CS"
#define DECLARE_MS(name) name "/MS"

#define SDEF IF_CONSTEXPR static const char*
    static struct InternalShaderTable
    {
        IF_CONSTEXPR static struct
        {
            SDEF HBAOCS = DECLARE_CS("GI/HBAO");
            SDEF SSGICS = DECLARE_CS("GI/SSGI");
        } GI;

        IF_CONSTEXPR static struct
        {
            SDEF PASIndirectRadianceCS   = DECLARE_CS("Atmo/PAS/IndirectRadiance");
            SDEF PASIrradianceCS         = DECLARE_CS("Atmo/PAS/Irradiance");
            SDEF PASMultipleScatteringCS = DECLARE_CS("Atmo/PAS/MultipleScattering"); // TODO: add multiple scattering
            SDEF PASScatteringDensityCS  = DECLARE_CS("Atmo/PAS/ScatteringDensity");
            SDEF PASSingleScatteringCS   = DECLARE_CS("Atmo/PAS/SingleScattering");
            SDEF PASTransmittanceCS      = DECLARE_CS("Atmo/PAS/Transmittance");
        } Atmosphere;

        IF_CONSTEXPR static struct
        {
            SDEF CopyFS                    = DECLARE_FS("Ayanami/Copy");
            SDEF CopyVS                    = DECLARE_VS("Ayanami/Copy");
            SDEF DirectRadianceInjectionCS = DECLARE_CS("Ayanami/DirectRadianceInjection");
            SDEF GlobalDFRayMarchCS        = DECLARE_CS("Ayanami/GlobalDFRayMarch");
            SDEF RayMarchCS                = DECLARE_CS("Ayanami/RayMarch");
            SDEF SurfaceCacheGenFS         = DECLARE_FS("Ayanami/SurfaceCacheGen");
            SDEF SurfaceCacheGenVS         = DECLARE_VS("Ayanami/SurfaceCacheGen");
            SDEF TrivialGlobalDFCompCS     = DECLARE_CS("Ayanami/TrivialGlobalDFComp");
            SDEF DFShadowTileCullingMS     = DECLARE_MS("Ayanami/DFShadowTileCulling");
            SDEF DFShadowTileCullingFS     = DECLARE_FS("Ayanami/DFShadowTileCulling");
            SDEF DFShadowFS                = DECLARE_FS("Ayanami/DFShadow");
            SDEF TestDeferShadingFS        = DECLARE_FS("Ayanami/TestDeferShading");
        } Ayanami;

        IF_CONSTEXPR static struct
        {
            SDEF FullScreenVS    = DECLARE_VS("Common/FullScreen");
            SDEF SinglePassHzbCS = DECLARE_CS("Common/SinglePassHiZ");
        } Common;

        IF_CONSTEXPR static struct
        {
            SDEF ACESFS                   = DECLARE_FS("PostProc/ACES");
            SDEF FFTBloomCS               = DECLARE_CS("PostProc/FFTBloom");
            SDEF FFTBloomUpsampleCS       = DECLARE_CS("PostProc/FFTBloomUpsample");
            SDEF GaussianHoriFS           = DECLARE_FS("PostProc/GaussianHori");
            SDEF GaussianVertFS           = DECLARE_FS("PostProc/GaussianVert");
            SDEF GaussianKernelGenerateCS = DECLARE_CS("PostProc/GaussianKernelGenerate");
            SDEF GlobalFogFS              = DECLARE_FS("PostProc/GlobalFog");
            SDEF JointBilaterialFilterFS  = DECLARE_FS("PostProc/JointBilaterialFilter");
            SDEF StockhamDFT2CS           = DECLARE_CS("PostProc/StockhamDFT2");
        } Postprocess;

        IF_CONSTEXPR static struct
        {
            SDEF CommonVS = DECLARE_VS("PostProc/Common");
        } PostprocessVertex;

        IF_CONSTEXPR static struct
        {
            SDEF ClassifyMaterialCountCS   = DECLARE_CS("Syaro/ClassifyMaterial/Count");
            SDEF ClassifyMaterialReserveCS = DECLARE_CS("Syaro/ClassifyMaterial/Reserve");
            SDEF ClassifyMaterialScatterCS = DECLARE_CS("Syaro/ClassifyMaterial/Scatter");
            SDEF CombineVisBufferCS        = DECLARE_CS("Syaro/CombineVisBuffer");
            SDEF DeferredShadingFS         = DECLARE_FS("Syaro/DeferredShading");
            SDEF DeferredShadingVS         = DECLARE_VS("Syaro/DeferredShading");
            SDEF DeferredShadowingFS       = DECLARE_FS("Syaro/DeferredShadowing");
            SDEF DeferredShadowingVS       = DECLARE_VS("Syaro/DeferredShadowing");
            SDEF EmitDepthTargetCS         = DECLARE_CS("Syaro/EmitDepthTarget");
            SDEF EmitGBufferCS             = DECLARE_CS("Syaro/EmitGBuffer");
            SDEF InstanceCullingCS         = DECLARE_CS("Syaro/InstanceCulling");
            SDEF PersistentCullingCS       = DECLARE_CS("Syaro/PersistentCulling");
            SDEF PbrAtmoRenderCS           = DECLARE_CS("Syaro/PbrAtmoRender");
            SDEF SoftRasterizeCS           = DECLARE_CS("Syaro/SoftRasterize");
            SDEF TAAFS                     = DECLARE_FS("Syaro/TAA");
            SDEF TAAVS                     = DECLARE_VS("Syaro/TAA");
            SDEF TriangleViewFS            = DECLARE_FS("Syaro/TriangleView");
            SDEF TriangleViewVS            = DECLARE_VS("Syaro/TriangleView");
            SDEF VisBufferFS               = DECLARE_FS("Syaro/VisBuffer");
            SDEF VisBufferMS               = DECLARE_MS("Syaro/VisBuffer");
            SDEF VisBufferDepthMS          = DECLARE_MS("Syaro/VisBufferDepth");
        } Syaro;

    } kIntShaderTable;

#undef SDEF

#undef DECLARE_VS
#undef DECLARE_FS
#undef DECLARE_CS
#undef DECLARE_MS

} // namespace Ifrit::Runtime::Internal