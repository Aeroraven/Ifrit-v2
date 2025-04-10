
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

#define DECLARE_VS(name) name "/VS"
#define DECLARE_FS(name) name "/FS"
#define DECLARE_CS(name) name "/CS"
#define DECLARE_MS(name) name "/MS"

#define SDEF IF_CONSTEXPR static const char*
    static struct InternalShaderTableAyanami
    {
        SDEF CopyFS                    = DECLARE_FS("Ayanami/Copy");
        SDEF CopyVS                    = DECLARE_VS("Ayanami/Copy");
        SDEF DirectShadowVisibilityCS  = DECLARE_CS("Ayanami/DirectShadowVisibility");
        SDEF GlobalDFRayMarchCS        = DECLARE_CS("Ayanami/GlobalDFRayMarch");
        SDEF RayMarchCS                = DECLARE_CS("Ayanami/RayMarch");
        SDEF SurfaceCacheGenFS         = DECLARE_FS("Ayanami/SurfaceCacheGen");
        SDEF SurfaceCacheGenVS         = DECLARE_VS("Ayanami/SurfaceCacheGen");
        SDEF TrivialGlobalDFCompCS     = DECLARE_CS("Ayanami/TrivialGlobalDFComp");
        SDEF DFShadowTileCullingMS     = DECLARE_MS("Ayanami/DFShadowTileCulling");
        SDEF DFShadowTileCullingFS     = DECLARE_FS("Ayanami/DFShadowTileCulling");
        SDEF DFShadowFS                = DECLARE_FS("Ayanami/DFShadow");
        SDEF TestDeferShadingFS        = DECLARE_FS("Ayanami/TestDeferShading");
        SDEF DFShadowVisibilityCS      = DECLARE_CS("Ayanami/DFRadianceInjection");
        SDEF ObjectGridCompositionCS   = DECLARE_CS("Ayanami/ObjectGridComposition");
        SDEF RadiosityTraceCS          = DECLARE_CS("Ayanami/RadiosityTrace");
        SDEF SurfaceCacheDirectLightCS = DECLARE_CS("Ayanami/SurfaceCacheDirectLighting");

        SDEF DbgReconFromSurfaceCacheCS = DECLARE_CS("Ayanami/DbgReconFromSurfaceCache");
        SDEF DbgSampleReconDepthCS      = DECLARE_CS("Ayanami/DbgSampleReconDepthCS");
        SDEF DbgSampleObjectGridsCS     = DECLARE_CS("Ayanami/DbgSampleObjectGridsCS");
        SDEF DbgVisObjGridsMS           = DECLARE_MS("Ayanami/DbgVisObjGridsMS");
        SDEF DbgVisObjGridsFS           = DECLARE_FS("Ayanami/DbgVisObjGridsFS");

    } kIntShaderTableAyanami;

#undef SDEF

#undef DECLARE_VS
#undef DECLARE_FS
#undef DECLARE_CS
#undef DECLARE_MS

} // namespace Ifrit::Runtime::Internal