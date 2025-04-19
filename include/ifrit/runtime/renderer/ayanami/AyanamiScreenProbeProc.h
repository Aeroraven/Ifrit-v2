
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
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraph.h"

namespace Ifrit::Runtime::Ayanami
{
    struct AyanamiScreenProbeProcessorPrivate;
    class IFRIT_RUNTIME_API AyanamiScreenProbeProcessor
    {
    private:
        AyanamiScreenProbeProcessorPrivate* m_Private = nullptr;
        Graphics::Rhi::RhiBackend*          m_Rhi     = nullptr;

    public:
        AyanamiScreenProbeProcessor(Graphics::Rhi::RhiBackend* rhi);
        virtual ~AyanamiScreenProbeProcessor();

        void InitContext(FrameGraphBuilder& builder, u32 maxRtWidth, u32 maxRtHeight, f32 adaptiveProbesRatio);
        void AdaptiveScreenProbePlace(
            FrameGraphBuilder& builder, u32 perframeCBV, FGTextureNodeRef viewNormal, FGTextureNodeRef viewDepth);
        void ProbeScreenTrace(FrameGraphBuilder& builder, u32 perframeCBV, FGBufferNodeRef hizBuffer);
        void PrepareMeshDFCulling(
            FrameGraphBuilder& builder, u32 numTotalMdfs, Vector3f worldBoundMin, Vector3f worldBoundMax);
        void ScatterMeshDFToGrids(FrameGraphBuilder& builder, u32 perframeCBV, u32 numTotalMdfs, u32 meshDFDescUAV);
        void ProbeMDFTrace(
            FrameGraphBuilder& builder, u32 perframeCBV, u32 meshDFDescUAV, FGTextureNodeRef gbufferDepth);
        void             ProbeGDFTrace(FrameGraphBuilder& builder, u32 perframeCBV, FGTextureNodeRef gbufferDepth,
                        FGTextureNodeRef globalDF, u32 globalDFWSRang);

        FGBufferNodeRef  GetAdaptiveProbesList() const;
        FGBufferNodeRef  GetAdaptiveProbesCounter() const;
        FGTextureNodeRef GetScreenProbeRadianceAtlas() const;
    };
} // namespace Ifrit::Runtime::Ayanami