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

namespace Ifrit::Runtime::Artemis
{

    enum class MPMSimulatorTopologySource : u8
    {

        External,
        Preset
    };

    enum class MPMSimulatorProblemDimension : u8
    {
        TwoDimensional,
        ThreeDimensional,
    };

    enum class MPMSimulatorVariant : u8
    {
        NonMLS,
        MLS,
        PBMPM,
    };

    enum class MPMSimulatorParticleType : u8
    {
        Jelly = 0,
        Fluid = 1,
        Snow  = 2,
        Visco = 3
    };

    struct MPMSimulatorConfig
    {
        IF_CONSTEXPR static u32      kDefaultGridSizeX = 64;

        MPMSimulatorTopologySource   m_TopoSource = MPMSimulatorTopologySource::Preset;
        MPMSimulatorProblemDimension m_Dimension  = MPMSimulatorProblemDimension::TwoDimensional;
        MPMSimulatorVariant          m_Variant    = MPMSimulatorVariant::PBMPM;

        u32                          m_MaxParticles = 614514;
        u32                          m_MaxContacts  = 1919810;
        Vector3u                     m_GridSize     = Vector3u(kDefaultGridSizeX, kDefaultGridSizeX, kDefaultGridSizeX);
        Vector3f                     m_GridOffset   = Vector3f(0.0f);
        Vector3u                     m_GridBoundaryWidth = Vector3u(3, 3, 3);
        Vector3f                     m_Gravity           = Vector3f(0.0f, -1.0f, 0.0f);
        f32                          m_GridSpacing       = 1.0f / kDefaultGridSizeX;
        f32                          m_DefaultMass       = (0.5f / kDefaultGridSizeX);
        f32                          m_DefaultDensity    = 1.0f;

        f32                          m_DefaultYoungsModulus   = 200.0f;
        f32                          m_DefaultPoissonRatio    = 0.2f;
        f32                          m_DefaultViscoPlasticity = 0.7f;
        u32                          m_DefaultNumParticles    = 114514;
        u32                          m_Substeps               = 5;
        MPMSimulatorParticleType     m_DefaultParticleType    = MPMSimulatorParticleType::Fluid;

        // PBMPM
        u32                          m_PbMpmIterations                           = 4;
        f32                          m_PbMpmDefaultElasticityInterpolationFactor = 0.01f;
        f32                          m_PbMpmDefaultElasticityRelaxationFactor    = 1.5f;
        f32                          m_PbMpmDefaultLiquidViscosity               = 0.000f;
        f32                          m_PbMpmDefaultLiquidRelaxation              = 1.1f;

        // Rigid Coupling
        bool                         m_EnableRigidCoupling = true;

        // Interaction
        f32                          m_MouseActivation = 0.001f;
        f32                          m_MouseRadius     = 0.1f;
        bool                         m_EnableRendering = true;
    };

    struct MPMParticleEmitArgs
    {

        IF_CONSTEXPR static u32  kGlobalDefaultGridSizeX = MPMSimulatorConfig::kDefaultGridSizeX;

        Vector4f                 m_EmitColor     = Vector4f(1.0f, 1.0f, 1.0f, 1.0f);
        MPMSimulatorParticleType m_MaterialType  = MPMSimulatorParticleType::Fluid;
        f32                      m_Mass          = 0.5f / kGlobalDefaultGridSizeX;
        f32                      m_Density       = 1.0f;
        f32                      m_YoungsModulus = 200.0f;
        f32                      m_PoissonRatio  = 0.2f;
    };
} // namespace Ifrit::Runtime::Artemis
