
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

#include "ifrit/runtime/physics/internal/InternalShaderRegistry.Artemis.h"

namespace Ifrit::Runtime::Internal
{
    IFRIT_APIDECL void RegisterRuntimeInternalShadersArtemis(ShaderRegistry* shaderRegistry)
    {
#define REG_SHADER(name, path, stage) shaderRegistry->RegisterShader(name, path, "main", stage)
#define REG_COMPUTE(name, path) REG_SHADER(name, path ".comp.glsl", RHI::RhiShaderStage::Compute)
#define REG_VERTEX(name, path) REG_SHADER(name, path ".vert.glsl", RHI::RhiShaderStage::Vertex)
#define REG_FRAGMENT(name, path) REG_SHADER(name, path ".frag.glsl", RHI::RhiShaderStage::Fragment)
#define REG_MESH(name, path) REG_SHADER(name, path ".mesh.glsl", RHI::RhiShaderStage::Mesh)

#define REG_SHADER_NEO(name, path, stage, entry) shaderRegistry->RegisterShader(name, path, entry, stage)
#define REG_COMPUTE_NEO(name, path, entry) REG_SHADER_NEO(name, path ".comp.slang", RHI::RhiShaderStage::Compute, entry)
#define REG_VERTEX_NEO(name, path, entry) REG_SHADER_NEO(name, path ".vert.slang", RHI::RhiShaderStage::Vertex, entry)
#define REG_FRAGMENT_NEO(name, path, entry) \
    REG_SHADER_NEO(name, path ".frag.slang", RHI::RhiShaderStage::Fragment, entry)
#define REG_MESH_NEO(name, path, entry) REG_SHADER_NEO(name, path ".mesh.slang", RHI::RhiShaderStage::Mesh, entry)

        const auto& ISTArtemis = kIntShaderTableArtemis;

        // PBD
        REG_COMPUTE_NEO(ISTArtemis.PBDClothApplyCorrectionCS, "Artemis/PBD/PBDCloth.ApplyCorrection",
            "ArtemisPBDClothApplyCorrectionCS");
        REG_COMPUTE_NEO(ISTArtemis.PBDClothUpdateVelocityPostCS, "Artemis/PBD/PBDCloth.UpdateVelocityPost",
            "ArtemisPBDClothUpdateVelocityPostCS");
        REG_COMPUTE_NEO(ISTArtemis.PBDClothUpdateVelocityPreCS, "Artemis/PBD/PBDCloth.UpdateVelocityPre",
            "ArtemisPBDClothUpdateVelocityPreCS");
        REG_COMPUTE_NEO(ISTArtemis.PBDClothPredPositionGenCS, "Artemis/PBD/PBDCloth.PredPositionGen",
            "ArtemisPBDClothPredPositionGenCS");
        REG_COMPUTE_NEO(ISTArtemis.PBDClothDistanceConstraintProjectCS,
            "Artemis/PBD/PBDCloth.DistanceConstraintProject", "ArtemisPBDClothDistanceConstraintProjectCS");
        REG_COMPUTE_NEO(ISTArtemis.PBDClothBendingConstraintProjectCS, "Artemis/PBD/PBDCloth.BendingConstraintProject",
            "ArtemisPBDClothBendingConstraintProjectCS");
        REG_COMPUTE_NEO(ISTArtemis.PBDPredPositionGenCS, "Artemis/PBD/PBDCloth.PredPositionGen",
            "ArtemisPBDClothPredPositionGenCS");
        REG_COMPUTE_NEO(
            ISTArtemis.PBDClothNormalUpdateCS, "Artemis/PBD/PBDCloth.NormalUpdate", "ArtemisPBDClothNormalUpdateCS");
        REG_COMPUTE_NEO(ISTArtemis.PBDClothNormalRegularizeCS, "Artemis/PBD/PBDCloth.NormalRegularize",
            "ArtemisPBDClothNormalRegularizeCS");
        REG_COMPUTE_NEO(ISTArtemis.PBDClothGenerateSDFCollisionCS, "Artemis/PBD/PBDCloth.GenerateSDFCollision",
            "ArtemisPBDClothGenerateSDFCollisionCS");
        REG_COMPUTE_NEO(ISTArtemis.PBDClothCollisionConstraintProject,
            "Artemis/PBD/PBDCloth.CollisionConstraintProject", "ArtemisPBDClothCollisionConstraintProjectCS");
        REG_COMPUTE_NEO(ISTArtemis.PBDClothUpdateVelocityCollisionCS, "Artemis/PBD/PBDCloth.UpdateVelocityCollision",
            "ArtemisPBDClothUpdateVelocityCollisionCS");
        REG_COMPUTE_NEO(ISTArtemis.PBDClothVolumeConstraintProjectCS, "Artemis/PBD/PBDCloth.VolumeConstraintProject",
            "ArtemisPBDClothVolumeConstraintProjectCS");

        // Particle Render
        REG_VERTEX_NEO(ISTArtemis.ParticleRender2dVS, "Artemis/ParticleRender2D", "ArtemisParticleRender2DVS");
        REG_FRAGMENT_NEO(ISTArtemis.ParticleRender2dFS, "Artemis/ParticleRender2D", "ArtemisParticleRender2DPS");
        REG_VERTEX_NEO(ISTArtemis.ParticleRender3dVS, "Artemis/ParticleRender3D", "ArtemisParticleRender3DVS");
        REG_FRAGMENT_NEO(ISTArtemis.ParticleRender3dFS, "Artemis/ParticleRender3D", "ArtemisParticleRender3DPS");
        REG_COMPUTE_NEO(ISTArtemis.ParticleIndDrawBufferPrepCS, "Artemis/ParticleIndDrawBufferPrep",
            "ArtemisParticleIndirectDrawPrepCS");

        // APIC
        REG_COMPUTE_NEO(ISTArtemis.APICFluidG2PCS, "Artemis/APIC/APICFluid.G2P", "ApicGridToParticleCS");
        REG_COMPUTE_NEO(ISTArtemis.APICFluidGridResetCS, "Artemis/APIC/APICFluid.GridReset", "ApicGridResetCS");
        REG_COMPUTE_NEO(ISTArtemis.APICFluidGridProjectionApplyCS, "Artemis/APIC/APICFluid.GridProjectionApply",
            "ApicGridProjectionApplyCS");
        REG_COMPUTE_NEO(ISTArtemis.APICFluidGridProjectionSolveCS, "Artemis/APIC/APICFluid.GridProjectionSolve",
            "ApicGridProjectionSolveCS");
        REG_COMPUTE_NEO(ISTArtemis.APICFluidGridProjectionSolveVelPrecomputeCS,
            "Artemis/APIC/APICFluid.GridProjectionSolveVelPrecompute", "ApicGridProjectionSolveVelPrecomputeCS");
        REG_COMPUTE_NEO(ISTArtemis.APICFluidP2GCS, "Artemis/APIC/APICFluid.P2G", "ApicParticleToGridCS");
        REG_COMPUTE_NEO(ISTArtemis.APICFluidGridUpdateCS, "Artemis/APIC/APICFluid.GridUpdate", "ApicGridUpdateCS");
        REG_COMPUTE_NEO(
            ISTArtemis.APICFluidParticleInitCS, "Artemis/APIC/APICFluid.ParticleInit", "ApicParticleInitCS");
        REG_COMPUTE_NEO(
            ISTArtemis.APICFluidParticleUpdateCS, "Artemis/APIC/APICFluid.ParticleUpdate", "ApicParticleUpdateCS");

        // MPM
        REG_COMPUTE_NEO(ISTArtemis.MPMG2PCS, "Artemis/MPM/MPM.G2P", "MpmGridToParticleCS");
        REG_COMPUTE_NEO(ISTArtemis.MPMGridResetCS, "Artemis/MPM/MPM.GridReset", "MpmGridResetCS");
        REG_COMPUTE_NEO(ISTArtemis.MPMGridForceUpdateCS, "Artemis/MPM/MPM.GridForceUpdate", "MpmGridForceUpdateCS");
        REG_COMPUTE_NEO(
            ISTArtemis.MPMGridVelocityUpdateCS, "Artemis/MPM/MPM.GridVelocityUpdate", "MpmGridVelocityUpdateCS");
        REG_COMPUTE_NEO(ISTArtemis.MPMGridRegularizeCS, "Artemis/MPM/MPM.GridRegularize", "MpmGridRegularizeCS");
        REG_COMPUTE_NEO(ISTArtemis.MPMP2GCS, "Artemis/MPM/MPM.P2G", "MpmParticleToGridCS");
        REG_COMPUTE_NEO(
            ISTArtemis.MPMParticleAdvectionCS, "Artemis/MPM/MPM.ParticleAdvection", "MpmParticleAdvectionCS");
        REG_COMPUTE_NEO(ISTArtemis.MPMParticleInitCS, "Artemis/MPM/MPM.ParticleInit", "MpmParticleInitCS");
        REG_COMPUTE_NEO(ISTArtemis.MPMGridGravityApplyCS, "Artemis/MPM/MPM.GridGravityApply", "MpmGridGravityApplyCS");
        REG_COMPUTE_NEO(ISTArtemis.MPMPbMpmResolveConstraintsCS, "Artemis/MPM/MPM.PbMpmResolveConstraints",
            "PbMpmResolveConstraintsCS");
        REG_COMPUTE_NEO(ISTArtemis.MPMPbMpmParticleIntegrateCS, "Artemis/MPM/MPM.PbMpmParticleIntegrate",
            "PbMpmParticleIntegrateCS");
        REG_COMPUTE_NEO(ISTArtemis.MPMParticleEmitCS, "Artemis/MPM/MPM.ParticleEmit", "MpmParticleEmitCS");
        REG_COMPUTE_NEO(ISTArtemis.MPMParticleDrainAllCS, "Artemis/MPM/MPM.ParticleDrainAll", "MpmParticleDrainAllCS");

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