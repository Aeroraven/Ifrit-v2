
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

#include "ifrit/runtime/physics/internal/InternalShaderRegistry.Siro.h"

namespace Ifrit::Runtime::Internal
{
    IFRIT_APIDECL void RegisterRuntimeInternalShadersSiro(ShaderRegistry* shaderRegistry)
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

        const auto& ISTSiro = kIntShaderTableSiro;

        // PBD
        REG_COMPUTE_NEO(
            ISTSiro.PBDClothApplyCorrectionCS, "Siro/PBD/PBDCloth.ApplyCorrection", "SiroPBDClothApplyCorrectionCS");
        REG_COMPUTE_NEO(ISTSiro.PBDClothUpdateVelocityPostCS, "Siro/PBD/PBDCloth.UpdateVelocityPost",
            "SiroPBDClothUpdateVelocityPostCS");
        REG_COMPUTE_NEO(ISTSiro.PBDClothUpdateVelocityPreCS, "Siro/PBD/PBDCloth.UpdateVelocityPre",
            "SiroPBDClothUpdateVelocityPreCS");
        REG_COMPUTE_NEO(
            ISTSiro.PBDClothPredPositionGenCS, "Siro/PBD/PBDCloth.PredPositionGen", "SiroPBDClothPredPositionGenCS");
        REG_COMPUTE_NEO(ISTSiro.PBDClothDistanceConstraintProjectCS, "Siro/PBD/PBDCloth.DistanceConstraintProject",
            "SiroPBDClothDistanceConstraintProjectCS");
        REG_COMPUTE_NEO(ISTSiro.PBDClothBendingConstraintProjectCS, "Siro/PBD/PBDCloth.BendingConstraintProject",
            "SiroPBDClothBendingConstraintProjectCS");
        REG_COMPUTE_NEO(
            ISTSiro.PBDPredPositionGenCS, "Siro/PBD/PBDCloth.PredPositionGen", "SiroPBDClothPredPositionGenCS");
        REG_COMPUTE_NEO(ISTSiro.PBDClothNormalUpdateCS, "Siro/PBD/PBDCloth.NormalUpdate", "SiroPBDClothNormalUpdateCS");
        REG_COMPUTE_NEO(
            ISTSiro.PBDClothNormalRegularizeCS, "Siro/PBD/PBDCloth.NormalRegularize", "SiroPBDClothNormalRegularizeCS");
        REG_COMPUTE_NEO(ISTSiro.PBDClothGenerateSDFCollisionCS, "Siro/PBD/PBDCloth.GenerateSDFCollision",
            "SiroPBDClothGenerateSDFCollisionCS");
        REG_COMPUTE_NEO(ISTSiro.PBDClothCollisionConstraintProject, "Siro/PBD/PBDCloth.CollisionConstraintProject",
            "SiroPBDClothCollisionConstraintProjectCS");
        REG_COMPUTE_NEO(ISTSiro.PBDClothUpdateVelocityCollisionCS, "Siro/PBD/PBDCloth.UpdateVelocityCollision",
            "SiroPBDClothUpdateVelocityCollisionCS");
        REG_COMPUTE_NEO(ISTSiro.PBDClothVolumeConstraintProjectCS, "Siro/PBD/PBDCloth.VolumeConstraintProject",
            "SiroPBDClothVolumeConstraintProjectCS");

        REG_VERTEX_NEO(ISTSiro.ParticleRender2dVS, "Siro/ParticleRender2D", "SiroParticleRender2DVS");
        REG_FRAGMENT_NEO(ISTSiro.ParticleRender2dFS, "Siro/ParticleRender2D", "SiroParticleRender2DPS");
        REG_VERTEX_NEO(ISTSiro.ParticleRender3dVS, "Siro/ParticleRender3D", "SiroParticleRender3DVS");
        REG_FRAGMENT_NEO(ISTSiro.ParticleRender3dFS, "Siro/ParticleRender3D", "SiroParticleRender3DPS");

        // APIC
        REG_COMPUTE_NEO(ISTSiro.APICFluidG2PCS, "Siro/APIC/APICFluid.G2P", "ApicGridToParticleCS");
        REG_COMPUTE_NEO(ISTSiro.APICFluidGridResetCS, "Siro/APIC/APICFluid.GridReset", "ApicGridResetCS");
        REG_COMPUTE_NEO(ISTSiro.APICFluidGridProjectionApplyCS, "Siro/APIC/APICFluid.GridProjectionApply",
            "ApicGridProjectionApplyCS");
        REG_COMPUTE_NEO(ISTSiro.APICFluidGridProjectionSolveCS, "Siro/APIC/APICFluid.GridProjectionSolve",
            "ApicGridProjectionSolveCS");
        REG_COMPUTE_NEO(ISTSiro.APICFluidGridProjectionSolveVelPrecomputeCS,
            "Siro/APIC/APICFluid.GridProjectionSolveVelPrecompute", "ApicGridProjectionSolveVelPrecomputeCS");
        REG_COMPUTE_NEO(ISTSiro.APICFluidP2GCS, "Siro/APIC/APICFluid.P2G", "ApicParticleToGridCS");
        REG_COMPUTE_NEO(ISTSiro.APICFluidGridUpdateCS, "Siro/APIC/APICFluid.GridUpdate", "ApicGridUpdateCS");
        REG_COMPUTE_NEO(ISTSiro.APICFluidParticleInitCS, "Siro/APIC/APICFluid.ParticleInit", "ApicParticleInitCS");
        REG_COMPUTE_NEO(
            ISTSiro.APICFluidParticleUpdateCS, "Siro/APIC/APICFluid.ParticleUpdate", "ApicParticleUpdateCS");

        // MPM
        REG_COMPUTE_NEO(ISTSiro.MPMG2PCS, "Siro/MPM/MPM.G2P", "MpmGridToParticleCS");
        REG_COMPUTE_NEO(ISTSiro.MPMGridResetCS, "Siro/MPM/MPM.GridReset", "MpmGridResetCS");
        REG_COMPUTE_NEO(ISTSiro.MPMGridForceUpdateCS, "Siro/MPM/MPM.GridForceUpdate", "MpmGridForceUpdateCS");
        REG_COMPUTE_NEO(ISTSiro.MPMGridVelocityUpdateCS, "Siro/MPM/MPM.GridVelocityUpdate", "MpmGridVelocityUpdateCS");
        REG_COMPUTE_NEO(ISTSiro.MPMGridRegularizeCS, "Siro/MPM/MPM.GridRegularize", "MpmGridRegularizeCS");
        REG_COMPUTE_NEO(ISTSiro.MPMP2GCS, "Siro/MPM/MPM.P2G", "MpmParticleToGridCS");
        REG_COMPUTE_NEO(ISTSiro.MPMParticleAdvectionCS, "Siro/MPM/MPM.ParticleAdvection", "MpmParticleAdvectionCS");
        REG_COMPUTE_NEO(ISTSiro.MPMParticleInitCS, "Siro/MPM/MPM.ParticleInit", "MpmParticleInitCS");
        REG_COMPUTE_NEO(ISTSiro.MPMGridGravityApplyCS, "Siro/MPM/MPM.GridGravityApply", "MpmGridGravityApplyCS");

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