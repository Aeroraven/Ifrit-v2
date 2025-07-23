
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
#define DECLARE_FS(name) name "/PS"
#define DECLARE_CS(name) name "/CS"
#define DECLARE_MS(name) name "/MS"

#define SDEF IF_CONSTEXPR static const char*
    IFRIT_RUNTIME_API void RegisterRuntimeInternalShadersArtemis(ShaderRegistry* shaderRegistry);

    static struct InternalShaderTableArtemis
    {
        // PBD
        SDEF PBDClothApplyCorrectionCS           = DECLARE_CS("Artemis/PBDCloth.ApplyCorrection");
        SDEF PBDClothUpdateVelocityPostCS        = DECLARE_CS("Artemis/PBDCloth.UpdateVelocityPost");
        SDEF PBDClothUpdateVelocityPreCS         = DECLARE_CS("Artemis/PBDCloth.UpdateVelocityPre");
        SDEF PBDClothPredPositionGenCS           = DECLARE_CS("Artemis/PBDCloth.PredPositionGen");
        SDEF PBDClothDistanceConstraintProjectCS = DECLARE_CS("Artemis/PBDCloth.DistanceConstraintProject");
        SDEF PBDClothBendingConstraintProjectCS  = DECLARE_CS("Artemis/PBDCloth.BendingConstraintProject");
        SDEF PBDPredPositionGenCS                = DECLARE_CS("Artemis/PBD.PredPositionGen");
        SDEF PBDClothNormalUpdateCS              = DECLARE_CS("Artemis/PBDCloth.NormalUpdate");
        SDEF PBDClothNormalRegularizeCS          = DECLARE_CS("Artemis/PBDCloth.NormalRegularize");

        SDEF PBDClothGenerateSDFCollisionCS     = DECLARE_CS("Artemis/PBDCloth.GenerateSDFCollision");
        SDEF PBDClothCollisionConstraintProject = DECLARE_CS("Artemis/PBDCloth.CollisionConstraintProject");
        SDEF PBDClothUpdateVelocityCollisionCS  = DECLARE_CS("Artemis/PBDCloth.UpdateVelocityCollision");

        SDEF PBDClothVolumeConstraintProjectCS = DECLARE_CS("Artemis/PBDCloth.VolumeConstraintProject");

        // Particle Render
        SDEF ParticleRender2dVS          = DECLARE_VS("Artemis/ParticleRender2D");
        SDEF ParticleRender2dFS          = DECLARE_FS("Artemis/ParticleRender2D");
        SDEF ParticleRender3dVS          = DECLARE_VS("Artemis/ParticleRender3D");
        SDEF ParticleRender3dFS          = DECLARE_FS("Artemis/ParticleRender3D");
        SDEF ParticleIndDrawBufferPrepCS = DECLARE_CS("Artemis/ParticleRender.IndirectDrawBufferPrep");

        // APIC
        SDEF APICFluidG2PCS                 = DECLARE_CS("Artemis/APICFluid.G2P");
        SDEF APICFluidGridResetCS           = DECLARE_CS("Artemis/APICFluid.GridReset");
        SDEF APICFluidGridProjectionApplyCS = DECLARE_CS("Artemis/APICFluid.GridProjectionApply");
        SDEF APICFluidGridProjectionSolveCS = DECLARE_CS("Artemis/APICFluid.GridProjectionSolve");
        SDEF APICFluidGridProjectionSolveVelPrecomputeCS =
            DECLARE_CS("Artemis/APICFluid.GridProjectionSolveVelPrecompute");
        SDEF APICFluidGridUpdateCS     = DECLARE_CS("Artemis/APICFluid.GridUpdate");
        SDEF APICFluidP2GCS            = DECLARE_CS("Artemis/APICFluid.P2G");
        SDEF APICFluidParticleInitCS   = DECLARE_CS("Artemis/APICFluid.ParticleInit");
        SDEF APICFluidParticleUpdateCS = DECLARE_CS("Artemis/APICFluid.ParticleUpdate");

        // MPM
        SDEF MPMG2PCS                = DECLARE_CS("Artemis/MPM/MPM.G2P");
        SDEF MPMGridResetCS          = DECLARE_CS("Artemis/MPM/MPM.GridReset");
        SDEF MPMGridForceUpdateCS    = DECLARE_CS("Artemis/MPM/MPM.GridForceUpdate");
        SDEF MPMGridVelocityUpdateCS = DECLARE_CS("Artemis/MPM/MPM.GridVelocityUpdate");
        SDEF MPMGridRegularizeCS     = DECLARE_CS("Artemis/MPM/MPM.GridRegularize");
        SDEF MPMP2GCS                = DECLARE_CS("Artemis/MPM/MPM.P2G");
        SDEF MPMParticleAdvectionCS  = DECLARE_CS("Artemis/MPM/MPM.ParticleAdvection");
        SDEF MPMParticleInitCS       = DECLARE_CS("Artemis/MPM/MPM.ParticleInit");
        SDEF MPMGridGravityApplyCS   = DECLARE_CS("Artemis/MPM/MPM.GridGravityApply");
        SDEF MPMParticleEmitCS       = DECLARE_CS("Artemis/MPM/MPM.ParticleEmit");
        SDEF MPMParticleDrainAllCS   = DECLARE_CS("Artemis/MPM/MPM.ParticleDrainAll");

        // MPM Rigid Coupling
        SDEF MPMRigidContactConstraintResolveCS =
            DECLARE_CS("Artemis/MPM/RigidCoupling/MPMRigid.ContactConstraintResolve");
        SDEF MPMRigidResetCollisionPairCounterCS =
            DECLARE_CS("Artemis/MPM/RigidCoupling/MPMRigid.ResetCollisionPairCounter");
        SDEF MPMRigidCollectCollisionPairsCS  = DECLARE_CS("Artemis/MPM/RigidCoupling/MPMRigid.CollectCollisionPairs");
        SDEF MPMRigidCollectBoundaryContactCS = DECLARE_CS("Artemis/MPM/RigidCoupling/MPMRigid.CollectBoundaryContact");
        SDEF MPMRigidBoundaryConstraintResolveCS =
            DECLARE_CS("Artemis/MPM/RigidCoupling/MPMRigid.BoundaryConstraintResolve");

        // PBMPM
        SDEF MPMPbMpmResolveConstraintsCS = DECLARE_CS("Artemis/MPM/MPM.PbMpmResolveConstraints");
        SDEF MPMPbMpmParticleIntegrateCS  = DECLARE_CS("Artemis/MPM/MPM.PbMpmParticleIntegrate");

        // Rigid
        SDEF RigidMotionTestCS      = DECLARE_CS("Artemis/Rigid/MotionTest");
        SDEF RigidPostStateUpdateCS = DECLARE_CS("Artemis/Rigid/PostStateUpdate");
        SDEF RigidSyncTransformCS   = DECLARE_CS("Artemis/Rigid/SyncTransform");
        SDEF RigidLoadTransformCS   = DECLARE_CS("Artemis/Rigid/LoadTransform");

    } kIntShaderTableArtemis;

#undef SDEF

#undef DECLARE_VS
#undef DECLARE_FS
#undef DECLARE_CS
#undef DECLARE_MS

} // namespace Ifrit::Runtime::Internal