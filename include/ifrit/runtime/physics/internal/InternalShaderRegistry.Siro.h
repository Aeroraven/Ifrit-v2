
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
    IFRIT_RUNTIME_API void RegisterRuntimeInternalShadersSiro(ShaderRegistry* shaderRegistry);

    static struct InternalShaderTableSiro
    {

        SDEF PBDClothApplyCorrectionCS           = DECLARE_CS("Siro/PBDCloth.ApplyCorrection");
        SDEF PBDClothUpdateVelocityPostCS        = DECLARE_CS("Siro/PBDCloth.UpdateVelocityPost");
        SDEF PBDClothUpdateVelocityPreCS         = DECLARE_CS("Siro/PBDCloth.UpdateVelocityPre");
        SDEF PBDClothPredPositionGenCS           = DECLARE_CS("Siro/PBDCloth.PredPositionGen");
        SDEF PBDClothDistanceConstraintProjectCS = DECLARE_CS("Siro/PBDCloth.DistanceConstraintProject");
        SDEF PBDClothBendingConstraintProjectCS  = DECLARE_CS("Siro/PBDCloth.BendingConstraintProject");
        SDEF PBDPredPositionGenCS                = DECLARE_CS("Siro/PBD.PredPositionGen");
        SDEF PBDClothNormalUpdateCS              = DECLARE_CS("Siro/PBDCloth.NormalUpdate");
        SDEF PBDClothNormalRegularizeCS          = DECLARE_CS("Siro/PBDCloth.NormalRegularize");

        SDEF PBDClothGenerateSDFCollisionCS     = DECLARE_CS("Siro/PBDCloth.GenerateSDFCollision");
        SDEF PBDClothCollisionConstraintProject = DECLARE_CS("Siro/PBDCloth.CollisionConstraintProject");
        SDEF PBDClothUpdateVelocityCollisionCS  = DECLARE_CS("Siro/PBDCloth.UpdateVelocityCollision");

        SDEF PBDClothVolumeConstraintProjectCS = DECLARE_CS("Siro/PBDCloth.VolumeConstraintProject");

        SDEF ParticleRender2dVS = DECLARE_VS("Siro/ParticleRender2D");
        SDEF ParticleRender2dFS = DECLARE_FS("Siro/ParticleRender2D");
        SDEF ParticleRender3dVS = DECLARE_VS("Siro/ParticleRender3D");
        SDEF ParticleRender3dFS = DECLARE_FS("Siro/ParticleRender3D");

        SDEF APICFluidG2PCS                 = DECLARE_CS("Siro/APICFluid.G2P");
        SDEF APICFluidGridResetCS           = DECLARE_CS("Siro/APICFluid.GridReset");
        SDEF APICFluidGridProjectionApplyCS = DECLARE_CS("Siro/APICFluid.GridProjectionApply");
        SDEF APICFluidGridProjectionSolveCS = DECLARE_CS("Siro/APICFluid.GridProjectionSolve");
        SDEF APICFluidGridProjectionSolveVelPrecomputeCS =
            DECLARE_CS("Siro/APICFluid.GridProjectionSolveVelPrecompute");
        SDEF APICFluidGridUpdateCS     = DECLARE_CS("Siro/APICFluid.GridUpdate");
        SDEF APICFluidP2GCS            = DECLARE_CS("Siro/APICFluid.P2G");
        SDEF APICFluidParticleInitCS   = DECLARE_CS("Siro/APICFluid.ParticleInit");
        SDEF APICFluidParticleUpdateCS = DECLARE_CS("Siro/APICFluid.ParticleUpdate");

        SDEF MPMG2PCS                = DECLARE_CS("Siro/MPM/MPM.G2P");
        SDEF MPMGridResetCS          = DECLARE_CS("Siro/MPM/MPM.GridReset");
        SDEF MPMGridForceUpdateCS    = DECLARE_CS("Siro/MPM/MPM.GridForceUpdate");
        SDEF MPMGridVelocityUpdateCS = DECLARE_CS("Siro/MPM/MPM.GridVelocityUpdate");
        SDEF MPMGridRegularizeCS     = DECLARE_CS("Siro/MPM/MPM.GridRegularize");
        SDEF MPMP2GCS                = DECLARE_CS("Siro/MPM/MPM.P2G");
        SDEF MPMParticleAdvectionCS  = DECLARE_CS("Siro/MPM/MPM.ParticleAdvection");
        SDEF MPMParticleInitCS       = DECLARE_CS("Siro/MPM/MPM.ParticleInit");
        SDEF MPMGridGravityApplyCS   = DECLARE_CS("Siro/MPM/MPM.GridGravityApply");

        // PBMPM
        SDEF MPMPbMpmResolveConstraintsCS = DECLARE_CS("Siro/MPM/MPM.PbMpmResolveConstraints");
        SDEF MPMPbMpmParticleIntegrateCS  = DECLARE_CS("Siro/MPM/MPM.PbMpmParticleIntegrate");

    } kIntShaderTableSiro;

#undef SDEF

#undef DECLARE_VS
#undef DECLARE_FS
#undef DECLARE_CS
#undef DECLARE_MS

} // namespace Ifrit::Runtime::Internal