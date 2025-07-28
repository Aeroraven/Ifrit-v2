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
#include "ifrit/runtime/base/Component.h"
#include "ifrit/runtime/physics/artemis/ArtemisIntegrator.h"

#include "ifrit/runtime/renderer/ayanami/AyanamiMeshDF.h"

namespace Ifrit::Runtime::Artemis
{

    enum class EPBDClothSimulationType : u8
    {
        FlatCloth,
        Volume,
    };

    enum class EPBDSimulatorAlgorithm : u8
    {
        TrivialPBD,  // PBD
        ExtendedPBD, // XPBD
    };

    struct PBDClothPrivateData;

    // PBDCloth is a component that simulates cloth physics using Position Based Dynamics (PBD).
    // This component relies on MeshFilter to provide the mesh data for the cloth simulation.
    // It does not support dynamic mesh topology changes.
    class IFRIT_RUNTIME_API IF_CLASS() PBDCloth : public Component, public IArtemisSolver
    {
    public:
        IF_PROPERTY()
        f32 mDefaultBendingStiffness = 0.5f;

        IF_PROPERTY()
        f32 mDefaultStretchingStiffness = 0.5f;

    private:
        PBDClothPrivateData* m_Data;
        void                 BuildConstraints();
        void                 BuildConstraintsVolume();
        void                 Initialize();
        void                 PrepareRDGResources(FrameGraphBuilder& builder);
        void                 PrepareColliders(FrameGraphBuilder& builder);

        void                 ProjectConstraints(FrameGraphBuilder& builder, u32 numIterations, f32 deltaTime);
        void                 ProjectConstraintsDistance(FrameGraphBuilder& builder, u32 numIterations, f32 deltaTime);
        void                 ProjectConstraintsBending(FrameGraphBuilder& builder, u32 numIterations);
        void                 ProjectConstraintsVolume(FrameGraphBuilder& builder, u32 numIterations);

        void                 ProjectConstraintsCollision(FrameGraphBuilder& builder);
        void                 ApplyCorrections(FrameGraphBuilder& builder);
        void                 UpdateVelocityPre(FrameGraphBuilder& builder, f32 deltaTime);
        void                 UpdateVelocityPost(FrameGraphBuilder& builder, f32 deltaTime);
        void                 GeneratePredictedPosition(FrameGraphBuilder& builder, f32 deltaTime);
        void                 UpdateNormals(FrameGraphBuilder& builder);

        void                 ResetLambdas(FrameGraphBuilder& builder);

        void                 GenerateCollisionConstraints(FrameGraphBuilder& builder);
        void                 UpdateVelocityCollision(FrameGraphBuilder& builder);

        void                 CollectScene(Scene* scene) override;

    public:
        PBDCloth() { Initialize(); }
        PBDCloth(GameObject* parent) : Component(parent) { Initialize(); }
        virtual ~PBDCloth();

        inline void  SetupProperties() override {}

        virtual void RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime) override;

        void         AddFixedParticles(Vec<u32> fixedParticles);
        void         AddCollider(Ayanami::AyanamiMeshDF* collider);

        void         SetType(EPBDClothSimulationType type);
        void         SetSimulationAlgorithm(EPBDSimulatorAlgorithm algorithm);
    };

} // namespace Ifrit::Runtime::Artemis

IFRIT_COMPONENT_REGISTER(Ifrit::Runtime::Artemis::PBDCloth);