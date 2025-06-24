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
#include "ifrit/runtime/physics/siro/SiroIntegrator.h"

#include "ifrit/runtime/renderer/ayanami/AyanamiMeshDF.h"

namespace Ifrit::Runtime::Siro
{
    struct XPBDClothAttribute
    {
        f32 m_DefaultBendingStiffness    = 0.5f;
        f32 m_DefaultStretchingStiffness = 0.5f;

        IFRIT_STRUCT_SERIALIZE(m_DefaultBendingStiffness, m_DefaultStretchingStiffness);
    };

    enum class EXPBDClothSimulationType : u8
    {
        FlatCloth,
        Volume,
    };

    struct XPBDClothPrivateData;

    // XPBDCloth is a component that simulates cloth physics using eXtended Position Based Dynamics (XPBD).
    // This component relies on MeshFilter to provide the mesh data for the cloth simulation.
    // It does not support dynamic mesh topology changes.
    class IFRIT_RUNTIME_API XPBDCloth :
        public Component,
        public AttributeOwner<XPBDClothAttribute>,
        public ISiroExplicitEulerSolver
    {
    private:
        XPBDClothPrivateData* m_Data = nullptr;
        void                  Initialize();
        void                  BuildConstraint();
        void                  PrepareRDGResources(FrameGraphBuilder& builder);

    public:
        XPBDCloth() { Initialize(); }
        XPBDCloth(Ref<GameObject> parent) : Component(parent), AttributeOwner<XPBDClothAttribute>() { Initialize(); }
        virtual ~XPBDCloth();

        String       Serialize() override { return SerializeAttribute(); }
        void         Deserialize() override { DeserializeAttribute(); }

        virtual void RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime) override;

        void         AddFixedParticles(Vec<u32> fixedParticles);
        void         AddCollider(Ayanami::AyanamiMeshDF* collider);

        void         SetType(EXPBDClothSimulationType type);
    };

} // namespace Ifrit::Runtime::Siro

IFRIT_COMPONENT_REGISTER(Ifrit::Runtime::Siro::XPBDCloth);