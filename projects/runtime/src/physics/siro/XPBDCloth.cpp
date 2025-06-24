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

#include "ifrit/runtime/physics/siro/XPBDCloth.h"
#include "ifrit/runtime/base/Mesh.h"
#include "ifrit/core/math/Intrinsics.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/physics/internal/InternalShaderRegistry.Siro.h"

using namespace Ifrit::Math;
using namespace Ifrit::Graphics::Rhi;
using namespace Ifrit::Runtime::FrameGraphUtils;
using namespace Ifrit::Runtime::Internal;

namespace Ifrit::Runtime::Siro
{
    struct FXPBDClothDistanceConstraint
    {
        u32 m_ParticleA  = 0;
        u32 m_ParticleB  = 0;
        f32 m_RestLength = 0.0f;
        f32 m_Stiffness  = 0.85f;
    };

    struct XPBDClothPrivateData
    {
        HashSet<u32>                      m_FixedParticles;
        Vec<f32>                          m_InverseMass;
        Vec<Ayanami::AyanamiMeshDF*>      m_Colliders;
        EXPBDClothSimulationType          m_SimulationType = EXPBDClothSimulationType::FlatCloth;
        Vec<FXPBDClothDistanceConstraint> m_DistanceConstraints;

        bool                              m_Initialized = false;

        RhiBufferRef                      m_GPUPredLocation;
        RhiBufferRef                      m_GPUVelocity;
        RhiBufferRef                      m_GPUExternalForce;
        RhiBufferRef                      m_GPUInverseMass;
        RhiBufferRef                      m_GPUFixedState;
        Array<RhiBufferRef, 2>            m_GPULambda;
        Array<RhiBufferRef, 2>            m_GPURefinedPredLocation;

        FGBufferNodeRef                   m_RDGPredLocation;
        FGBufferNodeRef                   m_RDGVelocity;
        FGBufferNodeRef                   m_RDGExternalForce;
        FGBufferNodeRef                   m_RDGInverseMass;
        FGBufferNodeRef                   m_RDGFixedState;
        Array<FGBufferNodeRef, 2>         m_RDGLambda;
        Array<FGBufferNodeRef, 2>         m_RDGRefinedPredLocation;

        FGBufferNodeRef                   m_RDGPosition;
        FGBufferNodeRef                   m_RDGNormal;
        FGBufferNodeRef                   m_RDGIndex;

        u32                               m_NumParticles;
        u32                               m_NumIndices;
    };

    IFRIT_APIDECL void XPBDCloth::Initialize() { m_Data = new XPBDClothPrivateData(); }

    IFRIT_APIDECL      XPBDCloth::~XPBDCloth()
    {
        if (m_Data)
            delete m_Data;
        m_Data = nullptr;
    }

    IFRIT_APIDECL void XPBDCloth::BuildConstraint()
    {
        auto meshFilter = GetParentUnsafe()->GetComponent<MeshFilter>();
        iAssertion(
            meshFilter != nullptr, "Siro.PBDCloth: PBDCloth requires a MeshFilter component on the parent GameObject");

        auto meshObject = meshFilter->GetMesh();
        iAssertion(meshObject != nullptr, "Siro.PBDCloth: MeshFilter has no mesh data");

        auto               vertexBuffer = meshObject->GetVertexBufferHost();
        auto               indexBuffer  = meshObject->GetIndexBufferHost();
        auto               vertexSize   = SizeCast<u32>(vertexBuffer.size());
        auto               indexSize    = SizeCast<u32>(indexBuffer.size());

        // Distance constraints
        HashMap<u64, bool> distanceConstraintMap;
        for (u32 i = 0; i < indexSize; i += 3)
        {
            u32 a = indexBuffer[i];
            u32 b = indexBuffer[i + 1];
            u32 c = indexBuffer[i + 2];

            if (a >= vertexSize || b >= vertexSize || c >= vertexSize)
            {
                continue; // Invalid index
            }

            // Add distance constraints for edges
            auto addDistanceConstraint = [&](u32 p1, u32 p2) {
                u64 key = OrderedPack32(p1, p2);
                if (distanceConstraintMap.find(key) == distanceConstraintMap.end())
                {
                    f32 restLength = Length(vertexBuffer[p1] - vertexBuffer[p2]);
                    m_Data->m_DistanceConstraints.push_back({ p1, p2, restLength, 0.5f });
                    distanceConstraintMap[key] = true;
                }
            };

            addDistanceConstraint(a, b);
            addDistanceConstraint(b, c);
            addDistanceConstraint(c, a);
        }

        // Create GPU resources for simulation
        m_Data->m_NumParticles = SizeCast<u32>(vertexBuffer.size());
        m_Data->m_NumIndices   = SizeCast<u32>(indexBuffer.size());

        for (auto i = 0u; i < m_Data->m_NumParticles; i++)
        {
            m_Data->m_InverseMass.push_back(1.0f);
        }
    }

    IFRIT_APIDECL void XPBDCloth::AddFixedParticles(Vec<u32> fixedParticles)
    {
        for (const auto& particle : fixedParticles)
        {
            if (m_Data->m_FixedParticles.find(particle) == m_Data->m_FixedParticles.end())
            {
                m_Data->m_FixedParticles.insert(particle);
            }
        }
    }

    IFRIT_APIDECL void XPBDCloth::AddCollider(Ayanami::AyanamiMeshDF* collider)
    {
        iAssertion(collider != nullptr, "Siro.XPBDCloth: Cannot add a null collider");
        m_Data->m_Colliders.push_back(collider);
    }

    IFRIT_APIDECL void XPBDCloth::PrepareRDGResources(FrameGraphBuilder& builder)
    {
        if (!m_Data->m_Initialized)
        {
            BuildConstraint();
            m_Data->m_Initialized = true;
        }
    }

    IFRIT_APIDECL void XPBDCloth::RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime)
    {
        // TODO
        PrepareRDGResources(builder);
    }

    IFRIT_APIDECL void XPBDCloth::SetType(EXPBDClothSimulationType type) { m_Data->m_SimulationType = type; }

} // namespace Ifrit::Runtime::Siro