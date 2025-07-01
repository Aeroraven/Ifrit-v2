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

#include "ifrit/runtime/physics/siro/pbd/PBDCloth.h"
#include "ifrit/runtime/base/Mesh.h"
#include "ifrit/core/math/Intrinsics.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/physics/internal/InternalShaderRegistry.Siro.h"

#include "ifrit.shader.neo/Siro/PBD/PBDCloth.Shared.hlsli"
#include "ifrit/runtime/physics/siro/geometry/TetrahedralMesh.h"

using namespace Ifrit::Math;
using namespace Ifrit::Graphics::Rhi;
using namespace Ifrit::Runtime::FrameGraphUtils;
using namespace Ifrit::Runtime::Internal;

namespace Ifrit::Runtime::Siro
{

    struct PBDClothDistanceConstraint
    {
        u32 m_ParticleA  = 0;
        u32 m_ParticleB  = 0;
        f32 m_RestLength = 0.0f;
        f32 m_Stiffness  = 0.85f;
    };

    struct PBDClothBendingConstraint
    {
        u32 m_ParticleA = 0;
        u32 m_ParticleB = 0;
        u32 m_ParticleC = 0;
        u32 m_ParticleD = 0;
        f32 m_RestAngle = 0.0f;
        f32 m_Stiffness = 0.5f;
    };

    struct FPBDCollsionConstraint
    {
        Vector4f m_CollisionPos;
        Vector4f m_CollisionNormal;
        Vector4f m_CollisionVelocity;
        u32      m_ParticleA;
    };

    struct FPBDClothVolumeConstraint
    {
        u32 m_ParticleA  = 0;
        u32 m_ParticleB  = 0;
        u32 m_ParticleC  = 0;
        u32 m_ParticleD  = 0;
        f32 m_RestVolume = 0;
        f32 m_Stiffness  = 0.85f;
    };

    struct FColliderData
    {
        u32 m_SdfId;
        u32 m_ColliderMeshDataId;
    };

    struct PBDClothPrivateData
    {
        Vec<PBDClothDistanceConstraint> m_DistanceConstraints;
        Vec<PBDClothBendingConstraint>  m_BendingConstraints;
        Vec<FPBDClothVolumeConstraint>  m_VolumeConstraints;
        Vec<f32>                        m_InverseMass;
        HashSet<u32>                    m_FixedParticles;
        Vec<Ayanami::AyanamiMeshDF*>    m_Colliders;
        bool                            m_ColliderStateChange = true;
        EPBDClothSimulationType         m_SimulationType      = EPBDClothSimulationType::FlatCloth;
        EPBDSimulatorAlgorithm          m_SimulationAlgorithm = EPBDSimulatorAlgorithm::TrivialPBD;

        RhiBufferRef                    m_ParticleExternalForces;
        RhiBufferRef                    m_ParticlePredPositions;
        RhiBufferRef                    m_ParticleVelocities;
        RhiBufferRef                    m_ParticleCorrections;
        RhiBufferRef                    m_ParticleFixed;
        RhiBufferRef                    m_ParticleInverseMass;
        RhiBufferRef                    m_ParticleCollisions;
        RhiBufferRef                    m_ParticleCollisionsCounter;

        RhiBufferRef                    m_GPUDistanceConstraints;
        RhiBufferRef                    m_GPUBendingConstraints;
        RhiBufferRef                    m_GPUVolumeConstraints;
        RhiBufferRef                    m_GPUColliderData;

        FGBufferNodeRef                 m_RDGParticleExternalForces;
        FGBufferNodeRef                 m_RDGParticlePredPositions;
        FGBufferNodeRef                 m_RDGParticleVelocities;
        FGBufferNodeRef                 m_RDGParticleCorrections;
        FGBufferNodeRef                 m_RDGParticlePositions;
        FGBufferNodeRef                 m_RDGParticleNormals;
        FGBufferNodeRef                 m_RDGParticleIndices;
        FGBufferNodeRef                 m_RDGParticleFixed;
        FGBufferNodeRef                 m_RDGParticleInverseMass;
        FGBufferNodeRef                 m_RDGDistanceConstraints;
        FGBufferNodeRef                 m_RDGBendingConstraints;
        FGBufferNodeRef                 m_RDGVolumeConstraints;

        FGBufferNodeRef                 m_RDGParticleCollisions;
        FGBufferNodeRef                 m_RDGParticleCollisionsCounter;
        FGBufferNodeRef                 m_RDGColliderData;

        u32                             m_NumParticles           = 0;
        u32                             m_NumIndices             = 0;
        bool                            m_ResourcePrepared       = false;
        u32                             m_SolverIterations       = 200;
        f32                             m_DefaultGravityY        = -5e-2f;
        f32                             m_VelocityDamping        = 0.999f; // Damping factor for velocity updates
        u32                             m_MaxCollisionsPerVertex = 4;
        u32                             m_MaxColliders           = 128;

        // XPBD specific
        RhiBufferRef                    m_DistanceLambda;
        FGBufferNodeRef                 m_RDGDistanceLambda;
        f32                             m_DefaultCompliance = 0.000001f;
    };

    IFRIT_APIDECL PBDCloth::~PBDCloth()
    {
        if (m_Data)
        {
            delete m_Data;
            m_Data = nullptr;
        }
    }

    IFRIT_APIDECL void PBDCloth::Initialize() { m_Data = new PBDClothPrivateData(); }

    IFRIT_APIDECL void PBDCloth::SetType(EPBDClothSimulationType type) { m_Data->m_SimulationType = type; }

    IFRIT_APIDECL void PBDCloth::SetSimulationAlgorithm(EPBDSimulatorAlgorithm algorithm)
    {
        m_Data->m_SimulationAlgorithm = algorithm;
    }

    IFRIT_APIDECL void PBDCloth::BuildConstraints()
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

        // Bending constraints
        HashMap<u64, Vec<u32>> edgeToFaces;
        for (u32 i = 0; i < indexSize; i += 3)
        {
            u32 a = indexBuffer[i];
            u32 b = indexBuffer[i + 1];
            u32 c = indexBuffer[i + 2];

            if (a >= vertexSize || b >= vertexSize || c >= vertexSize)
            {
                continue;
            }

            auto addEdge = [&](u32 p1, u32 p2) {
                u64 key = OrderedPack32(p1, p2);
                edgeToFaces[key].push_back(i / 3);
            };

            addEdge(a, b);
            addEdge(b, c);
            addEdge(c, a);
        }
        for (const auto& edge : edgeToFaces)
        {
            const auto& faceIndices = edge.second;
            if (faceIndices.size() < 2)
            {
                continue;
            }

            u32 p1 = edge.first & 0xFFFFFFFF;
            u32 p2 = (edge.first >> 32) & 0xFFFFFFFF;

            // Create bending constraints for pairs of faces sharing the edge
            for (size_t i = 0; i < faceIndices.size(); ++i)
            {
                for (size_t j = i + 1; j < faceIndices.size(); ++j)
                {
                    u32 faceA = faceIndices[i];
                    u32 faceB = faceIndices[j];

                    // find the vertex in triangle that is not part of the edge
                    u32 vertexA = 0;
                    if (indexBuffer[faceA * 3] != p1 && indexBuffer[faceA * 3] != p2)
                    {
                        vertexA = indexBuffer[faceA * 3];
                    }
                    else if (indexBuffer[faceA * 3 + 1] != p1 && indexBuffer[faceA * 3 + 1] != p2)
                    {
                        vertexA = indexBuffer[faceA * 3 + 1];
                    }
                    else
                    {
                        vertexA = indexBuffer[faceA * 3 + 2];
                    }
                    u32 vertexB = 0;
                    if (indexBuffer[faceB * 3] != p1 && indexBuffer[faceB * 3] != p2)
                    {
                        vertexB = indexBuffer[faceB * 3];
                    }
                    else if (indexBuffer[faceB * 3 + 1] != p1 && indexBuffer[faceB * 3 + 1] != p2)
                    {
                        vertexB = indexBuffer[faceB * 3 + 1];
                    }
                    else
                    {
                        vertexB = indexBuffer[faceB * 3 + 2];
                    }

                    Vector3f                  p1Pos      = vertexBuffer[p1];
                    Vector3f                  p2Pos      = vertexBuffer[p2];
                    Vector3f                  vertexAPos = vertexBuffer[vertexA];
                    Vector3f                  vertexBPos = vertexBuffer[vertexB];

                    Vector3f                  p1Rel      = p1Pos - p1Pos;
                    Vector3f                  p2Rel      = p2Pos - p1Pos;
                    Vector3f                  vertexARel = vertexAPos - p1Pos;
                    Vector3f                  vertexBRel = vertexBPos - p1Pos;

                    Vector3f                  n1    = Normalize(Cross(p2Rel, vertexARel));
                    Vector3f                  n2    = Normalize(Cross(p2Rel, vertexBRel));
                    f32                       angle = acosf(Dot(n1, n2));

                    PBDClothBendingConstraint bendingConstraint;
                    bendingConstraint.m_ParticleA = p1;
                    bendingConstraint.m_ParticleB = p2;
                    bendingConstraint.m_ParticleC = vertexA;
                    bendingConstraint.m_ParticleD = vertexB;
                    bendingConstraint.m_RestAngle = angle;

                    // Add the constraint
                    m_Data->m_BendingConstraints.push_back(bendingConstraint);
                }
            }
        }

        // Create GPU resources for simulation
        m_Data->m_NumParticles = SizeCast<u32>(vertexBuffer.size());
        m_Data->m_NumIndices   = SizeCast<u32>(indexBuffer.size());

        for (auto i = 0u; i < m_Data->m_NumParticles; i++)
        {
            m_Data->m_InverseMass.push_back(1.0f);
        }
    }

    IFRIT_APIDECL void PBDCloth::BuildConstraintsVolume()
    {

        auto meshFilter = GetParentUnsafe()->GetComponent<MeshFilter>();
        iAssertion(
            meshFilter != nullptr, "Siro.PBDCloth: PBDCloth requires a MeshFilter component on the parent GameObject");

        auto meshObject = meshFilter->GetMesh();
        iAssertion(meshObject != nullptr, "Siro.PBDCloth: MeshFilter has no mesh data");

        auto meshData = meshObject->LoadMesh();
        iAssertion(meshData->m_MeshType == MeshType::Solid,
            "Siro.PBDCloth: Tetrahedral mesh is required for volume constraints");

        auto solidVertices     = meshObject->GetSolidMeshVertices();
        auto solidIndices      = meshObject->GetSolidMeshIndices();
        auto triangularIndices = meshObject->GetIndexBufferHost();

        // Volume constraint
        for (u32 i = 0; i < solidIndices.size(); i += 4)
        {
            auto                      iA = solidIndices[i];
            auto                      iB = solidIndices[i + 1];
            auto                      iC = solidIndices[i + 2];
            auto                      iD = solidIndices[i + 3];

            auto                      pA = solidVertices[iA];
            auto                      pB = solidVertices[iB];
            auto                      pC = solidVertices[iC];
            auto                      pD = solidVertices[iD];

            float                     volume = Dot(Cross(pB - pA, pC - pA), pD - pA) / 6.0f;

            FPBDClothVolumeConstraint volumeConstraint;
            volumeConstraint.m_ParticleA  = iA;
            volumeConstraint.m_ParticleB  = iB;
            volumeConstraint.m_ParticleC  = iC;
            volumeConstraint.m_ParticleD  = iD;
            volumeConstraint.m_RestVolume = volume;

            m_Data->m_VolumeConstraints.push_back(volumeConstraint);
        }

        // Distance constraint
        HashSet<u64> distanceConstraintMap;
        for (u32 i = 0; i < solidIndices.size(); i += 4)
        {
            auto          iA = solidIndices[i];
            auto          iB = solidIndices[i + 1];
            auto          iC = solidIndices[i + 2];
            auto          iD = solidIndices[i + 3];

            Array<u64, 6> edges = { OrderedPack32(iA, iB), OrderedPack32(iA, iC), OrderedPack32(iA, iD),
                OrderedPack32(iB, iC), OrderedPack32(iB, iD), OrderedPack32(iC, iD) };

            for (auto& e : edges)
            {
                if (distanceConstraintMap.find(e) == distanceConstraintMap.end())
                {
                    distanceConstraintMap.insert(e);
                    auto                       endPtA     = solidVertices[e & 0xFFFFFFFF];
                    auto                       endPtB     = solidVertices[(e >> 32) & 0xFFFFFFFF];
                    f32                        restLength = Length(endPtA - endPtB);

                    PBDClothDistanceConstraint distanceConstraint;
                    distanceConstraint.m_ParticleA  = e & 0xFFFFFFFF;
                    distanceConstraint.m_ParticleB  = (e >> 32) & 0xFFFFFFFF;
                    distanceConstraint.m_RestLength = restLength;

                    m_Data->m_DistanceConstraints.push_back(distanceConstraint);
                }
            }
        }
        m_Data->m_NumParticles = SizeCast<u32>(solidVertices.size());
        m_Data->m_NumIndices   = SizeCast<u32>(triangularIndices.size());

        for (auto i = 0u; i < m_Data->m_NumParticles; i++)
        {
            m_Data->m_InverseMass.push_back(1.0f);
        }
    }

    IFRIT_APIDECL void PBDCloth::PrepareRDGResources(FrameGraphBuilder& builder)
    {
        if (!m_Data->m_ResourcePrepared)
        {
            if (m_Data->m_SimulationType == EPBDClothSimulationType::FlatCloth)
            {
                BuildConstraints();
            }
            else if (m_Data->m_SimulationType == EPBDClothSimulationType::Volume)
            {
                BuildConstraintsVolume();
            }
            else
            {
                iError("Siro.PBDCloth: Unsupported simulation type");
                return;
            }
            Vec<u32> fixedParticles; // Can be compressed
            for (u32 i = 0; i < m_Data->m_NumParticles; ++i)
            {
                if (m_Data->m_FixedParticles.contains(i))
                {
                    fixedParticles.push_back(1);
                }
                else
                {
                    fixedParticles.push_back(0);
                }
            }

            auto rhi           = builder.GetRhi();
            auto v4fSize       = SizeCast<u32>(m_Data->m_NumParticles * sizeof(Vector4f));
            auto v1fSize       = SizeCast<u32>(m_Data->m_NumParticles * sizeof(f32));
            auto usage         = RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst;
            auto indirectUsage = RhiBufferUsage::RhiBufferUsage_Indirect | RhiBufferUsage::RhiBufferUsage_CopyDst
                | RhiBufferUsage::RhiBufferUsage_SSBO;
            auto colliderSize = SizeCast<u32>(m_Data->m_MaxColliders * sizeof(FColliderData));

            m_Data->m_ParticleCorrections = rhi->CreateBuffer("PBDCloth.Corrections", v4fSize, usage, false, true);
            m_Data->m_ParticleExternalForces =
                rhi->CreateBuffer("PBDCloth.ExternalForces", v4fSize, usage, false, true);
            m_Data->m_ParticlePredPositions = rhi->CreateBuffer("PBDCloth.PredPositions", v4fSize, usage, false, true);
            m_Data->m_ParticleVelocities    = rhi->CreateBuffer("PBDCloth.Velocities", v4fSize, usage, false, true);
            m_Data->m_ParticleFixed         = rhi->CreateBuffer("PBDCloth.FixedParticles", v1fSize, usage, false, true);
            m_Data->m_ParticleInverseMass   = rhi->CreateBuffer("PBDCloth.InverseMass", v1fSize, usage, false, true);
            m_Data->m_ParticleCollisionsCounter = rhi->CreateBuffer(
                "PBDCloth.CollisionCounter", SizeCast<u32>(4 * sizeof(u32)), indirectUsage, false, true);
            m_Data->m_GPUColliderData = rhi->CreateBuffer("PBDCloth.ColliderData", colliderSize, usage, false, true);

            auto distanceConstraintSize =
                SizeCast<u32>(m_Data->m_DistanceConstraints.size() * sizeof(PBDClothDistanceConstraint));
            auto bendingConstraintSize =
                SizeCast<u32>(m_Data->m_BendingConstraints.size() * sizeof(PBDClothBendingConstraint));
            auto collisionConstraintSize = SizeCast<u32>(
                m_Data->m_NumParticles * m_Data->m_MaxCollisionsPerVertex * sizeof(FPBDCollsionConstraint));
            auto volumeConstraintSize =
                SizeCast<u32>(m_Data->m_VolumeConstraints.size() * sizeof(FPBDClothVolumeConstraint));
            auto distanceLambdaSize = SizeCast<u32>(m_Data->m_NumParticles * sizeof(f32)); // For XPBD

            auto distanceConstraintSizeCR  = std::max(distanceConstraintSize, 1u);
            auto bendingConstraintSizeCR   = std::max(bendingConstraintSize, 1u);
            auto collisionConstraintSizeCR = std::max(collisionConstraintSize, 1u);
            auto volumeConstraintSizeCR    = std::max(volumeConstraintSize, 1u);
            auto distanceLambdaSizeCR      = std::max(distanceLambdaSize, 1u);

            m_Data->m_GPUDistanceConstraints =
                rhi->CreateBuffer("PBDCloth.DistanceConstraints", distanceConstraintSizeCR, usage, false, true);
            m_Data->m_GPUBendingConstraints =
                rhi->CreateBuffer("PBDCloth.BendingConstraints", bendingConstraintSizeCR, usage, false, true);
            m_Data->m_ParticleCollisions =
                rhi->CreateBuffer("PBDCloth.CollisionConstraints", collisionConstraintSizeCR, usage, false, true);
            m_Data->m_GPUVolumeConstraints =
                rhi->CreateBuffer("PBDCloth.VolumeConstraints", volumeConstraintSizeCR, usage, false, true);
            m_Data->m_DistanceLambda =
                rhi->CreateBuffer("PBDCloth.DistanceLambda", distanceLambdaSizeCR, usage, false, true);

            // Launch a immediate command to upload data (not good)
            auto tq                       = rhi->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
            auto stagedDistanceConstraint = rhi->CreateStagedSingleBuffer(m_Data->m_GPUDistanceConstraints.get());
            auto stagedBendingConstraint  = rhi->CreateStagedSingleBuffer(m_Data->m_GPUBendingConstraints.get());
            auto stagedFixedParticles     = rhi->CreateStagedSingleBuffer(m_Data->m_ParticleFixed.get());
            auto stagedInverseMass        = rhi->CreateStagedSingleBuffer(m_Data->m_ParticleInverseMass.get());
            auto stagedVolumeConstraint   = rhi->CreateStagedSingleBuffer(m_Data->m_GPUVolumeConstraints.get());

            tq->RunSyncCommand([&](const RhiCommandList* cmd) {
                stagedDistanceConstraint->CmdCopyToDevice(
                    cmd, m_Data->m_DistanceConstraints.data(), distanceConstraintSize, 0);
                if (bendingConstraintSize > 0)
                    stagedBendingConstraint->CmdCopyToDevice(
                        cmd, m_Data->m_BendingConstraints.data(), bendingConstraintSize, 0);
                stagedFixedParticles->CmdCopyToDevice(cmd, fixedParticles.data(), v1fSize, 0);
                stagedInverseMass->CmdCopyToDevice(cmd, m_Data->m_InverseMass.data(), v1fSize, 0);
                if (volumeConstraintSize > 0)
                    stagedVolumeConstraint->CmdCopyToDevice(
                        cmd, m_Data->m_VolumeConstraints.data(), volumeConstraintSize, 0);

                // clear velocity
            });

            // m_Data->m_ResourcePrepared = true;
        }
        PrepareColliders(builder);
        // todo
        m_Data->m_RDGParticleCorrections =
            &builder.ImportBuffer("PBDCloth.Corrections", m_Data->m_ParticleCorrections.get());
        m_Data->m_RDGParticleExternalForces =
            &builder.ImportBuffer("PBDCloth.ExternalForces", m_Data->m_ParticleExternalForces.get());
        m_Data->m_RDGParticlePredPositions =
            &builder.ImportBuffer("PBDCloth.PredPositions", m_Data->m_ParticlePredPositions.get());
        m_Data->m_RDGParticleVelocities =
            &builder.ImportBuffer("PBDCloth.Velocities", m_Data->m_ParticleVelocities.get());
        m_Data->m_RDGParticleFixed = &builder.ImportBuffer("PBDCloth.FixedParticles", m_Data->m_ParticleFixed.get());
        m_Data->m_RDGParticleInverseMass =
            &builder.ImportBuffer("PBDCloth.InverseMass", m_Data->m_ParticleInverseMass.get());

        m_Data->m_RDGDistanceConstraints =
            &builder.ImportBuffer("PBDCloth.DistanceConstraints", m_Data->m_GPUDistanceConstraints.get());
        m_Data->m_RDGBendingConstraints =
            &builder.ImportBuffer("PBDCloth.BendingConstraints", m_Data->m_GPUBendingConstraints.get());
        m_Data->m_RDGVolumeConstraints =
            &builder.ImportBuffer("PBDCloth.VolumeConstraints", m_Data->m_GPUVolumeConstraints.get());

        m_Data->m_RDGParticleCollisions =
            &builder.ImportBuffer("PBDCloth.CollisionConstraints", m_Data->m_ParticleCollisions.get());
        m_Data->m_RDGParticleCollisionsCounter =
            &builder.ImportBuffer("PBDCloth.CollisionCounter", m_Data->m_ParticleCollisionsCounter.get());
        m_Data->m_RDGColliderData = &builder.ImportBuffer("PBDCloth.ColliderData", m_Data->m_GPUColliderData.get());

        auto meshFilter = GetParentUnsafe()->GetComponent<MeshFilter>();
        iAssertion(
            meshFilter != nullptr, "Siro.PBDCloth: PBDCloth requires a MeshFilter component on the parent GameObject");
        auto meshObject = meshFilter->GetMesh();
        iAssertion(meshObject != nullptr, "Siro.PBDCloth: MeshFilter has no mesh data");

        auto vertexBufferDevice = meshObject->m_resource.vertexBuffer;
        auto normalBufferDevice = meshObject->m_resource.normalBuffer;
        auto indexBufferDevice  = meshObject->m_resource.indexBuffer;

        iAssertion(vertexBufferDevice != nullptr, "Siro.PBDCloth: MeshFilter's mesh has no vertex buffer");
        iAssertion(normalBufferDevice != nullptr, "Siro.PBDCloth: MeshFilter's mesh has no normal buffer");

        m_Data->m_RDGParticlePositions = &builder.ImportBuffer("PBDCloth.Positions", vertexBufferDevice.get());
        m_Data->m_RDGParticleNormals   = &builder.ImportBuffer("PBDCloth.Normals", normalBufferDevice.get());
        m_Data->m_RDGParticleIndices   = &builder.ImportBuffer("PBDCloth.Indices", indexBufferDevice.get());

        // XPBD specific
        m_Data->m_RDGDistanceLambda = &builder.ImportBuffer("PBDCloth.DistanceLambda", m_Data->m_DistanceLambda.get());

        if (!m_Data->m_ResourcePrepared)
        {
            AddClearUAVPass(builder, "PBDCloth.ResetVelocity", *m_Data->m_RDGParticleVelocities, 0);
        }
        m_Data->m_ResourcePrepared = true;
    }

    IFRIT_APIDECL void PBDCloth::PrepareColliders(FrameGraphBuilder& builder)
    {
        if (m_Data->m_ColliderStateChange && m_Data->m_Colliders.size() > 0)
        {
            m_Data->m_ColliderStateChange = false;
            auto rhi                      = builder.GetRhi();
            auto stagedColliderData       = rhi->CreateStagedSingleBuffer(m_Data->m_GPUColliderData.get());

            auto tq = rhi->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
            tq->RunSyncCommand([&](const RhiCommandList* cmd) {
                Vec<FColliderData> colliderData;
                for (const auto& collider : m_Data->m_Colliders)
                {
                    auto parent     = collider->GetParent();
                    auto meshFilter = parent->GetComponent<MeshFilter>();
                    iAssertion(meshFilter != nullptr,
                        "Siro.PBDCloth: Collider's parent GameObject must have a MeshFilter component");
                    iAssertion(collider != nullptr, "Siro.PBDCloth: Collider cannot be null");
                    FColliderData data;
                    data.m_SdfId              = collider->GetMetaBufferId();
                    data.m_ColliderMeshDataId = meshFilter->GetMesh()->m_resource.objectBuffer->GetDescId();
                    colliderData.push_back(data);
                }
                stagedColliderData->CmdCopyToDevice(
                    cmd, colliderData.data(), SizeCast<u32>(colliderData.size() * sizeof(FColliderData)), 0);
            });
        }
    }

    IFRIT_APIDECL void PBDCloth::RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime)
    {
        PrepareRDGResources(builder);
        UpdateVelocityPre(builder, deltaTime);
        GeneratePredictedPosition(builder, deltaTime);
        if (m_Data->m_SimulationAlgorithm == EPBDSimulatorAlgorithm::TrivialPBD)
        {
            GenerateCollisionConstraints(builder);
            ProjectConstraints(builder, m_Data->m_SolverIterations, deltaTime);
            UpdateVelocityCollision(builder);
        }
        else if (m_Data->m_SimulationAlgorithm == EPBDSimulatorAlgorithm::ExtendedPBD)
        {
            ResetLambdas(builder);
            ProjectConstraints(builder, m_Data->m_SolverIterations, deltaTime);
        }
        UpdateVelocityPost(builder, deltaTime);
        UpdateNormals(builder);
    }

    IFRIT_APIDECL void PBDCloth::AddFixedParticles(Vec<u32> fixedParticles)
    {
        for (const auto& particle : fixedParticles)
        {
            if (m_Data->m_FixedParticles.find(particle) == m_Data->m_FixedParticles.end())
            {
                m_Data->m_FixedParticles.insert(particle);
            }
        }
    }

    IFRIT_APIDECL void PBDCloth::AddCollider(Ayanami::AyanamiMeshDF* collider)
    {
        iAssertion(collider != nullptr, "Siro.PBDCloth: Cannot add a null collider");
        m_Data->m_Colliders.push_back(collider);
    }

    IFRIT_APIDECL void PBDCloth::ProjectConstraints(FrameGraphBuilder& builder, u32 numIterations, f32 deltaTime)
    {
        if (m_Data->m_SimulationAlgorithm == EPBDSimulatorAlgorithm::TrivialPBD)
        {
            for (u32 i = 0; i < numIterations; ++i)
            {
                ProjectConstraintsDistance(builder, numIterations, deltaTime);
                if (m_Data->m_SimulationType == EPBDClothSimulationType::Volume)
                {
                    ProjectConstraintsVolume(builder, numIterations);
                }
                else if (m_Data->m_SimulationType == EPBDClothSimulationType::FlatCloth)
                {
                    ProjectConstraintsBending(builder, numIterations);
                }
                ProjectConstraintsCollision(builder);
                ApplyCorrections(builder);
            }
        }
        else if (m_Data->m_SimulationAlgorithm == EPBDSimulatorAlgorithm::ExtendedPBD)
        {
            for (u32 i = 0; i < numIterations; ++i)
            {
                ProjectConstraintsDistance(builder, numIterations, deltaTime);
                ApplyCorrections(builder);
            }
        }
    }

    IFRIT_APIDECL void PBDCloth::ProjectConstraintsDistance(
        FrameGraphBuilder& builder, u32 numIterations, f32 deltaTime)
    {
        struct PushConst
        {
            u32 m_PredPositions;
            u32 m_Corrections;
            u32 m_InverseMass;
            u32 m_DistanceConstraints;
            u32 m_FixedState;
            u32 m_NumConstraints;
            f32 m_InvSolverIters;

            // XPBD Params
            u32 m_LambdaId;
            f32 m_Compilance;
            f32 m_DeltaTime;
        } pc;

        pc.m_PredPositions       = 0;
        pc.m_Corrections         = 0;
        pc.m_InverseMass         = 0;
        pc.m_DistanceConstraints = 0;
        pc.m_FixedState          = 0;
        pc.m_NumConstraints      = SizeCast<u32>(m_Data->m_DistanceConstraints.size());
        pc.m_InvSolverIters      = 1.0f / f32(numIterations);

        pc.m_LambdaId   = 0;
        pc.m_Compilance = m_Data->m_DefaultCompliance;
        pc.m_DeltaTime  = deltaTime;

        i32         tgX = DivRoundUp(pc.m_NumConstraints, IfritShader::Siro::kSiroTGSizeX);

        Vec<String> shaderPerm;
        if (m_Data->m_SimulationAlgorithm == EPBDSimulatorAlgorithm::ExtendedPBD)
        {
            shaderPerm.push_back("IFSHADER_SIRO_XPBD");
        }

        auto& pass = AddComputePass<PushConst>(builder, "PBDCloth.ProjectConstraintsDistance",
            ShaderVariantDesc(kIntShaderTableSiro.PBDClothDistanceConstraintProjectCS, shaderPerm), Vector3i(tgX, 1, 1),
            pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_PredPositions       = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePredPositions);
                pc.m_Corrections         = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCorrections);
                pc.m_InverseMass         = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleInverseMass);
                pc.m_DistanceConstraints = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGDistanceConstraints);
                pc.m_FixedState          = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleFixed);
                pc.m_LambdaId            = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGDistanceLambda);

                SetRootConstant(pc, ctx);
            })
                         .AddWriteResource(*m_Data->m_RDGParticleCorrections)
                         .AddReadResource(*m_Data->m_RDGParticlePredPositions)
                         .AddReadResource(*m_Data->m_RDGParticleInverseMass)
                         .AddReadResource(*m_Data->m_RDGDistanceConstraints)
                         .AddReadWriteResource(*m_Data->m_RDGDistanceLambda)
                         .AddReadResource(*m_Data->m_RDGParticleFixed);
    }
    IFRIT_APIDECL void PBDCloth::ProjectConstraintsBending(FrameGraphBuilder& builder, u32 numIterations)
    {
        struct PushConst
        {
            u32 m_PredPositions;
            u32 m_Corrections;
            u32 m_InverseMass;
            u32 m_BendingConstraints;
            u32 m_FixedState;
            u32 m_NumConstraints;
            f32 m_InvSolverIters;
        } pc;

        pc.m_PredPositions      = 0;
        pc.m_Corrections        = 0;
        pc.m_InverseMass        = 0;
        pc.m_BendingConstraints = 0;
        pc.m_FixedState         = 0;
        pc.m_NumConstraints     = SizeCast<u32>(m_Data->m_BendingConstraints.size());
        pc.m_InvSolverIters     = 1.0f / f32(numIterations);

        i32   tgX  = DivRoundUp(pc.m_NumConstraints, IfritShader::Siro::kSiroTGSizeX);
        auto& pass = AddComputePass<PushConst>(builder, "PBDCloth.ProjectConstraintsBending",
            ShaderVariantDesc(kIntShaderTableSiro.PBDClothBendingConstraintProjectCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_PredPositions      = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePredPositions);
                pc.m_Corrections        = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCorrections);
                pc.m_InverseMass        = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleInverseMass);
                pc.m_BendingConstraints = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGBendingConstraints);
                pc.m_FixedState         = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleFixed);

                SetRootConstant(pc, ctx);
            })
                         .AddWriteResource(*m_Data->m_RDGParticleCorrections)
                         .AddReadResource(*m_Data->m_RDGParticlePredPositions)
                         .AddReadResource(*m_Data->m_RDGParticleInverseMass)
                         .AddReadResource(*m_Data->m_RDGBendingConstraints)
                         .AddReadResource(*m_Data->m_RDGParticleFixed);
    }

    IFRIT_APIDECL void PBDCloth::ProjectConstraintsVolume(FrameGraphBuilder& builder, u32 numIterations)
    {
        struct PushConst
        {
            u32 m_PredPositions;
            u32 m_Corrections;
            u32 m_InverseMass;
            u32 m_VolumeConstraints;
            u32 m_FixedState;
            u32 m_NumConstraints;
            f32 m_InvSolverIters;
        } pc;

        pc.m_PredPositions     = 0;
        pc.m_Corrections       = 0;
        pc.m_InverseMass       = 0;
        pc.m_VolumeConstraints = 0;
        pc.m_FixedState        = 0;
        pc.m_NumConstraints    = SizeCast<u32>(m_Data->m_VolumeConstraints.size());
        pc.m_InvSolverIters    = 1.0f / f32(numIterations);

        i32   tgX = DivRoundUp(pc.m_NumConstraints, IfritShader::Siro::kSiroTGSizeX);

        auto& pass = AddComputePass<PushConst>(builder, "PBDCloth.ProjectConstraintsVolume",
            ShaderVariantDesc(kIntShaderTableSiro.PBDClothVolumeConstraintProjectCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_PredPositions     = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePredPositions);
                pc.m_Corrections       = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCorrections);
                pc.m_InverseMass       = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleInverseMass);
                pc.m_VolumeConstraints = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGVolumeConstraints);
                pc.m_FixedState        = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleFixed);

                SetRootConstant(pc, ctx);
            })
                         .AddWriteResource(*m_Data->m_RDGParticleCorrections)
                         .AddReadResource(*m_Data->m_RDGParticlePredPositions)
                         .AddReadResource(*m_Data->m_RDGParticleInverseMass)
                         .AddReadResource(*m_Data->m_RDGVolumeConstraints)
                         .AddReadResource(*m_Data->m_RDGParticleFixed);
    }

    IFRIT_APIDECL void PBDCloth::ProjectConstraintsCollision(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_CollisionConstraintCounter;
            u32 m_CollisionConstraints;
            u32 m_PredPositions;
            u32 m_CorrectionHandle;
        } pc;

        pc.m_CollisionConstraintCounter = 0;
        pc.m_CollisionConstraints       = 0;
        pc.m_PredPositions              = 0;
        pc.m_CorrectionHandle           = 0;

        auto& pass = AddIndirectComputePass<PushConst>(builder, "PBDCloth.ProjectConstraintsCollision",
            ShaderVariantDesc(kIntShaderTableSiro.PBDClothCollisionConstraintProject, {}),
            *m_Data->m_RDGParticleCollisionsCounter, 4u, pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_CollisionConstraintCounter = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCollisionsCounter);
                pc.m_CollisionConstraints       = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCollisions);
                pc.m_PredPositions              = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePredPositions);
                pc.m_CorrectionHandle           = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCorrections);

                SetRootConstant(pc, ctx);
            })
                         .AddWriteResource(*m_Data->m_RDGParticleCorrections)
                         .AddReadResource(*m_Data->m_RDGParticlePredPositions)
                         .AddReadResource(*m_Data->m_RDGParticleCollisionsCounter)
                         .AddReadResource(*m_Data->m_RDGParticleCollisions);
    }

    IFRIT_APIDECL void PBDCloth::UpdateVelocityPre(FrameGraphBuilder& builder, f32 deltaTime)
    {
        struct PushConst
        {
            u32 m_ExternalForces;
            u32 m_InverseMass;
            u32 m_Velocities;
            u32 m_FixedState;
            u32 m_PredPositions;
            f32 m_DeltaTime;
            f32 m_DampingFactor;
            u32 m_NumParticles;

        } pc;

        pc.m_ExternalForces = 0;
        pc.m_InverseMass    = 0;
        pc.m_Velocities     = 0;
        pc.m_FixedState     = 0;
        pc.m_PredPositions  = 0;
        pc.m_DeltaTime      = deltaTime;
        pc.m_DampingFactor  = m_Data->m_VelocityDamping;
        pc.m_NumParticles   = m_Data->m_NumParticles;

        i32   tgX = DivRoundUp(pc.m_NumParticles, IfritShader::Siro::kSiroTGSizeX);

        auto& pass = AddComputePass<PushConst>(builder, "PBDCloth.UpdateVelocityPre",
            ShaderVariantDesc(kIntShaderTableSiro.PBDClothUpdateVelocityPreCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ExternalForces = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleExternalForces);
                pc.m_InverseMass    = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleInverseMass);
                pc.m_Velocities     = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleVelocities);
                pc.m_FixedState     = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleFixed);
                pc.m_PredPositions  = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePredPositions);

                SetRootConstant(pc, ctx);
            })
                         .AddWriteResource(*m_Data->m_RDGParticleVelocities)
                         .AddReadResource(*m_Data->m_RDGParticleExternalForces)
                         .AddReadResource(*m_Data->m_RDGParticleInverseMass)
                         .AddReadResource(*m_Data->m_RDGParticleFixed)
                         .AddReadResource(*m_Data->m_RDGParticlePredPositions);
    }
    IFRIT_APIDECL void PBDCloth::UpdateVelocityPost(FrameGraphBuilder& builder, f32 deltaTime)
    {
        struct PushConst
        {
            Vector4f m_GravityFactor;
            u32      m_Positions;
            u32      m_Velocities;
            u32      m_FixedState;
            u32      m_PredPositions;
            f32      m_DeltaTime;
            u32      m_NumParticles;
            u32      m_CollisionCountersId;
        } pc;

        pc.m_GravityFactor       = Vector4f(0.0f, m_Data->m_DefaultGravityY, 0.0f, 0.0f);
        pc.m_Positions           = 0;
        pc.m_Velocities          = 0;
        pc.m_FixedState          = 0;
        pc.m_PredPositions       = 0;
        pc.m_DeltaTime           = deltaTime;
        pc.m_NumParticles        = m_Data->m_NumParticles;
        pc.m_CollisionCountersId = 0;

        i32   tgX = DivRoundUp(pc.m_NumParticles, IfritShader::Siro::kSiroTGSizeX);

        auto& pass = AddComputePass<PushConst>(builder, "PBDCloth.UpdateVelocityPost",
            ShaderVariantDesc(kIntShaderTableSiro.PBDClothUpdateVelocityPostCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_Positions           = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePositions);
                pc.m_Velocities          = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleVelocities);
                pc.m_FixedState          = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleFixed);
                pc.m_PredPositions       = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePredPositions);
                pc.m_CollisionCountersId = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCollisionsCounter);

                SetRootConstant(pc, ctx);
            })
                         .AddWriteResource(*m_Data->m_RDGParticleVelocities)
                         .AddReadResource(*m_Data->m_RDGParticlePositions)
                         .AddReadResource(*m_Data->m_RDGParticleFixed)
                         .AddReadResource(*m_Data->m_RDGParticlePredPositions);
    }
    IFRIT_APIDECL void PBDCloth::GeneratePredictedPosition(FrameGraphBuilder& builder, f32 deltaTime)
    {
        struct PushConst
        {
            u32 m_PredPositions;
            u32 m_CurrentPositions;
            u32 m_Corrections;
            u32 m_InverseMass;
            u32 m_Velocities;
            u32 m_FixedState;
            f32 m_DeltaTime;
            u32 m_NumParticles;
        } pc;

        pc.m_PredPositions    = 0;
        pc.m_CurrentPositions = 0;
        pc.m_Corrections      = 0;
        pc.m_InverseMass      = 0;
        pc.m_Velocities       = 0;
        pc.m_FixedState       = 0;
        pc.m_DeltaTime        = deltaTime;
        pc.m_NumParticles     = m_Data->m_NumParticles;

        i32   tgX = DivRoundUp(pc.m_NumParticles, IfritShader::Siro::kSiroTGSizeX);

        auto& pass = AddComputePass<PushConst>(builder, "PBDCloth.GeneratePredictedPosition",
            ShaderVariantDesc(kIntShaderTableSiro.PBDClothPredPositionGenCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_PredPositions    = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePredPositions);
                pc.m_CurrentPositions = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePositions);
                pc.m_Corrections      = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCorrections);
                pc.m_InverseMass      = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleInverseMass);
                pc.m_Velocities       = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleVelocities);
                pc.m_FixedState       = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleFixed);

                SetRootConstant(pc, ctx);
            })
                         .AddWriteResource(*m_Data->m_RDGParticlePredPositions)
                         .AddWriteResource(*m_Data->m_RDGParticleCorrections)
                         .AddReadResource(*m_Data->m_RDGParticlePositions)
                         .AddReadResource(*m_Data->m_RDGParticleInverseMass)
                         .AddReadResource(*m_Data->m_RDGParticleVelocities)
                         .AddReadResource(*m_Data->m_RDGParticleFixed);
    }
    IFRIT_APIDECL void PBDCloth::ApplyCorrections(FrameGraphBuilder& builder)
    {

        struct PushConst
        {
            u32 m_PredPositions;
            u32 m_Corrections;
            u32 m_FixedState;
            u32 m_NumParticles;
        } pc;

        pc.m_PredPositions = 0;
        pc.m_Corrections   = 0;
        pc.m_FixedState    = 0;
        pc.m_NumParticles  = m_Data->m_NumParticles;

        i32   tgX = DivRoundUp(pc.m_NumParticles, IfritShader::Siro::kSiroTGSizeX);

        auto& pass = AddComputePass<PushConst>(builder, "PBDCloth.ApplyCorrections",
            ShaderVariantDesc(kIntShaderTableSiro.PBDClothApplyCorrectionCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_PredPositions = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePredPositions);
                pc.m_Corrections   = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCorrections);
                pc.m_FixedState    = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleFixed);

                SetRootConstant(pc, ctx);
            })
                         .AddWriteResource(*m_Data->m_RDGParticlePredPositions)
                         .AddReadWriteResource(*m_Data->m_RDGParticleCorrections)
                         .AddReadResource(*m_Data->m_RDGParticleFixed);
    }

    IFRIT_APIDECL void PBDCloth::UpdateNormals(FrameGraphBuilder& builder)
    {
        struct PushConst_1
        {
            u32 m_Normal;
            u32 m_Position;
            u32 m_Indices;
            u32 m_NumIndices;
        } pc1;

        pc1.m_Normal     = 0;
        pc1.m_Position   = 0;
        pc1.m_Indices    = 0;
        pc1.m_NumIndices = m_Data->m_NumIndices;

        i32   tgX = DivRoundUp(pc1.m_NumIndices, IfritShader::Siro::kSiroTGSizeX);

        auto& pass = AddComputePass<PushConst_1>(builder, "PBDCloth.UpdateNormals",
            ShaderVariantDesc(kIntShaderTableSiro.PBDClothNormalUpdateCS, {}), Vector3i(tgX, 1, 1), pc1,
            [this](PushConst_1 pc, const FrameGraphPassContext& ctx) {
                pc.m_Normal   = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleNormals);
                pc.m_Position = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePositions);
                pc.m_Indices  = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleIndices);

                SetRootConstant(pc, ctx);
            })
                         .AddWriteResource(*m_Data->m_RDGParticleNormals)
                         .AddReadResource(*m_Data->m_RDGParticlePositions)
                         .AddReadResource(*m_Data->m_RDGParticleIndices);

        struct PushConst_2
        {
            u32 m_Normal;
            u32 m_NumVertices;
        } pc2;

        pc2.m_Normal      = 0;
        pc2.m_NumVertices = m_Data->m_NumParticles;

        i32   tgX2 = DivRoundUp(pc2.m_NumVertices, IfritShader::Siro::kSiroTGSizeX);

        auto& pass2 = AddComputePass<PushConst_2>(builder, "PBDCloth.UpdateNormalsFinal",
            ShaderVariantDesc(kIntShaderTableSiro.PBDClothNormalRegularizeCS, {}), Vector3i(tgX2, 1, 1), pc2,
            [this](PushConst_2 pc, const FrameGraphPassContext& ctx) {
                pc.m_Normal = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleNormals);

                SetRootConstant(pc, ctx);
            }).AddWriteResource(*m_Data->m_RDGParticleNormals);
    }

    IFRIT_APIDECL void PBDCloth::ResetLambdas(FrameGraphBuilder& builder)
    {
        IF_CONSTEXPR auto DestVal = std::bit_cast<u32, f32>(0.0f);
        AddClearUAVPass(builder, "PBDCloth.ResetLambdas", *m_Data->m_RDGDistanceLambda, DestVal);
    }

    IFRIT_APIDECL void PBDCloth::GenerateCollisionConstraints(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_CollisionConstraintCounter;
            u32 m_ColliderData;
            u32 m_CollisionConstraints;
            u32 m_PredPositions;
            u32 m_Velocities;
            u32 m_Positions;
            u32 m_NumParticles;
            u32 m_NumSDFs;
        } pc;

        pc.m_CollisionConstraintCounter = 0;
        pc.m_ColliderData               = 0;
        pc.m_CollisionConstraints       = 0;
        pc.m_PredPositions              = 0;
        pc.m_Velocities                 = 0;
        pc.m_Positions                  = 0;
        pc.m_NumParticles               = m_Data->m_NumParticles;
        pc.m_NumSDFs                    = SizeCast<u32>(m_Data->m_Colliders.size());

        i32   tgX = DivRoundUp(pc.m_NumParticles, IfritShader::Siro::kSiroTGSizeX);

        auto& pass = AddComputePass<PushConst>(builder, "PBDCloth.GenerateCollisionConstraints",
            ShaderVariantDesc(kIntShaderTableSiro.PBDClothGenerateSDFCollisionCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_CollisionConstraintCounter = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCollisionsCounter);
                pc.m_ColliderData               = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGColliderData);
                pc.m_CollisionConstraints       = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCollisions);
                pc.m_PredPositions              = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePredPositions);
                pc.m_Velocities                 = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleVelocities);
                pc.m_Positions                  = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePositions);

                SetRootConstant(pc, ctx);
            })
                         .AddWriteResource(*m_Data->m_RDGParticleCollisionsCounter)
                         .AddWriteResource(*m_Data->m_RDGParticleCollisions)
                         .AddReadResource(*m_Data->m_RDGParticlePredPositions)
                         .AddReadResource(*m_Data->m_RDGParticleVelocities)
                         .AddReadResource(*m_Data->m_RDGColliderData)
                         .AddReadResource(*m_Data->m_RDGParticlePositions);
    }

    IFRIT_APIDECL void PBDCloth::UpdateVelocityCollision(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_Positions;
            u32 m_CollisionConstraints;
            u32 m_Velocities;
            u32 m_CollisionConstraintCounter;
        } pc;
        pc.m_Positions                  = 0;
        pc.m_CollisionConstraints       = 0;
        pc.m_Velocities                 = 0;
        pc.m_CollisionConstraintCounter = 0;

        auto& pass = AddIndirectComputePass<PushConst>(builder, "PBDCloth.UpdateVelocityCollision",
            ShaderVariantDesc(kIntShaderTableSiro.PBDClothUpdateVelocityCollisionCS, {}),
            *m_Data->m_RDGParticleCollisionsCounter, 4u, pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_Positions                  = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePositions);
                pc.m_CollisionConstraints       = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCollisions);
                pc.m_Velocities                 = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleVelocities);
                pc.m_CollisionConstraintCounter = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCollisionsCounter);

                SetRootConstant(pc, ctx);
            })
                         .AddWriteResource(*m_Data->m_RDGParticleVelocities)
                         .AddReadResource(*m_Data->m_RDGParticlePositions)
                         .AddReadResource(*m_Data->m_RDGParticleCollisions)
                         .AddReadResource(*m_Data->m_RDGParticleCollisionsCounter);
    }

} // namespace Ifrit::Runtime::Siro