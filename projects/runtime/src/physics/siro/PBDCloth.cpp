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

#include "ifrit/runtime/physics/siro/PBDCloth.h"
#include "ifrit/runtime/base/Mesh.h"
#include "ifrit/core/math/Intrinsics.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/physics/internal/InternalShaderRegistry.Siro.h"

#include "ifrit.shader.neo/Siro/PBDCloth.Shared.hlsli"

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
        f32 m_Stiffness  = 0.5f;
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

    struct PBDClothPrivateData
    {
        Vec<PBDClothDistanceConstraint> m_DistanceConstraints;
        Vec<PBDClothBendingConstraint>  m_BendingConstraints;
        Vec<f32>                        m_InverseMass;
        HashSet<u32>                    m_FixedParticles;

        RhiBufferRef                    m_ParticleExternalForces;
        RhiBufferRef                    m_ParticlePredPositions;
        RhiBufferRef                    m_ParticleVelocities;
        RhiBufferRef                    m_ParticleCorrections;
        RhiBufferRef                    m_ParticleFixed;
        RhiBufferRef                    m_ParticleInverseMass;

        RhiBufferRef                    m_GPUDistanceConstraints;
        RhiBufferRef                    m_GPUBendingConstraints;

        FGBufferNodeRef                 m_RDGParticleExternalForces;
        FGBufferNodeRef                 m_RDGParticlePredPositions;
        FGBufferNodeRef                 m_RDGParticleVelocities;
        FGBufferNodeRef                 m_RDGParticleCorrections;
        FGBufferNodeRef                 m_RDGParticlePositions;
        FGBufferNodeRef                 m_RDGParticleFixed;
        FGBufferNodeRef                 m_RDGParticleInverseMass;
        FGBufferNodeRef                 m_RDGDistanceConstraints;
        FGBufferNodeRef                 m_RDGBendingConstraints;

        u32                             m_NumParticles     = 0;
        u32                             m_NumIndices       = 0;
        bool                            m_ResourcePrepared = false;
        u32                             m_SolverIterations = 10;
        f32                             m_DefaultGravityY  = -5e-2f;
        f32                             m_VelocityDamping  = 0.999f; // Damping factor for velocity updates
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
                    bendingConstraint.m_Stiffness = 0.5f;

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

    IFRIT_APIDECL void PBDCloth::PrepareRDGResources(FrameGraphBuilder& builder)
    {
        if (!m_Data->m_ResourcePrepared)
        {
            BuildConstraints();
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

            m_Data->m_ResourcePrepared = true;
            auto rhi                   = builder.GetRhi();
            auto v4fSize               = SizeCast<u32>(m_Data->m_NumParticles * sizeof(Vector4f));
            auto v1fSize               = SizeCast<u32>(m_Data->m_NumParticles * sizeof(f32));
            auto usage                 = RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst;
            m_Data->m_ParticleCorrections = rhi->CreateBuffer("PBDCloth.Corrections", v4fSize, usage, false, true);
            m_Data->m_ParticleExternalForces =
                rhi->CreateBuffer("PBDCloth.ExternalForces", v4fSize, usage, false, true);
            m_Data->m_ParticlePredPositions = rhi->CreateBuffer("PBDCloth.PredPositions", v4fSize, usage, false, true);
            m_Data->m_ParticleVelocities    = rhi->CreateBuffer("PBDCloth.Velocities", v4fSize, usage, false, true);
            m_Data->m_ParticleFixed         = rhi->CreateBuffer("PBDCloth.FixedParticles", v1fSize, usage, false, true);
            m_Data->m_ParticleInverseMass   = rhi->CreateBuffer("PBDCloth.InverseMass", v1fSize, usage, false, true);

            auto distanceConstraintSize =
                SizeCast<u32>(m_Data->m_DistanceConstraints.size() * sizeof(PBDClothDistanceConstraint));
            auto bendingConstraintSize =
                SizeCast<u32>(m_Data->m_BendingConstraints.size() * sizeof(PBDClothBendingConstraint));
            m_Data->m_GPUDistanceConstraints =
                rhi->CreateBuffer("PBDCloth.DistanceConstraints", distanceConstraintSize, usage, false, true);
            m_Data->m_GPUBendingConstraints =
                rhi->CreateBuffer("PBDCloth.BendingConstraints", bendingConstraintSize, usage, false, true);

            // Launch a immediate command to upload data (not good)
            auto tq                       = rhi->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
            auto stagedDistanceConstraint = rhi->CreateStagedSingleBuffer(m_Data->m_GPUDistanceConstraints.get());
            auto stagedBendingConstraint  = rhi->CreateStagedSingleBuffer(m_Data->m_GPUBendingConstraints.get());
            auto stagedFixedParticles     = rhi->CreateStagedSingleBuffer(m_Data->m_ParticleFixed.get());
            auto stagedInverseMass        = rhi->CreateStagedSingleBuffer(m_Data->m_ParticleInverseMass.get());

            tq->RunSyncCommand([&](const RhiCommandList* cmd) {
                stagedDistanceConstraint->CmdCopyToDevice(
                    cmd, m_Data->m_DistanceConstraints.data(), distanceConstraintSize, 0);
                stagedBendingConstraint->CmdCopyToDevice(
                    cmd, m_Data->m_BendingConstraints.data(), bendingConstraintSize, 0);
                stagedFixedParticles->CmdCopyToDevice(cmd, fixedParticles.data(), v1fSize, 0);
                stagedInverseMass->CmdCopyToDevice(cmd, m_Data->m_InverseMass.data(), v1fSize, 0);
            });

            m_Data->m_ResourcePrepared = true;
        }
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

        auto meshFilter = GetParentUnsafe()->GetComponent<MeshFilter>();
        iAssertion(
            meshFilter != nullptr, "Siro.PBDCloth: PBDCloth requires a MeshFilter component on the parent GameObject");
        auto meshObject = meshFilter->GetMesh();
        iAssertion(meshObject != nullptr, "Siro.PBDCloth: MeshFilter has no mesh data");

        auto vertexBufferDevice = meshObject->m_resource.vertexBuffer;
        auto normalBufferDevice = meshObject->m_resource.normalBuffer;

        iAssertion(vertexBufferDevice != nullptr, "Siro.PBDCloth: MeshFilter's mesh has no vertex buffer");
        iAssertion(normalBufferDevice != nullptr, "Siro.PBDCloth: MeshFilter's mesh has no normal buffer");

        m_Data->m_RDGParticlePositions = &builder.ImportBuffer("PBDCloth.Positions", vertexBufferDevice.get());
    }

    IFRIT_APIDECL void PBDCloth::RunApproximationStep(FrameGraphBuilder& builder, f32 deltaTime)
    {
        PrepareRDGResources(builder);
        UpdateVelocityPre(builder, deltaTime);
        GeneratePredictedPosition(builder, deltaTime);
        ProjectConstraints(builder, m_Data->m_SolverIterations);
        UpdateVelocityPost(builder, deltaTime);
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

    IFRIT_APIDECL void PBDCloth::ProjectConstraints(FrameGraphBuilder& builder, u32 numIterations)
    {
        for (u32 i = 0; i < numIterations; ++i)
        {
            ProjectConstraintsDistance(builder, numIterations);
            ProjectConstraintsBending(builder, numIterations);
            ApplyCorrections(builder);
        }
    }

    IFRIT_APIDECL void PBDCloth::ProjectConstraintsDistance(FrameGraphBuilder& builder, u32 numIterations)
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
        } pc;

        pc.m_PredPositions       = 0;
        pc.m_Corrections         = 0;
        pc.m_InverseMass         = 0;
        pc.m_DistanceConstraints = 0;
        pc.m_FixedState          = 0;
        pc.m_NumConstraints      = SizeCast<u32>(m_Data->m_DistanceConstraints.size());
        pc.m_InvSolverIters      = 1.0f / f32(numIterations);

        i32   tgX = DivRoundUp(pc.m_NumConstraints, IfritShader::Siro::kSiroTGSizeX);

        auto& pass = AddComputePass<PushConst>(builder, "PBDCloth.ProjectConstraintsDistance",
            ShaderVariantDesc(kIntShaderTableSiro.PBDClothDistanceConstraintProjectCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_PredPositions       = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePredPositions);
                pc.m_Corrections         = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCorrections);
                pc.m_InverseMass         = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticleInverseMass);
                pc.m_DistanceConstraints = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGDistanceConstraints);
                pc.m_FixedState          = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticleFixed);

                SetRootConstant(pc, ctx);
            })
                         .AddWriteResource(*m_Data->m_RDGParticleCorrections)
                         .AddReadResource(*m_Data->m_RDGParticlePredPositions)
                         .AddReadResource(*m_Data->m_RDGParticleInverseMass)
                         .AddReadResource(*m_Data->m_RDGDistanceConstraints)
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
                pc.m_InverseMass        = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticleInverseMass);
                pc.m_BendingConstraints = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGBendingConstraints);
                pc.m_FixedState         = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticleFixed);

                SetRootConstant(pc, ctx);
            })
                         .AddWriteResource(*m_Data->m_RDGParticleCorrections)
                         .AddReadResource(*m_Data->m_RDGParticlePredPositions)
                         .AddReadResource(*m_Data->m_RDGParticleInverseMass)
                         .AddReadResource(*m_Data->m_RDGBendingConstraints)
                         .AddReadResource(*m_Data->m_RDGParticleFixed);
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
                pc.m_ExternalForces = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticleExternalForces);
                pc.m_InverseMass    = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticleInverseMass);
                pc.m_Velocities     = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleVelocities);
                pc.m_FixedState     = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticleFixed);
                pc.m_PredPositions  = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticlePredPositions);

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
        } pc;

        pc.m_GravityFactor = Vector4f(0.0f, m_Data->m_DefaultGravityY, 0.0f, 0.0f);
        pc.m_Positions     = 0;
        pc.m_Velocities    = 0;
        pc.m_FixedState    = 0;
        pc.m_PredPositions = 0;
        pc.m_DeltaTime     = deltaTime;
        pc.m_NumParticles  = m_Data->m_NumParticles;

        i32   tgX = DivRoundUp(pc.m_NumParticles, IfritShader::Siro::kSiroTGSizeX);

        auto& pass = AddComputePass<PushConst>(builder, "PBDCloth.UpdateVelocityPost",
            ShaderVariantDesc(kIntShaderTableSiro.PBDClothUpdateVelocityPostCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_Positions     = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticlePositions);
                pc.m_Velocities    = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleVelocities);
                pc.m_FixedState    = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticleFixed);
                pc.m_PredPositions = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticlePredPositions);

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
                pc.m_CurrentPositions = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticlePositions);
                pc.m_Corrections      = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticleCorrections);
                pc.m_InverseMass      = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticleInverseMass);
                pc.m_Velocities       = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticleVelocities);
                pc.m_FixedState       = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticleFixed);

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
                pc.m_Corrections   = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticleCorrections);
                pc.m_FixedState    = ctx.m_FgDesc->GetSRV(*m_Data->m_RDGParticleFixed);

                SetRootConstant(pc, ctx);
            })
                         .AddWriteResource(*m_Data->m_RDGParticlePredPositions)
                         .AddReadWriteResource(*m_Data->m_RDGParticleCorrections)
                         .AddReadResource(*m_Data->m_RDGParticleFixed);
    }

} // namespace Ifrit::Runtime::Siro