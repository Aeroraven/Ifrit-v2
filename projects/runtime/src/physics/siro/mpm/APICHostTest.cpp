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
#include "ifrit/runtime/physics/siro/mpm/APICHostTest.h"
#include "ifrit/core/math/linalg/LinalgOps.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/physics/internal/InternalShaderRegistry.Siro.h"

#include <random>

using namespace Ifrit::Math;
using namespace Ifrit::Graphics::Rhi;
using namespace Ifrit::Runtime::FrameGraphUtils;

namespace Ifrit::Runtime::Siro
{
    struct APICParticle
    {
        Matrix2x2f m_C;
        Vector2f   m_Velocity;
        Vector2f   m_Position;
        f32        m_Mass;
    };

    struct APICGrid
    {
        Vector2f m_GridCenter;
        Vector2f m_GridSize;
        Vector2f m_Momentum;
        f32      m_Mass;
        Vector2f m_Velocity;
    };

    static f32 RandomUniform(f32 min, f32 max)
    {
        static std::mt19937              gen(std::random_device{}());
        std::uniform_real_distribution<> dis(min, max);
        return static_cast<f32>(dis(gen));
    }

    struct APICHostTestPrivateData
    {
        Vec<APICParticle>       m_Particles;
        Vec<APICGrid>           m_Grids;
        u32                     m_NumGridsX = 0;
        u32                     m_NumGridsY = 0;
        f32                     m_GridSizeX = 0.0f;
        f32                     m_GridSizeY = 0.0f;

        IF_CONSTEXPR static u32 kDefaultGridsPerDim  = 256;
        IF_CONSTEXPR static u32 kDefaultParticles    = 1024;
        IF_CONSTEXPR static f32 kDefaultGridSize     = 1.0f;
        IF_CONSTEXPR static f32 kDefaultParticleMass = 1.0f;
        IF_CONSTEXPR static f32 kDefaultTimeStep     = 0.008f;

        Vec<Vector2f>           m_GPUParticlePositionsData;
        RhiBufferRef            m_GPUParticlePositions = nullptr;
        RhiBufferRef            m_GPUIndexBuffer       = nullptr;
        FGBufferNodeRef         m_RDGParticlePositions = nullptr;

        bool                    m_ResourcePrepared = false;
    };

    u32 PositionToGridIndex(APICHostTestPrivateData* data, Vector2f position)
    {
        u32 x = static_cast<u32>(position.x / data->m_GridSizeX);
        u32 y = static_cast<u32>(position.y / data->m_GridSizeY);
        return x + y * data->m_NumGridsX;
    }

    f32 P2GQuadraticInterpolation(f32 x)
    {
        auto absX  = std::abs(x);
        auto absX2 = absX * absX;
        if (absX < 0.5f)
        {
            return 0.75f - absX2;
        }
        else if (absX < 1.5f)
        {
            return 0.5f * (1.5f - absX) * (1.5f - absX);
        }
        else
        {
            return 0.0f;
        }
    }

    IFRIT_APIDECL u32 APICHostTest::GridToIndex(u32 x, u32 y) { return x + y * m_Data->m_NumGridsX; }
    IFRIT_APIDECL Pair<u32, u32> APICHostTest::IndexToGrid(u32 index)
    {
        u32 x = index % m_Data->m_NumGridsX;
        u32 y = index / m_Data->m_NumGridsX;
        return Pair<u32, u32>(x, y);
    }

    IFRIT_APIDECL APICHostTest::APICHostTest() : m_Data(new APICHostTestPrivateData())
    {
        m_Data->m_Particles.resize(APICHostTestPrivateData::kDefaultParticles);
        m_Data->m_GPUParticlePositionsData.resize(APICHostTestPrivateData::kDefaultParticles);
        for (auto& p : m_Data->m_Particles)
        {
            p.m_C = ZeroMatrix<f32, 2, 2>();
            p.m_Velocity =
                Vector2f(RandomUniform(-1.0f, 1.0f), RandomUniform(-1.0f, 1.0f)) * 0.0f + Vector2f(0.0f, 0.00f);
            p.m_Position =
                Vector2f(RandomUniform(-1.0f, 1.0f), RandomUniform(-1.0f, 1.0f)) * 16.0f + Vector2f(32.0f, 32.0f);
            p.m_Mass = APICHostTestPrivateData::kDefaultParticleMass;
        }

        m_Data->m_NumGridsX = APICHostTestPrivateData::kDefaultGridsPerDim;
        m_Data->m_NumGridsY = APICHostTestPrivateData::kDefaultGridsPerDim;
        m_Data->m_GridSizeX = APICHostTestPrivateData::kDefaultGridSize;
        m_Data->m_GridSizeY = APICHostTestPrivateData::kDefaultGridSize;

        auto gridCount = m_Data->m_NumGridsX * m_Data->m_NumGridsY;
        m_Data->m_Grids.resize(gridCount);

        f32 totalGridWidth  = m_Data->m_NumGridsX * m_Data->m_GridSizeX;
        f32 totalGridHeight = m_Data->m_NumGridsY * m_Data->m_GridSizeY;

        for (u32 dx = 0; dx < m_Data->m_NumGridsX; ++dx)
        {
            for (u32 dy = 0; dy < m_Data->m_NumGridsY; ++dy)
            {
                auto& grid        = m_Data->m_Grids[GridToIndex(dx, dy)];
                grid.m_GridCenter = Vector2f((dx + 0.5f) * m_Data->m_GridSizeX, (dy + 0.5f) * m_Data->m_GridSizeY);
                grid.m_GridSize   = Vector2f(m_Data->m_GridSizeX, m_Data->m_GridSizeY);
                grid.m_Momentum   = Vector2f(0.0f, 0.0f);
                grid.m_Mass       = 0.0f;
                grid.m_Velocity   = 0.0f;
            }
        }
    }

    IFRIT_APIDECL      APICHostTest::~APICHostTest() { delete m_Data; }

    IFRIT_APIDECL void APICHostTest::TestAPIC(FrameGraphBuilder& builder, FGTextureNode* renderTarget)
    {
        // Simulation on CPU

        // Zero out grids
        for (auto& grid : m_Data->m_Grids)
        {
            grid.m_Momentum = Vector2f(0.0f);
            grid.m_Mass     = 0.0f;
            grid.m_Velocity = Vector2f(0.0f);
        }

        // P2G
        for (auto& p : m_Data->m_Particles)
        {
            auto centerGridIndex            = PositionToGridIndex(m_Data, p.m_Position);
            auto [centerGridX, centerGridY] = IndexToGrid(centerGridIndex);

            for (i32 dx = -1; dx <= 1; ++dx)
            {
                for (i32 dy = -1; dy <= 1; ++dy)
                {
                    i32      gridX     = (i32)centerGridX + dx;
                    i32      gridY     = (i32)centerGridY + dy;
                    u32      gridIndex = GridToIndex(gridX, gridY);
                    auto&    grid      = m_Data->m_Grids[gridIndex];

                    Vector2f posToGridCenter    = p.m_Position - grid.m_GridCenter;
                    Vector2f posToGridCenterNeg = grid.m_GridCenter - p.m_Position;

                    Vector2f posToGridCenterInGridSize = posToGridCenter / grid.m_GridSize;

                    // Quadratic interpolation
                    f32      contribWeightX = P2GQuadraticInterpolation(posToGridCenterInGridSize.x);
                    f32      contribWeightY = P2GQuadraticInterpolation(posToGridCenterInGridSize.y);
                    f32      contribWeight  = contribWeightX * contribWeightY;

                    Vector2f velocity = p.m_Velocity + MatMul(p.m_C, posToGridCenterNeg);
                    grid.m_Momentum += p.m_Mass * velocity * contribWeight;
                    grid.m_Mass += p.m_Mass * contribWeight;
                }
            }
        }

        // Grid Operations
        for (auto& grid : m_Data->m_Grids)
        {
            if (grid.m_Mass > 0.0f)
            {
                grid.m_Velocity = grid.m_Momentum / grid.m_Mass;
            }
            else
            {
                grid.m_Velocity = Vector2f(0.0f);
            }
            grid.m_Velocity += Vector2f(0.0f, 9.8f) * APICHostTestPrivateData::kDefaultTimeStep;

            auto gridIndex      = PositionToGridIndex(m_Data, grid.m_GridCenter);
            auto [gridX, gridY] = IndexToGrid(gridIndex);
            if (gridX <= 2 || gridX >= m_Data->m_NumGridsX - 2)
            {
                grid.m_Velocity.x = 0.0f;
            }
            if (gridY <= 2 || gridY >= m_Data->m_NumGridsY - 2)
            {
                grid.m_Velocity.y = 0.0f;
            }
        }
        // TODO: Projection

        // G2P
        for (auto& p : m_Data->m_Particles)
        {
            auto centerGridIndex            = PositionToGridIndex(m_Data, p.m_Position);
            auto [centerGridX, centerGridY] = IndexToGrid(centerGridIndex);

            Matrix2x2f B           = ZeroMatrix<f32, 2, 2>();
            Vector2f   newVelocity = Vector2f(0.0f);
            float      D           = 4.0f / (m_Data->m_GridSizeX * m_Data->m_GridSizeX);

            for (i32 dx = -1; dx <= 1; ++dx)
            {
                for (i32 dy = -1; dy <= 1; ++dy)
                {
                    i32      gridX     = (i32)centerGridX + dx;
                    i32      gridY     = (i32)centerGridY + dy;
                    u32      gridIndex = GridToIndex(gridX, gridY);
                    auto&    grid      = m_Data->m_Grids[gridIndex];

                    Vector2f posToGridCenter           = p.m_Position - grid.m_GridCenter;
                    Vector2f posToGridCenterInGridSize = posToGridCenter / grid.m_GridSize;

                    // Quadratic interpolation
                    f32      contribWeightX = P2GQuadraticInterpolation(posToGridCenterInGridSize.x);
                    f32      contribWeightY = P2GQuadraticInterpolation(posToGridCenterInGridSize.y);
                    f32      contribWeight  = contribWeightX * contribWeightY;

                    f32      gridVx = grid.m_Velocity.x;
                    f32      gridVy = grid.m_Velocity.y;
                    f32      deltaX = -posToGridCenterInGridSize.x;
                    f32      deltaY = -posToGridCenterInGridSize.y;

                    B[0][0] += gridVx * deltaX * contribWeight * D;
                    B[0][1] += gridVx * deltaY * contribWeight * D;
                    B[1][0] += gridVy * deltaX * contribWeight * D;
                    B[1][1] += gridVy * deltaY * contribWeight * D;

                    newVelocity.x += gridVx * contribWeight;
                    newVelocity.y += gridVy * contribWeight;
                }
            }
            p.m_Velocity = newVelocity;
            p.m_C        = B;
        }

        // Particle advection
        for (auto& p : m_Data->m_Particles)
        {
            p.m_Position += p.m_Velocity * APICHostTestPrivateData::kDefaultTimeStep;
            p.m_Position = Clamp(p.m_Position, 2.0f, m_Data->m_NumGridsX * m_Data->m_GridSizeX - 2.0f);
        }
        for (auto i = 0; i < m_Data->m_Particles.size(); ++i)
        {
            m_Data->m_GPUParticlePositionsData[i] = m_Data->m_Particles[i].m_Position;
        }

        // Prepare GPU resources
        if (!m_Data->m_ResourcePrepared)
        {
            auto rhi                       = builder.GetRhi();
            m_Data->m_GPUParticlePositions = rhi->CreateBuffer("APICHostTest.ParticlePositions",
                SizeCast<u32>(m_Data->m_GPUParticlePositionsData.size() * sizeof(Vector2f)),
                RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst, false, true);
            m_Data->m_ResourcePrepared     = true;

            Vec<u32> indexData;
            indexData.resize(m_Data->m_Particles.size());
            for (u32 i = 0; i < m_Data->m_Particles.size(); ++i)
            {
                indexData[i] = i;
            }
            m_Data->m_GPUIndexBuffer =
                rhi->CreateBuffer("APICHostTest.ParticleIndexBuffer", SizeCast<u32>(indexData.size() * sizeof(u32)),
                    RhiBufferUsage::RhiBufferUsage_Index | RhiBufferUsage::RhiBufferUsage_CopyDst, false, false);

            auto tq            = rhi->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
            auto stagingBuffer = rhi->CreateStagedSingleBuffer(m_Data->m_GPUIndexBuffer.get());
            tq->RunSyncCommand([&](const RhiCommandList* cmd) {
                stagingBuffer->CmdCopyToDevice(cmd, indexData.data(), SizeCast<u32>(indexData.size() * sizeof(u32)), 0);
            });
        }
        auto rhi           = builder.GetRhi();
        auto tq            = rhi->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
        auto stagingBuffer = rhi->CreateStagedSingleBuffer(m_Data->m_GPUParticlePositions.get());
        tq->RunSyncCommand([&](const RhiCommandList* cmd) {
            stagingBuffer->CmdCopyToDevice(cmd, m_Data->m_GPUParticlePositionsData.data(),
                SizeCast<u32>(m_Data->m_GPUParticlePositionsData.size() * sizeof(Vector2f)), 0);
        });

        // Draw
        struct PushConst
        {
            u32 m_PositionId;
            f32 m_GridRange;
        };

        m_Data->m_RDGParticlePositions =
            &builder.ImportBuffer("APICHostTest.ParticlePositions", m_Data->m_GPUParticlePositions.get());

        auto& pass = builder.AddGraphicsPass("APICHostTest.Draw",
            ShaderVariantDesc(Internal::kIntShaderTableSiro.ParticleRenderVS, {}),
            ShaderVariantDesc(Internal::kIntShaderTableSiro.ParticleRenderFS, {}), GetPushConstSize<PushConst>(),
            RhiRasterizerTopology::Point);

        pass.SetExecutionFunction([renderTarget, this](const FrameGraphPassContext& ctx) {
            auto      rt = renderTarget;

            auto      cmd      = ctx.m_CmdList;
            auto      rtWidth  = rt->GetWidth();
            auto      rtHeight = rt->GetHeight();

            PushConst pc;
            pc.m_PositionId = ctx.m_FgDesc->GetUAV(*m_Data->m_RDGParticlePositions);
            pc.m_GridRange  = m_Data->m_NumGridsX * m_Data->m_GridSizeX;

            cmd->AttachIndexBuffer(m_Data->m_GPUIndexBuffer.get());
            cmd->SetCullMode(RhiCullMode::None);
            cmd->SetPushConst(&pc, 0, sizeof(PushConst));
            cmd->DrawIndexed(m_Data->m_Particles.size(), 1, 0, 0, 0);
        });

        pass.AddRenderTarget(*renderTarget);
    }

} // namespace Ifrit::Runtime::Siro