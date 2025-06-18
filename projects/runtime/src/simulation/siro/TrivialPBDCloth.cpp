#include "ifrit/runtime/simulation/siro/TrivialPBDCloth.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraph.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/simulation/internal/InternalShaderRegistry.Siro.h"

namespace Ifrit::Runtime::Siro
{
    /**
     * Private data structure for TrivialPBDCloth implementation
     * Contains all the buffers and parameters required for position-based dynamics cloth simulation
     */
    struct TrivialPBDClothPrivateData
    {
        // Cloth dimensions and properties
        u32                         m_Width              = 0;     // Number of vertices in width
        u32                         m_Height             = 0;     // Number of vertices in height
        u32                         m_TotalVertices      = 0;     // Total number of vertices (width * height)
        u32                         m_Mass               = 1;     // Mass value for each vertex
        f32                         m_InvMass            = 1.0f;  // 1/Mass, precomputed
        f32                         m_StiffnessStretch   = 0.9f;  // Stiffness coefficient for stretch constraints
        f32                         m_StiffnessBend      = 0.5f;  // Stiffness coefficient for bend constraints
        f32                         m_DampingCoefficient = 0.03f; // Damping coefficient for velocity
        f32                         m_Gravity            = 9.8f;  // Gravity force magnitude (along y-axis)

        // Simulation state flags
        bool                        m_Initialized = false; // Whether the cloth has been initialized

        // RHI Buffers for simulation data
        Graphics::Rhi::RhiBufferRef m_PositionBuffer;          // Current positions (vec4: xyz=position, w=inverse mass)
        Graphics::Rhi::RhiBufferRef m_PrevPositionBuffer;      // Previous positions
        Graphics::Rhi::RhiBufferRef m_VelocityBuffer;          // Velocities
        Graphics::Rhi::RhiBufferRef m_OffsetBuffer;            // Position offsets from constraints
        Graphics::Rhi::RhiBufferRef m_ConstraintBuffer;        // Constraint data (indices for stretch and bend)
        Graphics::Rhi::RhiBufferRef m_NormalBuffer;            // Vertex normals
        Graphics::Rhi::RhiBufferRef m_CounterBuffer;           // Atomic counter for constraint iterations
        Graphics::Rhi::RhiBufferRef m_IndexBuffer;             // Triangle indices for rendering
        Graphics::Rhi::RhiBufferRef m_StretchConstraintBuffer; // Stretch constraint data for GPU
        Graphics::Rhi::RhiBufferRef m_BendConstraintBuffer;    // Bend constraint data for GPU
        Graphics::Rhi::RhiBufferRef m_FixedVertexBuffer;       // Fixed vertices mask/data

        // FrameGraph Buffer Node References (for framegraph integration)
        FGBufferNodeRef             m_PositionBufferNode          = nullptr;
        FGBufferNodeRef             m_PrevPositionBufferNode      = nullptr;
        FGBufferNodeRef             m_VelocityBufferNode          = nullptr;
        FGBufferNodeRef             m_OffsetBufferNode            = nullptr;
        FGBufferNodeRef             m_ConstraintBufferNode        = nullptr;
        FGBufferNodeRef             m_NormalBufferNode            = nullptr;
        FGBufferNodeRef             m_CounterBufferNode           = nullptr;
        FGBufferNodeRef             m_IndexBufferNode             = nullptr;
        FGBufferNodeRef             m_StretchConstraintBufferNode = nullptr;
        FGBufferNodeRef             m_BendConstraintBufferNode    = nullptr;
        FGBufferNodeRef             m_FixedVertexBufferNode       = nullptr;

        // GPU-compatible constraint structures
        struct StretchConstraint
        {
            u32 idxA;       // First vertex index
            u32 idxB;       // Second vertex index
            f32 restLength; // Rest length of the spring
            f32 stiffness;  // Constraint stiffness multiplier
        };

        struct BendConstraint
        {
            u32 idxA;      // First vertex index
            u32 idxB;      // Second vertex index
            u32 idxC;      // Third vertex index (typically from adjacent row/column)
            u32 idxD;      // Fourth vertex index
            f32 restAngle; // Rest angle between triangles
            f32 stiffness; // Constraint stiffness multiplier
        };

        // Simple buffer size helpers (kept for convenience)
        u32 GetPositionBufferSize() const { return m_TotalVertices * sizeof(Vector4f); }
        u32 GetVelocityBufferSize() const { return m_TotalVertices * sizeof(Vector3f); }
        u32 GetNormalBufferSize() const { return m_TotalVertices * sizeof(Vector3f); }
        u32 GetOffsetBufferSize() const { return m_TotalVertices * sizeof(Vector3f); }
    };
    IFRIT_APIDECL TrivialPBDCloth::TrivialPBDCloth() {}

    IFRIT_APIDECL TrivialPBDCloth::~TrivialPBDCloth()
    {
        if (m_Data)
        {
            delete m_Data;
            m_Data = nullptr;
        }
    }
    IFRIT_APIDECL void TrivialPBDCloth::Initialize(FrameGraphBuilder& builder, u32 width, u32 height, u32 mass)
    {
        // Create private data if it doesn't exist
        if (!m_Data)
        {
            m_Data = new TrivialPBDClothPrivateData();
        }

        // Initialize basic cloth properties
        m_Data->m_Width         = width;
        m_Data->m_Height        = height;
        m_Data->m_TotalVertices = width * height;
        m_Data->m_Mass          = mass;
        m_Data->m_InvMass       = mass > 0 ? 1.0f / static_cast<f32>(mass) : 0.0f;

        // Default PBD parameters (can be customized later if needed)
        m_Data->m_StiffnessStretch   = 0.9f;
        m_Data->m_StiffnessBend      = 0.5f;
        m_Data->m_DampingCoefficient = 0.03f;
        m_Data->m_Gravity            = 9.8f;

        // Get RHI backend from the FrameGraphBuilder
        Graphics::Rhi::RhiBackend* rhi = builder.GetRhi();

        // Calculate buffer sizes
        u32                        positionBufferSize = m_Data->GetPositionBufferSize();
        u32                        velocityBufferSize = m_Data->GetVelocityBufferSize();
        u32                        normalBufferSize   = m_Data->GetNormalBufferSize();
        u32                        offsetBufferSize   = m_Data->GetOffsetBufferSize();

        // Number of triangles = 2 * (width-1) * (height-1)
        u32                        triangleCount   = 2 * (width - 1) * (height - 1);
        u32                        indexBufferSize = triangleCount * 3 * sizeof(u32);

        // Estimate constraint counts
        u32 stretchConstraintCount = (width - 1) * height + width * (height - 1) + 2 * (width - 1) * (height - 1);
        u32 bendConstraintCount    = (width - 2) * height + width * (height - 2);

        u32 stretchConstraintBufferSize =
            stretchConstraintCount * sizeof(TrivialPBDClothPrivateData::StretchConstraint);
        u32 bendConstraintBufferSize = bendConstraintCount * sizeof(TrivialPBDClothPrivateData::BendConstraint);
        u32 fixedVertexBufferSize    = width * height * sizeof(u32); // Worst case: all vertices fixed
        u32 counterBufferSize        = sizeof(u32);                  // Single atomic counter

        // Create GPU buffers (all as SSBO for compute shader access)
        u32 ssboUsage = Graphics::Rhi::RhiBufferUsage_SSBO | Graphics::Rhi::RhiBufferUsage_CopySrc
            | Graphics::Rhi::RhiBufferUsage_CopyDst;

        // Position buffers (current and previous)
        m_Data->m_PositionBuffer = rhi->CreateBuffer("PBDCloth_Position", positionBufferSize, ssboUsage, false, true);
        m_Data->m_PrevPositionBuffer =
            rhi->CreateBuffer("PBDCloth_PrevPosition", positionBufferSize, ssboUsage, false, true);

        // Velocity buffer
        m_Data->m_VelocityBuffer = rhi->CreateBuffer("PBDCloth_Velocity", velocityBufferSize, ssboUsage, false, true);

        // Offset buffer for constraint projection
        m_Data->m_OffsetBuffer = rhi->CreateBuffer("PBDCloth_Offset", offsetBufferSize, ssboUsage, false, true);

        // Normal buffer for rendering and collision
        m_Data->m_NormalBuffer = rhi->CreateBuffer("PBDCloth_Normal", normalBufferSize, ssboUsage, false, true);

        // Index buffer for rendering (can also be used as vertex buffer)
        u32 indexUsage        = ssboUsage | Graphics::Rhi::RhiBufferUsage_Index;
        m_Data->m_IndexBuffer = rhi->CreateBuffer("PBDCloth_Indices", indexBufferSize, indexUsage, false, true);

        // Constraint buffers
        m_Data->m_StretchConstraintBuffer =
            rhi->CreateBuffer("PBDCloth_StretchConstraints", stretchConstraintBufferSize, ssboUsage, false, true);

        m_Data->m_BendConstraintBuffer =
            rhi->CreateBuffer("PBDCloth_BendConstraints", bendConstraintBufferSize, ssboUsage, false, true);

        // Fixed vertex buffer (mask)
        m_Data->m_FixedVertexBuffer =
            rhi->CreateBuffer("PBDCloth_FixedVertices", fixedVertexBufferSize, ssboUsage, false, true);

        // Atomic counter buffer
        m_Data->m_CounterBuffer = rhi->CreateBuffer("PBDCloth_Counter", counterBufferSize, ssboUsage, false, true);

        // Create FrameGraph buffer nodes
        auto& positionBufferNode     = builder.ImportBuffer("PBDCloth_Position", m_Data->m_PositionBuffer.get());
        m_Data->m_PositionBufferNode = &positionBufferNode;

        auto& prevPositionBufferNode =
            builder.ImportBuffer("PBDCloth_PrevPosition", m_Data->m_PrevPositionBuffer.get());
        m_Data->m_PrevPositionBufferNode = &prevPositionBufferNode;

        auto& velocityBufferNode     = builder.ImportBuffer("PBDCloth_Velocity", m_Data->m_VelocityBuffer.get());
        m_Data->m_VelocityBufferNode = &velocityBufferNode;

        auto& offsetBufferNode     = builder.ImportBuffer("PBDCloth_Offset", m_Data->m_OffsetBuffer.get());
        m_Data->m_OffsetBufferNode = &offsetBufferNode;

        auto& normalBufferNode     = builder.ImportBuffer("PBDCloth_Normal", m_Data->m_NormalBuffer.get());
        m_Data->m_NormalBufferNode = &normalBufferNode;

        auto& indexBufferNode     = builder.ImportBuffer("PBDCloth_Indices", m_Data->m_IndexBuffer.get());
        m_Data->m_IndexBufferNode = &indexBufferNode;

        auto& stretchConstraintBufferNode =
            builder.ImportBuffer("PBDCloth_StretchConstraints", m_Data->m_StretchConstraintBuffer.get());
        m_Data->m_StretchConstraintBufferNode = &stretchConstraintBufferNode;

        auto& bendConstraintBufferNode =
            builder.ImportBuffer("PBDCloth_BendConstraints", m_Data->m_BendConstraintBuffer.get());
        m_Data->m_BendConstraintBufferNode = &bendConstraintBufferNode;

        auto& fixedVertexBufferNode = builder.ImportBuffer("PBDCloth_FixedVertices", m_Data->m_FixedVertexBuffer.get());
        m_Data->m_FixedVertexBufferNode = &fixedVertexBufferNode;
        auto& counterBufferNode         = builder.ImportBuffer("PBDCloth_Counter", m_Data->m_CounterBuffer.get());
        m_Data->m_CounterBufferNode     = &counterBufferNode;

        // Add a compute pass to initialize the cloth
        // Define push constant data structure for the initialization shader
        struct PushConstant
        {
            u32 m_Width;
            u32 m_Height;
            f32 m_InvMass;
            f32 m_Spacing;
            f32 m_DampingCoefficient;
            f32 m_Gravity;
            u32 m_NumStretchConstraints;
            u32 m_NumBendConstraints;
            u32 m_PositionBuffer;
            u32 m_PrevPositionBuffer;
            u32 m_VelocityBuffer;
            u32 m_OffsetBuffer;
            u32 m_FixedVertexBuffer;
            u32 m_IndexBuffer;
        };

        // Create push constant data
        PushConstant pc;
        pc.m_Width                 = width;
        pc.m_Height                = height;
        pc.m_InvMass               = m_Data->m_InvMass;
        pc.m_Spacing               = 0.1f; // Default spacing between vertices (can be parameterized later)
        pc.m_DampingCoefficient    = m_Data->m_DampingCoefficient;
        pc.m_Gravity               = m_Data->m_Gravity;
        pc.m_NumStretchConstraints = (width - 1) * height + width * (height - 1) + 2 * (width - 1) * (height - 1);
        pc.m_NumBendConstraints    = (width - 2) * height + width * (height - 2);
        // Compute workgroup dimensions based on cloth size
        // Use a block size of 16x16 (as defined in the shader)
        constexpr u32 kBlockSize    = 16;
        u32           dispatchSizeX = (width + kBlockSize - 1) / kBlockSize;
        u32           dispatchSizeY = (height + kBlockSize - 1) / kBlockSize;

        // Add the compute pass using the Init compute shader
        auto&         initPass = FrameGraphUtils::AddComputePass<PushConstant>(builder, "TrivialPBDCloth.Initialize",
            ShaderVariantDesc(Internal::kIntShaderTableSiro.TrivialPBDClothInit, {}),
            Vector3i(dispatchSizeX, dispatchSizeY, 1), pc,
            [positionBufferNode        = m_Data->m_PositionBufferNode,
                prevPositionBufferNode = m_Data->m_PrevPositionBufferNode,
                velocityBufferNode = m_Data->m_VelocityBufferNode, offsetBufferNode = m_Data->m_OffsetBufferNode,
                fixedVertexBufferNode = m_Data->m_FixedVertexBufferNode, indexBufferNode = m_Data->m_IndexBufferNode](
                PushConstant data, const FrameGraphPassContext& ctx) { // Set buffer handles in the push constants
                // Using UAV access for all buffers since we need to write to them in the initialization shader
                data.m_PositionBuffer     = ctx.m_FgDesc->GetUAV(*positionBufferNode);
                data.m_PrevPositionBuffer = ctx.m_FgDesc->GetUAV(*prevPositionBufferNode);
                data.m_VelocityBuffer     = ctx.m_FgDesc->GetUAV(*velocityBufferNode);
                data.m_OffsetBuffer       = ctx.m_FgDesc->GetUAV(*offsetBufferNode);
                data.m_FixedVertexBuffer  = ctx.m_FgDesc->GetUAV(*fixedVertexBufferNode);
                data.m_IndexBuffer        = ctx.m_FgDesc->GetUAV(*indexBufferNode);

                // Send the push constant data to the GPU
                FrameGraphUtils::SetRootConstant(data, ctx);
            });

        // Define resource dependencies for the pass
        // Write operations
        initPass.AddWriteResource(*m_Data->m_PositionBufferNode)
            .AddWriteResource(*m_Data->m_PrevPositionBufferNode)
            .AddWriteResource(*m_Data->m_VelocityBufferNode)
            .AddWriteResource(*m_Data->m_OffsetBufferNode)
            .AddWriteResource(*m_Data->m_FixedVertexBufferNode)
            .AddWriteResource(*m_Data->m_IndexBufferNode);

        // Mark as initialized
        m_Data->m_Initialized = true;
    }

    IFRIT_APIDECL void TrivialPBDCloth::VelocityUpdatePre(FrameGraphBuilder& builder, f32 deltaTime) {}

    IFRIT_APIDECL void TrivialPBDCloth::DampVelocity(FrameGraphBuilder& builder, f32 dampingFactor) {}

    IFRIT_APIDECL void TrivialPBDCloth::InitialOffsetGeneration(FrameGraphBuilder& builder, f32 deltaTime) {}

    IFRIT_APIDECL void TrivialPBDCloth::ProjectConstraintsSingleIteration(FrameGraphBuilder& builder) {}

    IFRIT_APIDECL void TrivialPBDCloth::ProjectConstraints(FrameGraphBuilder& builder, u32 iterations) {}

    IFRIT_APIDECL void TrivialPBDCloth::ApplyAdjustion(FrameGraphBuilder& builder) {}

    IFRIT_APIDECL void TrivialPBDCloth::VelocityUpdatePost(FrameGraphBuilder& builder) {}

    IFRIT_APIDECL void TrivialPBDCloth::Advance(FrameGraphBuilder& builder, f32 deltaTime, u32 solverIters) {}
} // namespace Ifrit::Runtime::Siro