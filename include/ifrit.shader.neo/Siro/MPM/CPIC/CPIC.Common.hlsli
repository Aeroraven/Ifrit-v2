#pragma once
//#define IFSHADER_MPM_3D

#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"
#include "ifrit.shader.neo/Math.Geometry.hlsli"

namespace IfritShader{
namespace Siro{
namespace MPM{

    // Compatible PIC, extra datastructures required
    // b_I (NB,MR): indices of rigid bodies in block i (NB=NC/4^D)
    //              the same affecting range as MPM block opt.
    // b_In(NB): number of rigid bodies in block i, (atomics)
    // g_T (NC,MR/32): sign of distance from grid i to boundary of rigid j
    // g_A (NC,MR/32): validity of distance from grid i to boundary of rigid j
    // g_d (NC): shortest distance from grid i to boundary
    // p_T (NC,MR/32): sign of distance from particle i to boundary of rigid j

    // MPM:
    // 1. Rasterize particles to grid cells
    // CPIC:
    // 1. Rasterize primitive to grid cells (conservative rasterization?) (=> b_I, b_In)
    // 2. Splat distance field (=> g_T, g_A, g_d)
    // 3. Recon particle CDF (=> p_T)

    IFSHADER_DEFINE_CONST_INT32(kCpicMaxRigidPerCell, 64);
    IFSHADER_DEFINE_CONST_INT32(kCpicPageSize, 32);
    IFSHADER_DEFINE_CONST_INT32(kCpicMaxRigidPerParticle, 64); //256 = 32* 8

    struct FCpicRigidBoundary
    {

#ifdef IFSHADER_MPM_3D
        FSpatialVectorAligned m_PA;
        FSpatialVectorAligned m_PB;
        FSpatialVectorAligned m_PC;
#else
        FSpatialVectorAligned m_BoundaryStart;
        FSpatialVectorAligned m_BoundaryEnd;
#endif
    
        FSpatialVector GetNormal()
        {
#ifdef IFSHADER_MPM_3D
            return Math::CalculateTriangleNormal(m_PA.xyz, m_PB.xyz, m_PC.xyz);
#else
            return Math::CalculateSegmentNormal(m_BoundaryStart, m_BoundaryEnd);
#endif      
        }

        FScalar GetDistanceToPoint(FSpatialVector Point)
        {
#ifdef IFSHADER_MPM_3D
            FSpatialVector Normal = GetNormal();
            FSpatialVector Dummy;
            FScalar UnsignedDist = Math::ShortestSignedDistanceToPlane(Point.xyz, Normal.xyz, m_PA.xyz);
            bool Inside = Math::ProjectedPointInTriangle(Point.xyz, m_PA.xyz, m_PB.xyz, m_PC.xyz, Dummy);
            return Inside ? UnsignedDist : 1e30;
#else
            FSpatialVector Normal = GetNormal();
            FScalar Dummy;
            FSpatialVector LineDirection = m_BoundaryEnd - m_BoundaryStart;
            FScalar SignedDist = Math::ShortestSignedDistanceToLine2D(Point.xy, m_BoundaryStart.xy, LineDirection.xy);
            bool Inside = Math::ProjectedPointInSegment2D(Point.xy, m_BoundaryEnd.xy, m_BoundaryStart.xy, Dummy);
            return Inside ? SignedDist : 1e30;
#endif
        }
    };

    struct FCpicRigidParticle
    {
        FSpatialVectorAligned m_Position;
    };

    int EncodeBlockRigidIdList(int BlockId, int RigidId)
    {
        return BlockId * kCpicMaxRigidPerCell + RigidId;
    }   

    int EncodeBlockRigidBitAttrIdList(int BlockId, int Offset)
    {
        return BlockId * kCpicMaxRigidPerCell / kCpicPageSize + Offset;
    }


}}}