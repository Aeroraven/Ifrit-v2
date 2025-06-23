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

#include "ifrit.shader.neo/Siro/PBDCloth.Shared.hlsli"
#include "ifrit.shader.neo/Ayanami/Ayanami.SharedUtils.hlsli"
#include "ifrit.shader.neo/ObjectDescriptor.hlsli"

#include "ifrit.shader.neo/Math.RayTrace.hlsli"

namespace IfritShader{
namespace Siro{

    struct FColliderData
    {
        TRWStructuredBufferHandle<Ayanami::FMeshDFDetail> m_CollisionSDFs;
        TRWStructuredBufferHandle<FPerObjectData> m_ObjectData;
    };

    struct PushConst_T{
        FPBDCollisionProcessIndirectArgs m_CollisionConstraintCounter;
        TRWStructuredBufferHandle<FColliderData> m_ColliderData;
        TRWStructuredBufferHandle<FPBDCollsionConstraint> m_CollisionConstraints;
        TRWStructuredBufferHandle<float4> m_PredPositions;
        TRWStructuredBufferHandle<float4> m_Velocities;
        TRWStructuredBufferHandle<float4> m_Positions;
        uint m_NumParticles;
        uint m_NumSDFs;
    };
    static const float kSDFSurfaceThreshold = 0.04f; // Threshold for surface detection
    static const float kSDFSurfaceShift = 0.01f; // Threshold for surface detection

    bool LineSDFIntersection(
        float3 SegStart,                    // Line segment start point
        float3 SegEnd,                      // Line segment end point
        Ayanami::FMeshDFDetail SDFDetail,   // SDF detail structure
        out float3 IntersectionPoint,
        out float3 IntersectionNormal)      // Output intersection point
    {
        IntersectionPoint = float3(0.0f, 0.0f, 0.0f);
        IntersectionNormal = float3(0.0f, 0.0f, 0.0f);
        
        // First check if line segment intersects the AABB
        float tStart, tEnd;
        float3 Dir = SegEnd - SegStart;
        float SegLength = length(Dir);
        float3 AABBMin = SDFDetail.m_BBoxMin.xyz;
        float3 AABBMax = SDFDetail.m_BBoxMax.xyz;
        
        if (SegLength < 1e-6f) {
            // Handle degenerate segment case - just check if the point is inside
            float3 normalizedPos = (SegStart - AABBMin) / (AABBMax - AABBMin);
            if (all(normalizedPos >= 0.0f) && all(normalizedPos <= 1.0f)) {
                // Point is inside AABB, sample SDF
                float sdfValue = SDFDetail.SampleSdf(normalizedPos) - kSDFSurfaceShift;
                if (sdfValue <= kSDFSurfaceThreshold) { // Small surface threshold
                    IntersectionPoint = SegStart;
                    IntersectionNormal = SDFDetail.SampleSdfNormal(normalizedPos);
                    return true;
                }
            }
            normalizedPos = (SegEnd - AABBMin) / (AABBMax - AABBMin);
            if (all(normalizedPos >= 0.0f) && all(normalizedPos <= 1.0f)) {
                // Point is inside AABB, sample SDF
                float sdfValue = SDFDetail.SampleSdf(normalizedPos) - kSDFSurfaceShift;
                if (sdfValue <= kSDFSurfaceThreshold) { // Small surface threshold
                    IntersectionPoint = SegEnd;
                    IntersectionNormal = SDFDetail.SampleSdfNormal(normalizedPos);
                    return true;
                }
            }

            return false;
        }
        
        Dir = normalize(Dir);
        
        if (!Math::RayBoxIntersection3D(SegStart, Dir, AABBMin, AABBMax, tStart, tEnd)) {
            return false; // No intersection with AABB
        }
        
        // Clamp intersection to segment length
        tStart = max(0.0f, tStart);
        tEnd = min(SegLength, tEnd);
        
        if (tStart >= tEnd) {
            return false; // No valid intersection range
        }
        
        // Ray march through the SDF using sphere tracing
        const int MAX_STEPS = 256;
        const float EPSILON = 0.001f;  // Surface threshold
        
        float t = tStart;
        
        // Binary search refinement to avoid missing thin surfaces
        for (int i = 0; i < MAX_STEPS && t <= tEnd; i++) {
            float3 p = SegStart + t * Dir;
            
            // Convert world position to normalized texture coordinates
            float3 texCoord = (p - AABBMin) / (AABBMax - AABBMin);
            if (any(texCoord < 0.0f) || any(texCoord > 1.0f)) {
                t += (tEnd - t) * 0.1f; // Move forward if we're outside the texture bounds
                continue;
            }
            
            // Sample SDF at this position
            float sdfValue = SDFDetail.SampleSdf(texCoord) - kSDFSurfaceShift;
            
            // If SDF value is near zero, we've hit the surface
            if (abs(sdfValue) <= EPSILON) {
                IntersectionPoint = p;
                IntersectionNormal = SDFDetail.SampleSdfNormal(texCoord);
                return true;
            }
            
            // If we crossed over the surface (sign changed), do binary search to refine
            if (i > 0 && sdfValue <= 0.0f) {
                // Binary search refinement
                float tLow = t - min(abs(sdfValue), (tEnd - tStart) / MAX_STEPS);
                float tHigh = t;
                
                for (int j = 0; j < 8; j++) { // Few binary search iterations
                    float tMid = (tLow + tHigh) * 0.5f;
                    float3 pMid = SegStart + tMid * Dir;
                    float3 texCoordMid = (pMid - AABBMin) / (AABBMax - AABBMin);
                    
                    float midSdfValue = SDFDetail.SampleSdf(texCoordMid)- kSDFSurfaceShift;
                    
                    if (abs(midSdfValue) <= EPSILON) {
                        IntersectionPoint = pMid;
                        IntersectionNormal = SDFDetail.SampleSdfNormal(texCoordMid);
                        return true;
                    }
                    
                    if (midSdfValue < 0.0f) {
                        tHigh = tMid;
                    } else {
                        tLow = tMid;
                    }
                }
                
                // Use the most refined estimate
                IntersectionPoint = SegStart + tLow * Dir;
                float3 finalTexCoord = (IntersectionPoint - AABBMin) / (AABBMax - AABBMin);
                IntersectionNormal = SDFDetail.SampleSdfNormal(finalTexCoord);
                return true;
            }
            
            // Adaptive step size based on SDF value
            // More conservative step size to avoid missing thin surfaces
            float step = max(sdfValue * 0.5f, (tEnd - tStart) / MAX_STEPS);
            t += step;
        }
        
        return false;
    }

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSiroTGSizeX,1,1)
    void SiroPBDClothGenerateSDFCollisionCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if (ThreadId.x >= PushConst.m_NumParticles) return;
        
        float3 TestSphereCenter = float3(0.0f, -0.3f, 0.0f);
        float TestSphereRadius = 0.5f;

        float3 ParticlePredPosition = PushConst.m_PredPositions.Load(ThreadId.x).xyz;
        float3 ParticleVelocity = PushConst.m_Velocities.Load(ThreadId.x).xyz;
        float3 ParticlePosition = PushConst.m_Positions.Load(ThreadId.x).xyz;

        bool bCollision = false;
        float3 CollisionPosition;
        float3 CollisionNormal;
        // bCollision = Math::LineSphereIntersection(ParticlePosition,ParticlePredPosition,
        //     TestSphereCenter, TestSphereRadius, CollisionPosition);
        // CollisionNormal = normalize(CollisionPosition - TestSphereCenter);


        bCollision = LineSDFIntersection(
            ParticlePosition, ParticlePredPosition,
            PushConst.m_ColliderData.Load(0).m_CollisionSDFs.Load(0),
            CollisionPosition, CollisionNormal);

        if (bCollision) {
            
            uint PropId = PushConst.m_CollisionConstraintCounter.AddCounter(1);
            FPBDCollsionConstraint Constraint;
            Constraint.m_ParticleA = ThreadId.x;
            Constraint.m_CollisionNormal =  float4(CollisionNormal, 0.0f);
            Constraint.m_CollisionPos = float4(CollisionPosition + CollisionNormal * 1e-3, 0.0f);
            Constraint.m_CollisionVelocity = float4(ParticleVelocity, 0.0f);

            PushConst.m_CollisionConstraints.Store(Constraint, PropId);
        }
    }
}
}