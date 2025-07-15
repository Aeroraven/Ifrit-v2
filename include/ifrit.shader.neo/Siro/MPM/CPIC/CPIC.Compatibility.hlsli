#pragma once

#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"
#include "ifrit.shader.neo/Siro/MPM/CPIC/CPIC.Common.hlsli"

namespace IfritShader {
namespace Siro {
namespace MPM {

    struct FCpicSignTagSet
    {
        int m_Affinity[kCpicMaxRigidPerCell/32];
        int m_Sign[kCpicMaxRigidPerCell/32];
    };

    FCpicSignTagSet GetCpicParticleAffinityAndTags(
        int ParticleIndex,
        TStructuredBufferHandle<int> hCpicPAffinity,
        TStructuredBufferHandle<int> hCpicPSign
    )
    {
        FCpicSignTagSet Result;
        const int NumTagsInI32 = kCpicMaxRigidPerCell / 32;

        for(int i = 0; i < NumTagsInI32; ++i)
        {
            Result.m_Affinity[i] = hCpicPAffinity.Load(ParticleIndex * NumTagsInI32 + i);
            Result.m_Sign[i] = hCpicPSign.Load(ParticleIndex * NumTagsInI32 + i);
        }
        return Result;
    }


    FCpicSignTagSet GetCpicGridAffinityAndTags(
        int GridIndex,
        TStructuredBufferHandle<int> hCpicGAffinity,
        TStructuredBufferHandle<int> hCpicGSign
    )
    {
        FCpicSignTagSet Result;
        const int NumTagsInI32 = kCpicMaxRigidPerCell / 32;

        for(int i = 0; i < NumTagsInI32; ++i)
        {
            Result.m_Affinity[i] = hCpicGAffinity.Load(GridIndex * NumTagsInI32 + i);
            Result.m_Sign[i] = hCpicGSign.Load(GridIndex * NumTagsInI32 + i);
        }
        return Result;
    }

    bool CheckCpicCompatibility(
        FCpicSignTagSet ParticleTags,
        FCpicSignTagSet GridTags
    )
    {
        const int NumTagsInI32 = kCpicMaxRigidPerCell / 32;

        for(int i = 0; i < NumTagsInI32; ++i)
        {
            int PA = ParticleTags.m_Affinity[i];
            int GA = GridTags.m_Affinity[i];
            int PT = ParticleTags.m_Sign[i];
            int GT = GridTags.m_Sign[i];

            int Intersection = PA & GA;
            int Incompatibility = PT ^ GT;
            int FilteredIncompatibility = Incompatibility & Intersection;

            if(FilteredIncompatibility != 0)
            {
                return false; 
            }
        }
        return true;
    }


}}}