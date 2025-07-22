#pragma once

#include "ifrit/runtime/common/Pch.h"

namespace Ifrit::Runtime::Artemis
{
    struct MPMGpuParticleBufferCollection
    {
        RHI::RhiBufferRef m_ParticleCount;
        RHI::RhiBufferRef m_ParticlePosition;
        RHI::RhiBufferRef m_ParticleVelocity;
        RHI::RhiBufferRef m_ParticleColor;
        RHI::RhiBufferRef m_ParticleMass;
        RHI::RhiBufferRef m_ParticleDeformGrad;
        RHI::RhiBufferRef m_ParticleDeformGradDet;
        RHI::RhiBufferRef m_ParticleVolume;
        RHI::RhiBufferRef m_ParticleApicB;
        RHI::RhiBufferRef m_ParticleIndex;
        RHI::RhiBufferRef m_ParticleDebug;
        RHI::RhiBufferRef m_ParticleStressContrib;
        RHI::RhiBufferRef m_ParticleMatProperty;
        RHI::RhiBufferRef m_ParticleLiquidDensity; // For PBMPM
    };
} // namespace Ifrit::Runtime::Artemis