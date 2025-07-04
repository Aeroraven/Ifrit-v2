#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/math/VectorOps.h"
#include "ifrit/meshproc/engine/base/MeshProcBase.h"
#include "ifrit/core/math/VectorGenerics.h"
#include <any>

namespace Ifrit::MeshProcLib::Sampler
{
    // References:
    // https://github.com/zenustech/zpc/blob/master/include/zensim/geometry/PoissonDisk.hpp

    template <typename T, u32 Dim IF_REQUIRES(std::is_floating_point_v<T>&& Dim >= 2 && Dim <= 3)>
    struct PoissonSamplerArgs
    {
        T                      m_CellDx;
        u32                    m_ParticlePerCell;
        TGenericVector<T, Dim> m_MinBound;
        TGenericVector<T, Dim> m_MaxBound;

        IF_FORCEINLINE T       GetMinDistance() const IF_NOEXCEPT
        {
            if IF_CONSTEXPR (Dim == 2)
                return std::sqrt(m_CellDx * (2.0 / 3.0));
            else if IF_CONSTEXPR (Dim == 3)
                return std::pow(m_CellDx * (13.0 / 18.0), 1.0 / 3.0);
            return T(0);
        }
    };

    template <typename T, u32 Dim IF_REQUIRES(std::is_floating_point_v<T>&& Dim >= 2 && Dim <= 3)>
    struct PoissonSamplerReferences
    {
        Vec<TGenericVector<T, Dim>> m_Samples;
        TGenericVector<T, Dim>      m_MinBound;
        TGenericVector<T, Dim>      m_MaxBound;
    };

    template <typename T, u32 Dim IF_REQUIRES(std::is_floating_point_v<T>&& Dim >= 2 && Dim <= 3)>
    IFRIT_MESHPROC_API Vec<TGenericVector<T, Dim>> PoissonSample(const PoissonSamplerArgs<T, Dim>& args,
        const PoissonSamplerReferences<T, Dim>& refs, Fn<bool(TGenericVector<T, Dim>)> predicate);

} // namespace Ifrit::MeshProcLib::Sampler