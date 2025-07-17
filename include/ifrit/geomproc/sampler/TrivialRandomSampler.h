#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/math/VectorOps.h"
#include "ifrit/geomproc/base/MeshProcBase.h"
#include "ifrit/core/math/VectorGenerics.h"
#include <any>

namespace Ifrit::GeometryProc::Sampler
{
    template <typename T, u32 Dim IF_REQUIRES(std::is_floating_point_v<T>&& Dim >= 2 && Dim <= 3)>
    struct TrivialRandomSamplerArgs
    {
        i32                    m_SampleCount;
        TGenericVector<T, Dim> m_MinBound;
        TGenericVector<T, Dim> m_MaxBound;
    };

    template <typename T, u32 Dim IF_REQUIRES(std::is_floating_point_v<T>&& Dim >= 2 && Dim <= 3)>
    IFRIT_GEOMPROC_API Vec<TGenericVector<T, Dim>> TrivialRandomSample(
        const TrivialRandomSamplerArgs<T, Dim>& args, Fn<bool(TGenericVector<T, Dim>)> predicate);

} // namespace Ifrit::GeometryProc::Sampler