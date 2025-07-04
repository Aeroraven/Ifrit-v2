#include "ifrit/meshproc/engine/sampler/PoissonSampler.h"
using namespace Ifrit::Math;
namespace Ifrit::MeshProcLib::Sampler
{
    template <typename T, u32 Dim IF_REQUIRES(std::is_floating_point_v<T>&& Dim >= 2 && Dim <= 3)>
    IFRIT_APIDECL Vec<TGenericVector<T, Dim>> PoissonSample(const PoissonSamplerArgs<T, Dim>& args,
        const PoissonSamplerReferences<T, Dim>& refs, Fn<bool(TGenericVector<T, Dim>)> predicate)
    {
        T                        minParticleDistance = args.GetMinDistance();
        TGenericVector<T, Dim>   chunkSize           = (args.m_MaxBound - args.m_MinBound) * minParticleDistance;
        TGenericVector<T, Dim>   volRange            = args.m_MaxBound - args.m_MinBound;
        TGenericVector<u32, Dim> numChunks =
            TypeCast<T, u32>(Max(Ceil(volRange / chunkSize), TGenericVector<T, Dim>(1)) + T(0.1));
        Vec<TGenericVector<T, Dim>> acceptedSamples;

        for (auto chunkOffset : TGenericVectorRange<u32, Dim>(TGenericVector<u32, Dim>(0), numChunks))
        {
            // todo
        }

        return acceptedSamples;
    }

#define INSTANTIATE_POISSON_SAMPLER(T, Dim)                                                                  \
    template IFRIT_APIDECL Vec<TGenericVector<T, Dim>> PoissonSample(const PoissonSamplerArgs<T, Dim>& args, \
        const PoissonSamplerReferences<T, Dim>& refs, Fn<bool(TGenericVector<T, Dim>)> predicate);

    INSTANTIATE_POISSON_SAMPLER(float, 2)
    INSTANTIATE_POISSON_SAMPLER(float, 3)
    INSTANTIATE_POISSON_SAMPLER(double, 2)
    INSTANTIATE_POISSON_SAMPLER(double, 3)
#undef INSTANTIATE_POISSON_SAMPLER

} // namespace Ifrit::MeshProcLib::Sampler