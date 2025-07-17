#include "ifrit/geomproc/sampler/TrivialRandomSampler.h"
#include <random>
namespace Ifrit::GeometryProc::Sampler
{
    template <typename T, u32 Dim IF_REQUIRES(std::is_floating_point_v<T>&& Dim >= 2 && Dim <= 3)>
    IFRIT_APIDECL Vec<TGenericVector<T, Dim>> TrivialRandomSample(
        const TrivialRandomSamplerArgs<T, Dim>& args, Fn<bool(TGenericVector<T, Dim>)> predicate)
    {
        static std::default_random_engine generator;
        std::uniform_real_distribution<T> distribution(0.0, 1.0);
        Vec<TGenericVector<T, Dim>>       samples;
        for (int i = 0; i < args.m_SampleCount; ++i)
        {
            TGenericVector<T, Dim> sample;
            for (u32 d = 0; d < Dim; ++d)
            {
                sample[d] = args.m_MinBound[d] + (args.m_MaxBound[d] - args.m_MinBound[d]) * distribution(generator);
            }
            if (predicate(sample))
            {
                samples.push_back(sample);
            }
        }
        return samples;
    }

#define INSTANTIATE_TRIVIAL_RANDOM_SAMPLER(T, Dim)                          \
    template IFRIT_APIDECL Vec<TGenericVector<T, Dim>> TrivialRandomSample( \
        const TrivialRandomSamplerArgs<T, Dim>& args, Fn<bool(TGenericVector<T, Dim>)> predicate);

    INSTANTIATE_TRIVIAL_RANDOM_SAMPLER(f32, 2)
    INSTANTIATE_TRIVIAL_RANDOM_SAMPLER(f32, 3)
    INSTANTIATE_TRIVIAL_RANDOM_SAMPLER(f64, 2)
    INSTANTIATE_TRIVIAL_RANDOM_SAMPLER(f64, 3)
#undef INSTANTIATE_TRIVIAL_RANDOM_SAMPLER

} // namespace Ifrit::GeometryProc::Sampler