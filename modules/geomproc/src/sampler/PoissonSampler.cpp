#include "ifrit/geomproc/sampler/PoissonSampler.h"
#include "ifrit/core/file/FileOps.h"
#include "ifrit/core/logging/Logging.h"
using namespace Ifrit::Math;
namespace Ifrit::GeometryProc::Sampler
{
    template <typename T, u32 Dim IF_REQUIRES(std::is_floating_point_v<T>&& Dim >= 2 && Dim <= 3)>
    IFRIT_APIDECL Vec<TGenericVector<T, Dim>> PoissonSample(const PoissonSamplerArgs<T, Dim>& args,
        const PoissonSamplerReferences<T, Dim>& refs, Fn<bool(TGenericVector<T, Dim>)> predicate)
    {
        T                        minParticleDistance = args.GetMinDistance();
        TGenericVector<T, Dim>   chunkSize           = (refs.m_MaxBound - refs.m_MinBound) * minParticleDistance;
        TGenericVector<T, Dim>   volRange            = args.m_MaxBound - args.m_MinBound;
        TGenericVector<u32, Dim> numChunks =
            TypeCast<T, u32>(Max(Ceil(volRange / chunkSize), TGenericVector<T, Dim>(1)) + T(0.1));
        Vec<TGenericVector<T, Dim>> acceptedSamples;

        for (auto chunkOffset : TGenericVectorRange<u32, Dim>(TGenericVector<u32, Dim>(0), numChunks))
        {
            for (auto rawSamples : refs.m_Samples)
            {
                auto newPoint       = rawSamples * minParticleDistance + args.m_MinBound;
                auto offsetNewPoint = newPoint + TypeCast<u32, T>(chunkOffset) * chunkSize;
                bool accepted       = true;
                if (Any(offsetNewPoint < args.m_MinBound) || Any(offsetNewPoint > args.m_MaxBound))
                {
                    accepted = false;
                }
                if (accepted && predicate(offsetNewPoint))
                {
                    acceptedSamples.push_back(offsetNewPoint);
                }
            }
        }

        return acceptedSamples;
    }

    IFRIT_APIDECL PoissonSamplerReferences<f32, 3> LoadZpcPoissonSamplerReferences()
    {
        // TODO: Endian
        String data       = ReadBinaryFile(IFRIT_GEOMPROC_SHARED_ASSET_PATH "/GeomUtils/particles-1000k.dat");
        auto   cnt        = *reinterpret_cast<const u64*>(data.data());
        auto   numSamples = cnt;

        PoissonSamplerReferences<f32, 3> refs;
        refs.m_Samples.reserve(numSamples);
        refs.m_MinBound = TGenericVector<f32, 3>(0.0f, 0.0f, 0.0f);
        refs.m_MaxBound = TGenericVector<f32, 3>(120.0f, 120.0f, 120.0f);

        char* rawData = const_cast<char*>(data.data());
        u32   offset  = sizeof(u64) * 2;
        for (u32 i = 0; i < numSamples; ++i)
        {
            f32 x = *reinterpret_cast<f32*>(rawData + offset);
            f32 y = *reinterpret_cast<f32*>(rawData + offset + sizeof(f32));
            f32 z = *reinterpret_cast<f32*>(rawData + offset + 2 * sizeof(f32));
            refs.m_Samples.push_back(TGenericVector<f32, 3>(x, y, z));
            // iDebug("Poisson sample {}: ({}, {}, {})", i, x, y, z);
            offset += 3 * sizeof(f32);
        }
        iInfo("Loaded {} samples from Poisson sampler references.", refs.m_Samples.size());
        return refs;
    }

#define INSTANTIATE_POISSON_SAMPLER(T, Dim)                                                                  \
    template IFRIT_APIDECL Vec<TGenericVector<T, Dim>> PoissonSample(const PoissonSamplerArgs<T, Dim>& args, \
        const PoissonSamplerReferences<T, Dim>& refs, Fn<bool(TGenericVector<T, Dim>)> predicate);

    INSTANTIATE_POISSON_SAMPLER(float, 2);
    INSTANTIATE_POISSON_SAMPLER(float, 3);
    INSTANTIATE_POISSON_SAMPLER(double, 2);
    INSTANTIATE_POISSON_SAMPLER(double, 3);
#undef INSTANTIATE_POISSON_SAMPLER

} // namespace Ifrit::GeometryProc::Sampler