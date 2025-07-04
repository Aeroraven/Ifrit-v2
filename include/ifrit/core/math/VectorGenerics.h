#pragma once
#include "ifrit/core/math/VectorDefs.h"
#include "ifrit/core/math/VectorOps.h"

namespace Ifrit
{
    template <typename T, u32 Dim> struct TGenericVectorMeta;

    template <typename T> struct TGenericVectorMeta<T, 1>
    {
        using Type                        = T;
        using ScalarType                  = T;
        static IF_CONSTEXPR u32 Dimension = 1;
    };

    template <typename T> struct TGenericVectorMeta<T, 2>
    {
        using Type                        = CoreVec2<T>;
        using ScalarType                  = T;
        static IF_CONSTEXPR u32 Dimension = 2;
    };
    template <typename T> struct TGenericVectorMeta<T, 3>
    {
        using Type                        = CoreVec3<T>;
        using ScalarType                  = T;
        static IF_CONSTEXPR u32 Dimension = 3;
    };
    template <typename T> struct TGenericVectorMeta<T, 4>
    {
        using Type                        = CoreVec4<T>;
        using ScalarType                  = T;
        static IF_CONSTEXPR u32 Dimension = 4;
    };

    template <typename T, u32 Dim IF_REQUIRES(Dim >= 1 && Dim <= 4)>
    using TGenericVector = typename TGenericVectorMeta<T, Dim>::Type;

    template <typename T, u32 Dim IF_REQUIRES(Dim >= 2 && Dim <= 4 && std::is_integral_v<T>)> class TGenericVectorRange
    {

    private:
        TGenericVector<T, Dim> m_Min;
        TGenericVector<T, Dim> m_Max;

    public:
        TGenericVectorRange(TGenericVector<T, Dim> min, TGenericVector<T, Dim> max) : m_Min(min), m_Max(max) {}

    public:
        class iterator
        {
        private:
            TGenericVector<T, Dim> m_Current;
            TGenericVector<T, Dim> m_Max;
            TGenericVector<T, Dim> m_Min;

        public:
            iterator(TGenericVector<T, Dim> current, TGenericVector<T, Dim> min, TGenericVector<T, Dim> max)
                : m_Current(current), m_Min(min), m_Max(max)
            {
            }
            iterator& operator++()
            {
                using namespace Ifrit::Math;
                if IF_CONSTEXPR (Dim == 2)
                {
                    m_Current.x++;
                    if (m_Current.x >= m_Max.x)
                    {
                        m_Current.x = m_Min.x;
                        m_Current.y++;
                    }
                }
                else if IF_CONSTEXPR (Dim == 3)
                {
                    m_Current.x++;
                    if (m_Current.x >= m_Max.x)
                    {
                        m_Current.x = m_Min.x;
                        m_Current.y++;
                        if (m_Current.y >= m_Max.y)
                        {
                            m_Current.y = m_Min.y;
                            m_Current.z++;
                        }
                    }
                }
                else if IF_CONSTEXPR (Dim == 4)
                {
                    m_Current.x++;
                    if (m_Current.x >= m_Max.x)
                    {
                        m_Current.x = m_Min.x;
                        m_Current.y++;
                        if (m_Current.y >= m_Max.y)
                        {
                            m_Current.y = m_Min.y;
                            m_Current.z++;
                            if (m_Current.z >= m_Max.z)
                            {
                                m_Current.z = m_Min.z;
                                m_Current.w++;
                            }
                        }
                    }
                }
                return *this;
            }

            bool operator==(const iterator& other) const
            {
                using namespace Ifrit::Math;
                return All(m_Current == other.m_Current);
            }

            TGenericVector<T, Dim> operator*() const { return m_Current; }
        };

        iterator begin() const { return iterator(m_Min, m_Min, m_Max); }
        iterator end() const
        {
            using namespace Ifrit::Math;
            TGenericVector<T, Dim> endValue = m_Min;
            if IF_CONSTEXPR (Dim == 2)
                endValue.y = m_Max.y;
            else if IF_CONSTEXPR (Dim == 3)
                endValue.z = m_Max.z;
            else if IF_CONSTEXPR (Dim == 4)
                endValue.w = m_Max.w;
            return iterator(endValue, m_Min, m_Max);
        }
    };

} // namespace Ifrit
