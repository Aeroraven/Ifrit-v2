#pragma once
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/core/typing/Traits.h"

namespace Ifrit
{

    template <typename T> class TCountRef
    {
    public:
        using RefType = T*;
        template <typename U> friend class TCountRef;

        TCountRef() : mRef(nullptr) {}
        TCountRef(nullptr_t) : mRef(nullptr) {}
        explicit TCountRef(T* ref) : mRef(ref)
        {
            if (mRef)
            {
                mRef->AddRef();
            }
        }

        TCountRef(const TCountRef& other)
        {
            mRef = other.mRef;
            if (mRef)
            {
                mRef->AddRef();
            }
        }

        TCountRef(TCountRef&& other) noexcept
        {
            mRef       = other.mRef;
            other.mRef = nullptr;
        }

        template <typename U>
            requires std::is_convertible_v<U*, T*>
        void MoveFrom(TSinkArg<TCountRef<U>> other)
        {
            mRef       = other.mRef;
            other.mRef = nullptr;
        }

        TCountRef& operator=(const TCountRef& other)
        {
            if (this != &other)
            {
                auto oldRef = mRef;
                mRef        = other.mRef;
                if (mRef)
                {
                    mRef->AddRef();
                }
                if (oldRef)
                {
                    oldRef->Release();
                }
            }
            return *this;
        }
        TCountRef& operator=(TCountRef&& other)
        {

            if (this != &other)
            {
                auto oldRef = mRef;
                mRef        = other.mRef;
                other.mRef  = nullptr;
                if (oldRef)
                {
                    oldRef->Release();
                }
            }
            return *this;
        }

        ~TCountRef()
        {
            if (mRef)
            {
                mRef->Release();
            }
        }

        RefType             operator->() const noexcept { return mRef; }
        RefType             Get() const noexcept { return mRef; }
        RefType             Get() noexcept { return mRef; }

        IF_FORCEINLINE bool operator==(const TCountRef& other) const noexcept { return mRef == other.mRef; }
        IF_FORCEINLINE bool operator!=(const TCountRef& other) const noexcept { return mRef != other.mRef; }
        IF_FORCEINLINE bool operator==(RefType other) const noexcept { return mRef == other; }
        IF_FORCEINLINE bool operator!=(RefType other) const noexcept { return mRef != other; }

        u32                 GetRefCount() const
        {
            if (mRef)
            {
                return mRef->GetRefCount();
            }
            return 0;
        }

    private:
        RefType mRef;
    };

    // traits for TCountRef
    template <typename T> struct TIsCountRef
    {
        static constexpr bool Value = false;
        using Type                  = void;
    };
    template <typename T> struct TIsCountRef<TCountRef<T>>
    {
        static constexpr bool Value = true;
        using Type                  = T;
    };
    template <typename T> inline constexpr bool ICountRef = TIsCountRef<T>::Value;

    template <IConceptCountReferable T, typename... Args>
        requires IConceptIsConstructible<T, Args...>
    TCountRef<T> MakeCountRef(Args&&... args)
    {
        auto         ref = new T(std::forward<Args>(args)...);
        TCountRef<T> result(ref);
        return result;
    }

    template <IConceptCountReferable T, typename U>
        requires IConceptCountReferable<typename TIsCountRef<U>::Type>
        && TTraitIsConvertible<typename TIsCountRef<U>::Type*, T*>::value
    TCountRef<T> CastAndMoveCountRef(TSinkArg<U> other)
    {
        TCountRef<T> result;
        result.MoveFrom(std::move(other));
        return result;
    }

} // namespace Ifrit