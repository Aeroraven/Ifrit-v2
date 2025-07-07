// #pragma once
// #include "ifrit/core/base/IfritBase.h"
// #include "ifrit/core/altina/AlBase.h"
// #include "ifrit/core/typing/Traits.h"
// #include "ifrit/core/altina/AlAllocator.h"
// namespace Ifrit::Altina
// {
//     template <typename T, typename Allocator = TAlAllocator<T>> class TAlVector
//     {
//     private:
//         Allocator         m_Allocator;
//         T*                m_Data     = nullptr;
//         AlContainerSizeTp m_Size     = 0;
//         AlContainerSizeTp m_Capacity = 0;

//     public:
//         struct Iterator
//         {
//             using iterator_category = std::random_access_iterator_tag;
//             using value_type        = T;
//             using difference_type   = AlContainerSizeTp;
//             using pointer           = T*;
//             using reference         = T&;

//             Iterator(T* ptr) : m_Ptr(ptr) {}
//             Iterator& operator++()
//             {
//                 ++m_Ptr;
//                 return *this;
//             }
//             Iterator operator++(int)
//             {
//                 Iterator tmp = *this;
//                 ++(*this);
//                 return tmp;
//             }
//             bool      operator==(const Iterator& other) const { return m_Ptr == other.m_Ptr; }
//             bool      operator!=(const Iterator& other) const { return !(*this == other); }
//             reference operator*() const { return *m_Ptr; }
//             pointer   operator->() const { return m_Ptr; }
//         };

//     private:
//         void _CopyFromImpl(T* data, T* dest, AlContainerSizeTp size)
//         {
//             std::copy(data, data + static_cast<AlPtrSizeTp>(size), dest);
//             return;
//         }
//         void _DestroyAll()
//         {
//             if IF_CONSTEXPR (!TpTrivallyDestructible_v<T>)
//             {
//                 for (AlContainerSizeTp i = 0; i < m_Size; ++i)
//                 {
//                     m_Allocator.Destroy(m_Data + static_cast<AlPtrSizeTp>(i));
//                 }
//             }
//         }
//         void _DestroyAllAndDeallocate()
//         {
//             _DestroyAll();
//             if (m_Data)
//             {
//                 m_Allocator.Deallocate(m_Data, m_Capacity);
//                 m_Data     = nullptr;
//                 m_Size     = 0;
//                 m_Capacity = 0;
//             }
//         }
//         template <typename U IF_REQUIRES(TpIsIterable_v<U>)> void _ConstructAllWithIterators(U&& iterable)
//         {
//             AlContainerSizeTp i = 0;
//             for (auto& item : iterable)
//             {
//                 m_Allocator.Construct(m_Data + static_cast<AlPtrSizeTp>(i++), item);
//             }
//         }

//         void _AssignFromInitializers(std::initializer_list<T> init)
//         {
//             m_Size     = static_cast<AlContainerSizeTp>(init.size());
//             m_Capacity = m_Size;
//             if (m_Size > 0) IF_LIKELY
//             {
//                 m_Data = m_Allocator.Allocate(m_Capacity);
//                 _ConstructAllWithIterators(init);
//             }
//             else
//             {
//                 m_Data = nullptr;
//             }
//         }

//         void _CopyFromVector(const TAlVector& other)
//         {
//             // if other's siz
//         }

//         void _MoveFromVector(TAlVector&& other) IF_NOEXCEPT
//         {
//             m_Allocator = std::move(other.m_Allocator);
//             m_Data      = other.m_Data;
//             m_Size      = other.m_Size;
//             m_Capacity  = other.m_Capacity;

//             other.m_Data     = nullptr;
//             other.m_Size     = 0;
//             other.m_Capacity = 0;
//         }

//     public:
//         TAlVector() IF_NOEXCEPT(IF_NOEXCEPT(Allocator())) : m_Allocator(Allocator()) {}
//         ~TAlVector() { _DestroyAllAndDeallocate(); }

//         explicit TAlVector(AlContainerSizeTp size, const Allocator& allocator = Allocator())
//             : m_Allocator(allocator), m_Size(size), m_Capacity(size)
//         {
//             if (size > 0) IF_LIKELY
//             {
//                 m_Data = m_Allocator.Allocate(size);
//                 for (AlContainerSizeTp i = 0; i < size; ++i)
//                 {
//                     m_Allocator.Construct(m_Data + static_cast<AlPtrSizeTp>(i), T());
//                 }
//             }
//         }

//         explicit TAlVector(AlContainerSizeTp size, const T& value, const Allocator& allocator = Allocator())
//             : m_Allocator(allocator), m_Size(size), m_Capacity(size)
//         {
//             if (size > 0) IF_LIKELY
//             {
//                 m_Data = m_Allocator.Allocate(size);
//                 for (AlContainerSizeTp i = 0; i < size; ++i)
//                 {
//                     m_Allocator.Construct(m_Data + static_cast<AlPtrSizeTp>(i), value);
//                 }
//             }
//         }

//         explicit TAlVector(std::initializer_list<T> init, const Allocator& allocator = Allocator())
//             : m_Allocator(allocator), m_Size(static_cast<AlContainerSizeTp>(init.size())), m_Capacity(m_Size)
//         {
//             if (m_Size > 0) IF_LIKELY
//             {
//                 m_Data = m_Allocator.Allocate(m_Capacity);
//                 _CopyFromImpl(init.begin(), m_Data, m_Size);
//             }
//         }

//         TAlVector(const TAlVector& other)
//         {
//             _DestroyAll();
//             _CopyFromVector(other);
//         }

//         TAlVector(TAlVector&& other) IF_NOEXCEPT
//         {
//             _DestroyAll();
//             _MoveFromVector(std::move(other));
//         }

//         IF_CONSTEXPR T& operator[](AlContainerSizeTp index) const
//         {
//             AlAssert(index < m_Size, "TAlVector: Index out of bounds");
//             return m_Data[index];
//         }
//         IF_CONSTEXPR const T& operator[](AlContainerSizeTp index) IF_NOEXCEPT
//         {
//             AlAssert(index < m_Size, "TAlVector: Index out of bounds");
//             return m_Data[index];
//         }

//         void Reserve(AlContainerSizeTp newCapacity)
//         {
//             if (newCapacity > m_Capacity)
//             {
//                 T* newData = m_Allocator.Allocate(newCapacity);
//                 if (m_Data)
//                 {
//                     _CopyFromImpl(m_Data, newData, m_Size);
//                     _DestroyAll();
//                     m_Allocator.Deallocate(m_Data, m_Capacity);
//                 }

//                 m_Data     = newData;
//                 m_Capacity = newCapacity;
//             }
//         }

//         void Resize(AlContainerSizeTp newSize, const T& value = T())
//         {
//             Reserve(newSize);

//             if (newSize > m_Size)
//             {
//                 for (AlContainerSizeTp i = m_Size; i < newSize; ++i)
//                 {
//                     m_Allocator.Construct(m_Data + static_cast<AlPtrSizeTp>(i), value);
//                 }
//             }
//             else if (newSize < m_Size)
//             {
//                 if IF_CONSTEXPR (!TpTrivallyDestructible_v<T>)
//                 {
//                     for (AlContainerSizeTp i = newSize; i < m_Size; ++i)
//                     {
//                         m_Allocator.Destroy(m_Data + static_cast<AlPtrSizeTp>(i));
//                     }
//                 }
//             }
//             m_Size = newSize;
//         }

//         template <typename... Args> T& EmplaceBack(Args&&... args)
//         {
//             Reserve(m_Size + 1);
//             m_Allocator.Construct(m_Data + static_cast<AlPtrSizeTp>(m_Size), std::forward<Args>(args)...);
//             return m_Data[m_Size++];
//         }

//         void       PushBack(const T& value) { EmplaceBack(value); }
//         void       PushBack(T&& value) { EmplaceBack(std::move(value)); }

//         // iterators
//         Iterator   begin() { return Iterator(m_Data); }
//         Iterator   end() { return Iterator(m_Data + m_Size); }

//         // assign
//         TAlVector& operator=(const TAlVector& other) {}

//     public:
//         IF_CONSTEXPR AlContainerSizeTp Size() const IF_NOEXCEPT { return m_Size; }
//         IF_CONSTEXPR AlContainerSizeTp Capacity() const IF_NOEXCEPT { return m_Capacity; }
//         IF_CONSTEXPR bool              Empty() const IF_NOEXCEPT { return m_Size == 0; }
//     };

// } // namespace Ifrit::Altina
