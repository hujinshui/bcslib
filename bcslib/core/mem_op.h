/**
 * @file mem_op.h
 *
 * Basic memory management and operations
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BEAVER_MEM_OP_H_
#define BEAVER_MEM_OP_H_

#include <bcslib/core/basic_defs.h>

#include <new>  		// for std::bad_alloc
#include <limits> 		// for allocator's max_size method


#define BCS_DEFAULT_ALIGNMENT 32

#include <bcslib/core/bits/mem_op_impl.h>
#include <bcslib/core/bits/mem_op_impl_static.h>

namespace bcs
{
	/********************************************
	 *
	 *	Basic memory operations
	 *
	 ********************************************/

	template<typename T, int N>
	struct mem
	{
		BCS_ENSURE_INLINE static void copy(const T* __restrict__ src, T* __restrict__ dst)
		{
			detail::mem<T, N>::copy(src, dst);
		}

		BCS_ENSURE_INLINE static void zero(T* __restrict__ dst)
		{
			detail::mem<T, N>::zero(dst);
		}

		BCS_ENSURE_INLINE static void fill(T* __restrict__ dst, const T &v)
		{
			detail::mem<T, N>::fill(dst, v);
		}

		BCS_ENSURE_INLINE static bool equal(const T* __restrict__ x, const T* __restrict__ y)
		{
			return detail::mem<T, N>::equal(x, y);
		}
	};

	template<typename T>
	BCS_ENSURE_INLINE
	inline void copy_elems(const index_t n, const T* __restrict__ src, T* __restrict__ dst)
	{
		detail::copy_elems(n, src, dst);
	}

	template<typename T>
	BCS_ENSURE_INLINE
	inline void zero_elems(const index_t n, T* __restrict__ dst)
	{
		detail::zero_elems(n, dst);
	}

	template<typename T>
	BCS_ENSURE_INLINE
	static void fill_elems(const index_t n, T* __restrict__ dst, const T& v)
	{
		detail::fill_elems(n, dst, v);
	}

	template<typename T>
	BCS_ENSURE_INLINE
	static bool elems_equal(const index_t n, const T* __restrict__ s1, const T* __restrict__ s2)
	{
		return detail::elems_equal(n, s1, s2);
	}

	template<typename T>
	BCS_ENSURE_INLINE
	static bool elems_equal(const index_t n, const T* __restrict__ s, const T& v)
	{
		return detail::elems_equal(n, s, v);
	}


	template<typename T>
	BCS_ENSURE_INLINE
	inline void unpack_vector(const index_t len, const T *__restrict__ a, T *__restrict__ b, const index_t bstep)
	{
		for (index_t i = 0; i < len; ++i) b[i * bstep] = a[i];
	}

	template<typename T>
	BCS_ENSURE_INLINE
	inline void pack_vector(const index_t len, const T *__restrict__ a, const index_t astep, T *__restrict__ b)
	{
		for (index_t i = 0; i < len; ++i) b[i] = a[i * astep];
	}


	template<typename T>
	BCS_ENSURE_INLINE
	inline void copy_elems_2d(const index_t inner_dim, const index_t outer_dim,
			const T* __restrict__ src, index_t src_ext, T* __restrict__ dst, index_t dst_ext)
	{
		detail::copy_elems_2d(inner_dim, outer_dim, src, src_ext, dst, dst_ext);
	}

	template<typename T>
	BCS_ENSURE_INLINE
	inline void zero_elems_2d(const index_t inner_dim, const index_t outer_dim,
			T* __restrict__ dst, const index_t dst_ext)
	{
		detail::zero_elems_2d(inner_dim, outer_dim, dst, dst_ext);
	}

	template<typename T>
	BCS_ENSURE_INLINE
	inline void fill_elems_2d(const index_t inner_dim, const index_t outer_dim,
			T* __restrict__ dst, const index_t dst_ext, const T& v)
	{
		detail::fill_elems_2d(inner_dim, outer_dim, dst, dst_ext, v);
	}

	template<typename T>
	BCS_ENSURE_INLINE
	inline bool elems_equal_2d(const index_t inner_dim, const index_t outer_dim,
			const T* __restrict__ s1, index_t s1_ext, const T* __restrict__ s2, index_t s2_ext)
	{
		return detail::elems_equal_2d(inner_dim, outer_dim, s1, s1_ext, s2, s2_ext);
	}

	template<typename T>
	BCS_ENSURE_INLINE
	inline bool elems_equal_2d(const index_t inner_dim, const index_t outer_dim,
			const T* __restrict__ s1, index_t s1_ext, const T& v)
	{
		return detail::elems_equal_2d(inner_dim, outer_dim, s1, s1_ext, v);
	}


	/********************************************
	 *
	 *	aligned allocation
	 *
	 ********************************************/

	BCS_ENSURE_INLINE
	inline void* aligned_allocate(size_t nbytes, unsigned int alignment)
	{
		return detail::aligned_allocate(nbytes, alignment);
	}

	BCS_ENSURE_INLINE
	inline void aligned_release(void *p)
	{
		detail::aligned_release(p);
	}

    template<typename T>
    class aligned_allocator
    {
    public:
    	static const size_t default_memory_alignment = BCS_DEFAULT_ALIGNMENT;

    	typedef T value_type;
    	typedef T* pointer;
    	typedef T& reference;
    	typedef const T* const_pointer;
    	typedef const T& const_reference;
    	typedef size_t size_type;
    	typedef ptrdiff_t difference_type;

    	template<typename TOther>
    	struct rebind
    	{
    		typedef aligned_allocator<TOther> other;
    	};

    public:
    	aligned_allocator()
    	: m_alignment(default_memory_alignment)
    	{
    	}

    	explicit aligned_allocator(unsigned int align)
    	: m_alignment(align)
    	{
    	}

    	aligned_allocator(const aligned_allocator& r)
    	: m_alignment(r.alignment())
    	{
    	}

    	template<typename U>
    	aligned_allocator(const aligned_allocator<U>& r)
    	: m_alignment(r.alignment())
    	{
    	}

    	unsigned int alignment() const
    	{
    		return m_alignment;
    	}


    	pointer address( reference x ) const
    	{
    		return &x;
    	}

    	const_pointer address( const_reference x ) const
    	{
    		return &x;
    	}

    	size_type max_size() const
    	{
    		return std::numeric_limits<size_type>::max() / sizeof(value_type);
    	}

    	pointer allocate(size_type n, const void* hint=0)
    	{
    		return (pointer)aligned_allocate(n * sizeof(value_type), m_alignment);
    	}

    	void deallocate(pointer p, size_type)
    	{
    		aligned_release(p);
    	}

    	void construct (pointer p, const_reference val)
    	{
    		new (p) value_type(val);
    	}

    	void destroy (pointer p)
    	{
    		p->~value_type();
    	}

    private:
    	unsigned int m_alignment;

    }; // end class aligned_allocator



}

#endif
