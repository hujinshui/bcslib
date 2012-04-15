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

#include <bcslib/base/basic_defs.h>

#include <cstdlib>
#include <cstring>
#include <new>  		// for std::bad_alloc
#include <limits> 		// for allocator's max_size method

// for platform-dependent aligned allocation
#include <stdlib.h>
#if BCS_PLATFORM_INTERFACE == BCS_WINDOWS_INTERFACE
#include <malloc.h>
#endif

#define BCS_DEFAULT_ALIGNMENT 16

namespace bcs
{
	/********************************************
	 *
	 *	Basic memory operations
	 *
	 ********************************************/

	template<typename T>
	inline void copy_elems(const size_t n, const T *src, T *dst)
	{
		std::memcpy(dst, src, n * sizeof(T));
	}

	template<typename T>
	inline void zero_elems(const size_t n, T *dst)
	{
		std::memset(dst, 0, n * sizeof(T));
	}

	template<typename T>
	static void fill_elems(const size_t n, T *dst, const T& v)
	{
		for (size_t i = 0; i < n; ++i) dst[i] = v;
	}

	template<typename T>
	static bool elems_equal(const size_t n, const T *s1, const T *s2)
	{
		for (size_t i = 0; i < n; ++i)
		{
			if (s1[i] != s2[i]) return false;
		}
		return true;
	}

	template<typename T>
	static bool elems_equal(const size_t n, const T *s, const T& v)
	{
		for (size_t i = 0; i < n; ++i)
		{
			if (s[i] != v) return false;
		}
		return true;
	}


	template<typename T>
	inline void copy_elems_2d(const size_t inner_dim, const size_t outer_dim,
			const T *src, size_t src_ext, T *dst, size_t dst_ext)
	{
		for (size_t j = 0; j < outer_dim; ++j, src += src_ext, dst += dst_ext)
		{
			std::memcpy(dst, src, inner_dim * sizeof(T));
		}
	}

	template<typename T>
	inline void zero_elems_2d(const size_t inner_dim, const size_t outer_dim,
			T *dst, const size_t dst_ext)
	{
		for (size_t j = 0; j < outer_dim; ++j, dst += dst_ext)
		{
			std::memset(dst, 0, inner_dim * sizeof(T));
		}
	}

	template<typename T>
	inline void fill_elems_2d(const size_t inner_dim, const size_t outer_dim,
			T *dst, const size_t dst_ext, const T& v)
	{
		for (size_t j = 0; j < outer_dim; ++j, dst += dst_ext)
		{
			for (size_t i = 0; i < inner_dim; ++i) dst[i] = v;
		}
	}

	template<typename T>
	inline bool elems_equal_2d(const size_t inner_dim, const size_t outer_dim,
			const T *s1, size_t s1_ext, const T *s2, size_t s2_ext)
	{
		for (size_t i = 0; i < outer_dim; ++i, s1 += s1_ext, s2 += s2_ext)
		{
			for (size_t i = 0; i < inner_dim; ++i)
			{
				if (s1[i] != s2[i]) return false;
			}
		}
		return true;
	}

	template<typename T>
	inline bool elems_equal_2d(const size_t inner_dim, const size_t outer_dim,
			const T *s1, size_t s1_ext, const T& v)
	{
		for (size_t i = 0; i < outer_dim; ++i, s1 += s1_ext)
		{
			for (size_t i = 0; i < inner_dim; ++i)
			{
				if (s1[i] != v) return false;
			}
		}
		return true;
	}


	/********************************************
	 *
	 *	aligned allocation
	 *
	 ********************************************/

	inline void* aligned_allocate(size_t nbytes, unsigned int alignment)
	{
#if BCS_PLATFORM_INTERFACE == BCS_WINDOWS_INTERFACE
		void* p = ::_aligned_malloc(nbytes, alignment));
		if (!p)
		{
			throw std::bad_alloc();
		}
		return p;

#elif BCS_PLATFORM_INTERFACE == BCS_POSIX_INTERFACE
		char* p = 0;
		if (::posix_memalign((void**)(&p), alignment, nbytes) != 0)
		{
			throw std::bad_alloc();
		}
		return p;
#endif
	}


	inline void aligned_release(void *p)
	{
#if BCS_PLATFORM_INTERFACE == BCS_WINDOWS_INTERFACE
        ::_aligned_free(p);
#elif BCS_PLATFORM_INTERFACE == BCS_POSIX_INTERFACE
        ::free(p);
#endif
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
