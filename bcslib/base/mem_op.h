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

#define BEAVER_DEFAULT_ALIGNMENT 16

namespace bcs
{
	/********************************************
	 *
	 *	Basic memory operations
	 *
	 ********************************************/


	/**
	 * A static class for memory operations
	 */
	template<typename T>
	class mem
	{
	public:
		static void copy(const T *src, T *dst, size_t n)
		{
			std::memcpy(dst, src, n * sizeof(T));
		}

		static void zero(T *dst, size_t n)
		{
			std::memset(dst, 0, n * sizeof(T));
		}

		static void fill(T *dst, size_t n, const T& v)
		{
			while(n--) *(dst++) = v;
		}

		static bool equal(const T *s1, const T *s2, size_t n)
		{
			while(n--)
			{
				if (*(s1++) != *(s2++)) return false;
			}
			return true;
		}

		static bool equal(const T *s, const T& v, size_t n)
		{
			while (n--)
			{
				if (*(s++) != v) return false;
			}
			return true;
		}

	}; // end class mem


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
    	static const size_t default_memory_alignment = BEAVER_DEFAULT_ALIGNMENT;

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
