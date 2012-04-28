/*
 * @file mem_op_impl.h
 *
 * The implementation of optimized memory operations
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MEM_OP_IMPL_H_
#define BCSLIB_MEM_OP_IMPL_H_

#include <bcslib/core/basic_defs.h>
#include <cstring>

// for platform-dependent aligned allocation
#include <stdlib.h>
#if BCS_PLATFORM_INTERFACE == BCS_WINDOWS_INTERFACE
#include <malloc.h>
#endif

namespace bcs { namespace detail  {

	/********************************************
	 *
	 *  Element wise operations
	 *
	 ********************************************/

	template<typename T>
	BCS_ENSURE_INLINE
	inline void copy_elems(const index_t n, const T* __restrict__ src, T* __restrict__ dst)
	{
		if (n >> 3) // n >= 8
		{
			std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(T));
		}
		else
		{
			if (n & 4)
			{
				dst[0] = src[0];
				dst[1] = src[1];
				dst[2] = src[2];
				dst[3] = src[3];
				src += 4;
				dst += 4;
			}

			if (n & 2)
			{
				dst[0] = src[0];
				dst[1] = src[1];
				src += 2;
				dst += 2;
			}

			if (n & 1)
			{
				*dst = *src;
			}
		}
	}


	template<typename T>
	BCS_ENSURE_INLINE
	inline void zero_elems(const index_t n, T* __restrict__ dst)
	{
		std::memset(dst, 0, static_cast<size_t>(n) * sizeof(T));
	}

	template<typename T>
	BCS_ENSURE_INLINE
	static void fill_elems(const index_t n, T* __restrict__ dst, const T& v)
	{
		for (index_t i = 0; i < n; ++i) dst[i] = v;
	}

	template<typename T>
	BCS_ENSURE_INLINE
	static bool elems_equal(const index_t n, const T* __restrict__ s1, const T* __restrict__ s2)
	{
		for (index_t i = 0; i < n; ++i)
		{
			if (s1[i] != s2[i]) return false;
		}
		return true;
	}

	template<typename T>
	BCS_ENSURE_INLINE
	static bool elems_equal(const index_t n, const T* __restrict__ s, const T& v)
	{
		for (index_t i = 0; i < n; ++i)
		{
			if (s[i] != v) return false;
		}
		return true;
	}


	template<typename T>
	inline void copy_elems_2d(const index_t inner_dim, const index_t outer_dim,
			const T* __restrict__ src, index_t src_ext, T* __restrict__ dst, index_t dst_ext)
	{
		for (index_t j = 0; j < outer_dim; ++j, src += src_ext, dst += dst_ext)
		{
			copy_elems(inner_dim, src, dst);
		}
	}

	template<typename T>
	inline void zero_elems_2d(const index_t inner_dim, const index_t outer_dim,
			T* __restrict__ dst, const index_t dst_ext)
	{
		if (inner_dim == dst_ext)
		{
			zero_elems(inner_dim * outer_dim, dst);
		}
		else
		{
			for (index_t j = 0; j < outer_dim; ++j, dst += dst_ext)
			{
				zero_elems(inner_dim, dst);
			}
		}
	}

	template<typename T>
	inline void fill_elems_2d(const index_t inner_dim, const index_t outer_dim,
			T* __restrict__ dst, const index_t dst_ext, const T& v)
	{
		for (index_t j = 0; j < outer_dim; ++j, dst += dst_ext)
		{
			for (index_t i = 0; i < inner_dim; ++i) dst[i] = v;
		}
	}

	template<typename T>
	inline bool elems_equal_2d(const index_t inner_dim, const index_t outer_dim,
			const T* __restrict__ s1, index_t s1_ext, const T* __restrict__ s2, index_t s2_ext)
	{
		for (index_t i = 0; i < outer_dim; ++i, s1 += s1_ext, s2 += s2_ext)
		{
			for (index_t i = 0; i < inner_dim; ++i)
			{
				if (s1[i] != s2[i]) return false;
			}
		}
		return true;
	}

	template<typename T>
	inline bool elems_equal_2d(const index_t inner_dim, const index_t outer_dim,
			const T* __restrict__ s1, index_t s1_ext, const T& v)
	{
		for (index_t i = 0; i < outer_dim; ++i, s1 += s1_ext)
		{
			for (index_t i = 0; i < inner_dim; ++i)
			{
				if (s1[i] != v) return false;
			}
		}
		return true;
	}


	/********************************************
	 *
	 *  Aligned allocation
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



} }

#endif 
