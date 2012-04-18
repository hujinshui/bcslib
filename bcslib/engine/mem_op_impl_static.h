/**
 * @file mem_op_impl_static.h
 *
 * The implementation of fixed-dimension memory operations
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef MEM_OP_IMPL_STATIC_H_
#define MEM_OP_IMPL_STATIC_H_

#include <bcslib/core/basic_defs.h>

namespace bcs {  namespace engine {


	/********************************************
	 *
	 *	memory operations with known size
	 *
	 ********************************************/

	template<typename T, size_t N> struct mem;

	template<typename T>
	struct mem<T, 0>
	{
		BCS_ENSURE_INLINE static void copy(const T *src, T *dst)
		{
		}

		BCS_ENSURE_INLINE static void zero(T *dst)
		{
		}

		BCS_ENSURE_INLINE static void fill(T *dst, const T &v)
		{
		}

		BCS_ENSURE_INLINE static bool equal(const T *x, const T *y)
		{
			return true;
		}
	};

	template<typename T>
	struct mem<T, 1>
	{
		BCS_ENSURE_INLINE static void copy(const T *src, T *dst)
		{
			*dst = *src;
		}

		BCS_ENSURE_INLINE static void zero(T *dst)
		{
			*dst = T(0);
		}

		BCS_ENSURE_INLINE static void fill(T *dst, const T &v)
		{
			*dst = v;
		}

		BCS_ENSURE_INLINE static bool equal(const T *x, const T *y)
		{
			return *x == *y;
		}
	};

	template<typename T>
	struct mem<T, 2>
	{
		BCS_ENSURE_INLINE static void copy(const T *src, T *dst)
		{
			dst[0] = src[0];
			dst[1] = src[1];
		}

		BCS_ENSURE_INLINE static void zero(T *dst)
		{
			dst[0] = T(0);
			dst[1] = T(0);
		}

		BCS_ENSURE_INLINE static void fill(T *dst, const T &v)
		{
			dst[0] = v;
			dst[1] = v;
		}

		BCS_ENSURE_INLINE static bool equal(const T *x, const T *y)
		{
			return x[0] == y[0] && x[1] == y[1];
		}
	};

	template<typename T>
	struct mem<T, 3>
	{
		BCS_ENSURE_INLINE static void copy(const T *src, T *dst)
		{
			dst[0] = src[0];
			dst[1] = src[1];
			dst[2] = src[2];
		}

		BCS_ENSURE_INLINE static void zero(T *dst)
		{
			dst[0] = T(0);
			dst[1] = T(0);
			dst[2] = T(0);
		}

		BCS_ENSURE_INLINE static void fill(T *dst, const T &v)
		{
			dst[0] = v;
			dst[1] = v;
			dst[2] = v;
		}

		BCS_ENSURE_INLINE static bool equal(const T *x, const T *y)
		{
			return x[0] == y[0] && x[1] == y[1] && x[2] == y[2];
		}
	};

	template<typename T>
	struct mem<T, 4>
	{
		BCS_ENSURE_INLINE static void copy(const T *src, T *dst)
		{
			dst[0] = src[0];
			dst[1] = src[1];
			dst[2] = src[2];
			dst[3] = src[3];
		}

		BCS_ENSURE_INLINE static void zero(T *dst)
		{
			dst[0] = T(0);
			dst[1] = T(0);
			dst[2] = T(0);
			dst[3] = T(0);
		}

		BCS_ENSURE_INLINE static void fill(T *dst, const T &v)
		{
			dst[0] = v;
			dst[1] = v;
			dst[2] = v;
			dst[3] = v;
		}

		BCS_ENSURE_INLINE static bool equal(const T *x, const T *y)
		{
			return x[0] == y[0] && x[1] == y[1] && x[2] == y[2] && x[3] == y[3];
		}
	};


	template<typename T, size_t N>
	struct mem
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(N > 4, "Generic mem<T, N> should be instantiated only when N > 4");
#endif
		static const size_t M = N / 4;
		static const size_t NR = N - 4 * M;

		inline static void copy(const T *src, T *dst)
		{
			for (size_t i = 0; i < M; ++i, src += 4, dst += 4)
			{
				mem<T, 4>::copy(src, dst);
			}
			mem<T, NR>::copy(src, dst);
		}

		inline static void zero(T *dst)
		{
			for (size_t i = 0; i < M; ++i, dst += 4)
			{
				mem<T, 4>::zero(dst);
			}
			mem<T, NR>::zero(dst);
		}

		inline static void fill(T *dst, const T &v)
		{
			for (size_t i = 0; i < M; ++i, dst += 4)
			{
				mem<T, 4>::fill(dst, v);
			}
			mem<T, NR>::fill(dst, v);
		}

		inline static bool equal(const T *x, const T *y)
		{
			for (size_t i = 0; i < M; ++i, x += 4, y += 4)
			{
				if (!mem<T, 4>::equal(x, y)) return false;
			}
			return mem<T, NR>::equal(x, y);
		}
	};

} }

#endif /* MEM_OP_IMPL_STATIC_H_ */
