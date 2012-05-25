/*
 * @file blas_kernel.h
 *
 * The core building blocks for implementing BLAS
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_BLAS_KERNEL_H_
#define BCSLIB_BLAS_KERNEL_H_

#include <bcslib/core/basic_defs.h>
#include <bcslib/math/scalar_math.h>


namespace bcs { namespace engine {

	// cache

	template<typename T, int N>
	struct small_cache
	{
		T data[N] __attribute__(( aligned(32) ));
	};

	// vcopy

	template<typename T, int N>
	struct copy_ker
	{
		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) y[i] = x[i];
		}

		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, const int incx, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) y[i] = x[i * incx];
		}

		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) y[i * incy] = x[i];
		}

		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) y[i * incy] = x[i * incx];
		}
	};

	template<typename T>
	struct copy_ker<T, 1>
	{
		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, T* __restrict__ y)
		{
			y[0] = x[0];
		}

		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, const int, T* __restrict__ y)
		{
			y[0] = x[0];
		}

		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, T* __restrict__ y, const int)
		{
			y[0] = x[0];
		}

		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, const int, T* __restrict__ y, const int)
		{
			y[0] = x[0];
		}
	};


	// vadd

	template<typename T, int N>
	struct add_ker
	{
		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) y[i] += x[i];
		}

		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, const int incx, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) y[i] += x[i * incx];
		}

		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) y[i * incy] += x[i];
		}

		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) y[i * incy] += x[i * incx];
		}
	};

	template<typename T>
	struct add_ker<T, 1>
	{
		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, T* __restrict__ y)
		{
			y[0] += x[0];
		}

		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, const int, T* __restrict__ y)
		{
			y[0] += x[0];
		}

		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, T* __restrict__ y, const int)
		{
			y[0] += x[0];
		}

		BCS_ENSURE_INLINE
		static void eval(const T* __restrict__ x, const int, T* __restrict__ y, const int)
		{
			y[0] += x[0];
		}
	};


	// vmul

	template<typename T, int N>
	struct mul_ker
	{
		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) y[i] = a * x[i];
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, const int incx, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) y[i] = a * x[i * incx];
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) y[i * incy] = a * x[i];
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) y[i * incy] = a * x[i * incx];
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) y[i] *= a;
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) y[i * incy] *= a;
		}
	};

	template<typename T>
	struct mul_ker<T, 1>
	{
		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, T* __restrict__ y)
		{
			y[0] = a * x[0];
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, const int, T* __restrict__ y)
		{
			y[0] = a * x[0];
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, T* __restrict__ y, const int)
		{
			y[0] = a * x[0];
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, const int, T* __restrict__ y, const int)
		{
			y[0] = a * x[0];
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, T* __restrict__ y)
		{
			y[0] *= a;
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, T* __restrict__ y, const int)
		{
			y[0] *= a;
		}
	};


	// vaddx

	template<typename T, int N>
	struct addx_ker
	{
		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) y[i] += a * x[i];
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, const int incx, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) y[i] += a * x[i * incx];
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) y[i * incy] += a * x[i];
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) y[i * incy] += a * x[i * incx];
		}
	};

	template<typename T>
	struct addx_ker<T, 1>
	{
		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, T* __restrict__ y)
		{
			y[0] += a * x[0];
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, const int, T* __restrict__ y)
		{
			y[0] += a * x[0];
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, T* __restrict__ y, const int)
		{
			y[0] += a * x[0];
		}

		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, const int, T* __restrict__ y, const int)
		{
			y[0] += a * x[0];
		}
	};


	// vdot

	template<typename T, int N>
	struct dot_ker
	{
		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const T* __restrict__ y)
		{
			T s(0);
			for (int i = 0; i < N; ++i) s += x[i] * y[i];
			return s;
		}

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const int incx, const T* __restrict__ y)
		{
			T s(0);
			for (int i = 0; i < N; ++i) s += x[i * incx] * y[i];
			return s;
		}

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const T* __restrict__ y, const int incy)
		{
			T s(0);
			for (int i = 0; i < N; ++i) s += x[i] * y[i * incy];
			return s;
		}

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const int incx, const T* __restrict__ y, const int incy)
		{
			T s(0);
			for (int i = 0; i < N; ++i) s += x[i * incx] * y[i * incy];
			return s;
		}
	};

	template<typename T>
	struct dot_ker<T, 1>
	{
		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const T* __restrict__ y)
		{
			return x[0] * y[0];
		}

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const int, const T* __restrict__ y)
		{
			return x[0] * y[0];
		}

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const T* __restrict__ y, const int)
		{
			return x[0] * y[0];
		}

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const int, const T* __restrict__ y, const int)
		{
			return x[0] * y[0];
		}
	};

	template<typename T>
	struct dot_ker<T, 2>
	{
		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const T* __restrict__ y)
		{
			return x[0] * y[0] + x[1] * y[1];
		}

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const int incx, const T* __restrict__ y)
		{
			return x[0] * y[0] + x[incx] * y[1];
		}

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const T* __restrict__ y, const int incy)
		{
			return x[0] * y[0] + x[1] * y[incy];
		}

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const int incx, const T* __restrict__ y, const int incy)
		{
			return x[0] * y[0] + x[incx] * y[incy];
		}
	};


	// vsqsum

	template<typename T, int N>
	struct vsqsum_ker
	{
		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x)
		{
			T s(0);
			for (int i = 0; i < N; ++i) s += math::sqr(x[i]);
			return s;
		}

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const int incx)
		{
			T s(0);
			for (int i = 0; i < N; ++i) s += math::sqr(x[i * incx]);
			return s;
		}
	};

	template<typename T>
	struct vsqsum_ker<T, 1>
	{
		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x)
		{
			return math::sqr(x[0]);
		}

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const int incx)
		{
			return math::sqr(x[0]);
		}
	};

	template<typename T>
	struct vsqsum_ker<T, 2>
	{
		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x)
		{
			return math::sqr(x[0]) + math::sqr(x[1]);
		}

		BCS_ENSURE_INLINE
		static T eval(const T* __restrict__ x, const int incx)
		{
			return math::sqr(x[0]) + math::sqr(x[incx]);
		}
	};

} }

#endif 
