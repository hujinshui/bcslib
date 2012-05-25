/**
 * @file small_blasL1.h
 *
 * Small size BLAS Level 1
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_SMALL_BLASL1_H_
#define BCSLIB_SMALL_BLASL1_H_

#include <bcslib/core/basic_defs.h>

namespace bcs { namespace engine {

	/********************************************
	 *
	 *  small dot
	 *
	 ********************************************/

	template<typename T, int N>
	struct smalldot
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
	struct smalldot<T, 1>
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
	struct smalldot<T, 2>
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



	/********************************************
	 *
	 *  small mul
	 *
	 ********************************************/


	template<typename T, int N>
	struct smallmul
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
	};

	template<typename T>
	struct smallmul<T, 1>
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
	};



	/********************************************
	 *
	 *  small axpy
	 *
	 ********************************************/

	template<typename T, int N>
	struct smallaxpy
	{
		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, T* __restrict__ y)
		{
			if (a == 1) eval1(x, y);
			else evala(a, x, y);
		}

		BCS_ENSURE_INLINE
		static void eval1(const T* __restrict__ x, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) y[i] += x[i];
		}

		BCS_ENSURE_INLINE
		static void evala(const T a, const T* __restrict__ x, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) y[i] += a * x[i];
		}


		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, const int incx, T* __restrict__ y)
		{
			if (a == 1) eval1(x, incx, y);
			else evala(a, x, incx, y);
		}

		BCS_ENSURE_INLINE
		static void eval1(const T* __restrict__ x, const int incx, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) y[i] += x[i * incx];
		}

		BCS_ENSURE_INLINE
		static void evala(const T a, const T* __restrict__ x, const int incx, T* __restrict__ y)
		{
			for (int i = 0; i < N; ++i) y[i] += a * x[i * incx];
		}


		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, T* __restrict__ y, const int incy)
		{
			if (a == 1) eval1(x, y, incy);
			else evala(a, x, y, incy);
		}

		BCS_ENSURE_INLINE
		static void eval1(const T* __restrict__ x, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) y[i * incy] += x[i];
		}

		BCS_ENSURE_INLINE
		static void evala(const T a, const T* __restrict__ x, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) y[i * incy] += a * x[i];
		}


		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (a == 1) eval1(x, incx, y, incy);
			else evala(a, x, incx, y, incy);
		}

		BCS_ENSURE_INLINE
		static void eval1(const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) y[i * incy] += x[i * incx];
		}

		BCS_ENSURE_INLINE
		static void evala(const T a, const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) y[i * incy] += a * x[i * incx];
		}
	};


	template<typename T>
	struct smallaxpy<T, 1>
	{
		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, T* __restrict__ y)
		{
			y[0] += a * x[0];
		}

		BCS_ENSURE_INLINE
		static void eval1(const T* __restrict__ x, T* __restrict__ y)
		{
			y[0] += x[0];
		}

		BCS_ENSURE_INLINE
		static void evala(const T a, const T* __restrict__ x, T* __restrict__ y)
		{
			y[0] += a * x[0];
		}


		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, const int, T* __restrict__ y)
		{
			y[0] += a * x[0];
		}

		BCS_ENSURE_INLINE
		static void eval1(const T* __restrict__ x, const int, T* __restrict__ y)
		{
			y[0] += x[0];
		}

		BCS_ENSURE_INLINE
		static void evala(const T a, const T* __restrict__ x, const int, T* __restrict__ y)
		{
			y[0] += a * x[0];
		}


		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, T* __restrict__ y, const int)
		{
			y[0] += a * x[0];
		}

		BCS_ENSURE_INLINE
		static void eval1(const T* __restrict__ x, T* __restrict__ y, const int)
		{
			y[0] += x[0];
		}

		BCS_ENSURE_INLINE
		static void evala(const T a, const T* __restrict__ x, T* __restrict__ y, const int)
		{
			y[0] += a * x[0];
		}


		BCS_ENSURE_INLINE
		static void eval(const T a, const T* __restrict__ x, const int, const int, T* __restrict__ y, const int)
		{
			y[0] += a * x[0];
		}

		BCS_ENSURE_INLINE
		static void eval1(const T* __restrict__ x, const int, T* __restrict__ y, const int)
		{
			y[0] += x[0];
		}

		BCS_ENSURE_INLINE
		static void evala(const T a, const T* __restrict__ x, const int, T* __restrict__ y, const int)
		{
			y[0] += a * x[0];
		}
	};

} }

#endif /* SMALL_BLASL1_H_ */
