/**
 * @file small_blasL2.h
 *
 * Small BLAS L2
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_SMALL_BLASL2_H_
#define BCSLIB_SMALL_BLASL2_H_

#include "small_blasL1.h"

namespace bcs { namespace engine {

	// cache

	template<typename T, int N>
	struct small_cache
	{
		T data[N] __attribute__(( aligned(32) ));

		BCS_ENSURE_INLINE
		explicit small_cache() { }

		BCS_ENSURE_INLINE
		explicit small_cache(const T* __restrict__ x)
		{
			for (int i = 0; i < N; ++i) data[i] = x[i];
		}

		BCS_ENSURE_INLINE
		explicit small_cache(const T* __restrict__ x, const int incx)
		{
			for (int i = 0; i < N; ++i) data[i] = x[i * incx];
		}

		BCS_ENSURE_INLINE
		void copy_to(T *__restrict__ y)
		{
			for (int i = 0; i < N; ++i) y[i] = data[i];
		}

		BCS_ENSURE_INLINE
		void copy_to(T *__restrict__ y, const int incy)
		{
			for (int i = 0; i < N; ++i) y[i * incy] = data[i];
		}
	};


	/********************************************
	 *
	 *  small gemv
	 *
	 ********************************************/

	template<typename T, int M>
	struct smallgemv_M1
	{
		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			smallmul<T, M>::eval(alpha * x[0], a, y);
		}

		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			smallmul<T, M>::eval(alpha * x[0], a, y, incy);
		}


		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			smallaxpy<T, M>::evala(alpha * x[0], a, y);
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			smallaxpy<T, M>::evala(alpha * x[0], a, y, incy);
		}


		BCS_ENSURE_INLINE
		static void eval_t_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			y[0] = alpha * smalldot<T, M>::eval(a, x);
		}

		BCS_ENSURE_INLINE
		static void eval_t_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			y[0] = alpha * smalldot<T, M>::eval(a, x, incx);
		}

		BCS_ENSURE_INLINE
		static void eval_t_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			y[0] += alpha * smalldot<T, M>::eval(a, x);
		}

		BCS_ENSURE_INLINE
		static void eval_t_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			y[0] += alpha * smalldot<T, M>::eval(a, x, incx);
		}
	};


	template<typename T, int N>
	struct smallgemv_1N
	{
		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (lda == 1)
			{
				y[0] = alpha * smalldot<T, N>::eval(a, x);
			}
			else
			{
				y[0] = alpha * smalldot<T, N>::eval(a, lda, x);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (lda == 1)
			{
				y[0] = alpha * smalldot<T, N>::eval(a, x, incx);
			}
			else
			{
				y[0] = alpha * smalldot<T, N>::eval(a, lda, x, incx);
			}
		}


		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (lda == 1)
			{
				y[0] += alpha * smalldot<T, N>::eval(a, x);
			}
			else
			{
				y[0] += alpha * smalldot<T, N>::eval(a, lda, x);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (lda == 1)
			{
				y[0] += alpha * smalldot<T, N>::eval(a, x, incx);
			}
			else
			{
				y[0] += alpha * smalldot<T, N>::eval(a, lda, x, incx);
			}
		}


		BCS_ENSURE_INLINE
		static void eval_t_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (lda == 1)
			{
				smallmul<T, N>::eval(x[0], a, y);
			}
			else
			{
				smallmul<T, N>::eval(x[0], a, lda, y);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_t_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (lda == 1)
			{
				smallmul<T, N>::eval(x[0], a, y, incy);
			}
			else
			{
				smallmul<T, N>::eval(x[0], a, lda, y, incy);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_t_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (lda == 1)
			{
				smallaxpy<T, N>::eval(x[0], a, y);
			}
			else
			{
				smallaxpy<T, N>::eval(x[0], a, lda, y);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_t_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (lda == 1)
			{
				smallaxpy<T, N>::eval(x[0], a, y, incy);
			}
			else
			{
				smallaxpy<T, N>::eval(x[0], a, lda, y, incy);
			}
		}
	};


	template<typename T, int M, int N>
	struct smallgemv_MN
	{
		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (alpha == 1)
			{
				smallmul<T, M>::eval(x[0], a, y);

				for (int j = 1; j < N; ++j)
					smallaxpy<T, M>::evala(x[j], a + lda * j, y);
			}
			else
			{
				smallmul<T, M>::eval(alpha * x[0], a, y);

				for (int j = 1; j < N; ++j)
					smallaxpy<T, M>::evala(alpha * x[j], a + lda * j, y);
			}
		}


		BCS_ENSURE_INLINE
		static void eval_b0_(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y)
		{
			if (alpha == 1)
			{
				smallmul<T, M>::eval(x[0], a, y);

				for (int j = 1; j < N; ++j)
					smallaxpy<T, M>::evala(x[j * incx], a + lda * j, y);
			}
			else
			{
				smallmul<T, M>::eval(alpha * x[0], a, y);

				for (int j = 1; j < N; ++j)
					smallaxpy<T, M>::evala(alpha * x[j * incx], a + lda * j, y);
			}
		}


		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (incy == 1)
			{
				eval_b0_(alpha, a, lda, x, incx, y);
			}
			else
			{
				small_cache<T, M> cache_y;
				eval_b0_(alpha, a, lda, x, incx, cache_y.data);
				cache_y.copy_to(y, incy);
			}
		}


		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					smallaxpy<T, M>::evala(x[j], a + lda * j, y);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					smallaxpy<T, M>::evala(alpha * x[j], a + lda * j, y);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b1_(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					smallaxpy<T, M>::evala(x[j * incx], a + lda * j, y);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					smallaxpy<T, M>::evala(alpha * x[j * incx], a + lda * j, y);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (incy == 1)
			{
				eval_b1_(alpha, a, lda, x, incx, y);
			}
			else
			{
				small_cache<T, M> cache_y;
				eval_b0_(alpha, a, lda, x, incx, cache_y.data);
				smallaxpy<T, M>::eval1(cache_y.data, y);
			}
		}


		BCS_ENSURE_INLINE
		static void eval_t_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					y[j] = smalldot<T, M>::eval(a + lda * j, x);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					y[j] = alpha * smalldot<T, M>::eval(a + lda * j, x);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_t_b0_(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y, const int incy)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					y[j * incy] = smalldot<T, M>::eval(a + lda * j, x);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					y[j * incy] = alpha * smalldot<T, M>::eval(a + lda * j, x);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_t_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (incx == 1)
			{
				eval_t_b0_(alpha, a, lda, x, y, incy);
			}
			else
			{
				small_cache<T, M> cache_x(x, incx);
				eval_t_b0_(alpha, a, lda, cache_x.data, y, incy);
			}

		}


		BCS_ENSURE_INLINE
		static void eval_t_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					y[j] += smalldot<T, M>::eval(a + lda * j, x);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					y[j] += alpha * smalldot<T, M>::eval(a + lda * j, x);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_t_b1_(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y, const int incy)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					y[j * incy] += smalldot<T, M>::eval(a + lda * j, x);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					y[j * incy] += alpha * smalldot<T, M>::eval(a + lda * j, x);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_t_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (incx == 1)
			{
				eval_t_b1_(alpha, a, lda, x, y, incy);
			}
			else
			{
				small_cache<T, M> cache_x(x, incx);
				eval_t_b1_(alpha, a, lda, cache_x.data, y, incy);
			}
		}
	};


	template<typename T, int M, int N>
	struct smallgemv
	{
		typedef typename select_type<N == 1,
					smallgemv_M1<M, 1>,
					typename select_type<M == 1,
						smallgemv_1N<T, N>,
						smallgemv_MN<T, M, N>
					>::type
				>::type impl_t;

		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			impl_t::eval_b0(alpha, a, lda, x, y);
		}

		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			impl_t::eval_b0(alpha, a, lda, x, incx, y, incy);
		}


		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			impl_t::eval_b1(alpha, a, lda, x, y);
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			impl_t::eval_b1(alpha, a, lda, x, incx, y, incy);
		}


		BCS_ENSURE_INLINE
		static void eval_t_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			impl_t::eval_t_b0(alpha, a, lda, x, y);
		}

		BCS_ENSURE_INLINE
		static void eval_t_b0(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			impl_t::eval_t_b0(alpha, a, lda, x, incx, y, incy);
		}

		BCS_ENSURE_INLINE
		static void eval_t_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			impl_t::eval_t_b1(alpha, a, lda, x, y);
		}

		BCS_ENSURE_INLINE
		static void eval_t_b1(const T alpha, const T* __restrict__ a, const index_t lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			impl_t::eval_t_b1(alpha, a, lda, x, incx, y, incy);
		}
	};


	/********************************************
	 *
	 *  small ger
	 *
	 ********************************************/

	template<typename T>
	struct smallger_11
	{
		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			a[0] += alpha * (x[0] * y[0]);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			a[0] += alpha * (x[0] * y[0]);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y, const int,
				T *__restrict__ a, const int lda)
		{
			a[0] += alpha * (x[0] * y[0]);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int, const T* __restrict__ y, const int,
				T *__restrict__ a, const int lda)
		{
			a[0] += alpha * (x[0] * y[0]);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			a[0] = alpha * (x[0] * y[0]);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			a[0] = alpha * (x[0] * y[0]);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y, const int,
				T *__restrict__ a, const int lda)
		{
			a[0] = alpha * (x[0] * y[0]);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int, const T* __restrict__ y, const int,
				T *__restrict__ a, const int lda)
		{
			a[0] = alpha * (x[0] * y[0]);
		}
	};


	template<typename T, int N>
	struct smallger_1N
	{
		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
			{
				smallaxpy<T, N>::eval(x[0], y, a);
			}
			else
			{
				smallaxpy<T, N>::eval(x[0], y, a, lda);
			}
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
			{
				smallaxpy<T, N>::eval(x[0], y, a);
			}
			else
			{
				smallaxpy<T, N>::eval(x[0], y, a, lda);
			}
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
			{
				smallaxpy<T, N>::eval(x[0], y, incy, a);
			}
			else
			{
				smallaxpy<T, N>::eval(x[0], y, incy, a, lda);
			}
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
			{
				smallaxpy<T, N>::eval(x[0], y, incy, a);
			}
			else
			{
				smallaxpy<T, N>::eval(x[0], y, incy, a, lda);
			}
		}


		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
			{
				smallmul<T, N>::eval(x[0], y, a);
			}
			else
			{
				smallmul<T, N>::eval(x[0], y, a, lda);
			}
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
			{
				smallmul<T, N>::eval(x[0], y, a);
			}
			else
			{
				smallmul<T, N>::eval(x[0], y, a, lda);
			}
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
			{
				smallmul<T, N>::eval(x[0], y, incy, a);
			}
			else
			{
				smallmul<T, N>::eval(x[0], y, incy, a, lda);
			}
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
			{
				smallmul<T, N>::eval(x[0], y, incy, a);
			}
			else
			{
				smallmul<T, N>::eval(x[0], y, incy, a, lda);
			}
		}
	};


	template<typename T, int M>
	struct smallger_M1
	{
		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			smallaxpy<T, M>::eval(y[0], x, a);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			smallaxpy<T, M>::eval(y[0], x, incx, a);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y, const int,
				T *__restrict__ a, const int lda)
		{
			smallaxpy<T, M>::eval(y[0], x, a);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y, const int,
				T *__restrict__ a, const int lda)
		{
			smallaxpy<T, M>::eval(y[0], x, incx, a);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			smallmul<T, M>::eval(y[0], x, a);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			smallmul<T, M>::eval(y[0], x, incx, a);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y, const int,
				T *__restrict__ a, const int lda)
		{
			smallmul<T, M>::eval(y[0], x, a);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y, const int,
				T *__restrict__ a, const int lda)
		{
			smallmul<T, M>::eval(y[0], x, incx, a);
		}
	};


	template<typename T, int M, int N>
	struct smallger_MN
	{
		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					smallaxpy<T, M>::eval(a + lda * j, x, y[j]);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					smallaxpy<T, M>::eval(a + lda * j, x, alpha * y[j]);
			}
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			small_cache<T, M> cache_x(x, incx);
			eval(alpha, cache_x.data, y, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					smallaxpy<T, M>::eval(a + lda * j, x, y[j * incy]);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					smallaxpy<T, M>::eval(a + lda * j, x, alpha * y[j * incy]);
			}
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			small_cache<T, M> cache_x(x, incx);
			eval(alpha, cache_x.data, y, incy, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					smallmul<T, M>::eval(a + lda * j, x, y[j]);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					smallmul<T, M>::eval(a + lda * j, x, alpha * y[j]);
			}
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			small_cache<T, M> cache_x(x, incx);
			eval0(alpha, cache_x.data, y, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					smallmul<T, M>::eval(a + lda * j, x, y[j * incy]);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					smallmul<T, M>::eval(a + lda * j, x, alpha * y[j * incy]);
			}
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			small_cache<T, M> cache_x(x, incx);
			eval0(alpha, cache_x.data, y, incy, a, lda);
		}
	};


	template<typename T, int M, int N>
	struct smallger
	{
		typedef typename select_type<M == 1,
					typename select_type<N == 1,
						smallger_11<T>,
						smallger_1N<T, N>
					>::type,
					typename select_type<N == 1,
						smallger_M1<T, M>,
						smallger_MN<T, M, N>
					>::type
				>::type impl_t;

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			impl_t::eval(alpha, x, y, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			impl_t::eval(alpha, x, incx, y, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			impl_t::eval(alpha, x, y, incy, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			impl_t::eval(alpha, x, incx, y, incy, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			impl_t::eval0(alpha, x, y, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			impl_t::eval0(alpha, x, incx, y, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			impl_t::eval0(alpha, x, y, incy, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			impl_t::eval0(alpha, x, incx, y, incy, a, lda);
		}
	};



} }

#endif /* SMALL_BLASL2_H_ */
