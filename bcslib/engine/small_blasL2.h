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


	/********************************************
	 *
	 *  small gemv
	 *
	 ********************************************/

	template<typename T, int M>
	struct small_gemv_n_M1
	{
		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			mul_ker<T, M>::eval(alpha * x[0], a, y);
		}

		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			mul_ker<T, M>::eval(alpha * x[0], a, y, incy);
		}


		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			addx_ker<T, M>::eval(alpha * x[0], a, y);
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			addx_ker<T, M>::eval(alpha * x[0], a, y, incy);
		}
	};


	template<typename T, int M>
	struct small_gemv_t_M1
	{
		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			y[0] = alpha * dot_ker<T, M>::eval(a, x);
		}

		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			y[0] = alpha * dot_ker<T, M>::eval(a, x, incx);
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			y[0] += alpha * dot_ker<T, M>::eval(a, x);
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			y[0] += alpha * dot_ker<T, M>::eval(a, x, incx);
		}
	};


	template<typename T, int N>
	struct small_gemv_n_1N
	{
		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (lda == 1)
			{
				y[0] = alpha * dot_ker<T, N>::eval(a, x);
			}
			else
			{
				y[0] = alpha * dot_ker<T, N>::eval(a, lda, x);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (lda == 1)
			{
				y[0] = alpha * dot_ker<T, N>::eval(a, x, incx);
			}
			else
			{
				y[0] = alpha * dot_ker<T, N>::eval(a, lda, x, incx);
			}
		}


		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (lda == 1)
			{
				y[0] += alpha * dot_ker<T, N>::eval(a, x);
			}
			else
			{
				y[0] += alpha * dot_ker<T, N>::eval(a, lda, x);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (lda == 1)
			{
				y[0] += alpha * dot_ker<T, N>::eval(a, x, incx);
			}
			else
			{
				y[0] += alpha * dot_ker<T, N>::eval(a, lda, x, incx);
			}
		}
	};


	template<typename T, int N>
	struct small_gemv_t_1N
	{
		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (lda == 1)
			{
				mul_ker<T, N>::eval(alpha * x[0], a, y);
			}
			else
			{
				mul_ker<T, N>::eval(alpha * x[0], a, lda, y);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (lda == 1)
			{
				mul_ker<T, N>::eval(alpha * x[0], a, y, incy);
			}
			else
			{
				mul_ker<T, N>::eval(alpha * x[0], a, lda, y, incy);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (lda == 1)
			{
				addx_ker<T, N>::eval(alpha * x[0], a, y);
			}
			else
			{
				addx_ker<T, N>::eval(alpha * x[0], a, lda, y);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (lda == 1)
			{
				addx_ker<T, N>::eval(alpha * x[0], a, y, incy);
			}
			else
			{
				addx_ker<T, N>::eval(alpha * x[0], a, lda, y, incy);
			}
		}
	};


	template<typename T, int M, int N>
	struct small_gemv_n_MN
	{
		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (alpha == 1)
			{
				mul_ker<T, M>::eval(x[0], a, y);

				for (int j = 1; j < N; ++j)
					addx_ker<T, M>::eval(x[j], a + lda * j, y);
			}
			else
			{
				mul_ker<T, M>::eval(alpha * x[0], a, y);

				for (int j = 1; j < N; ++j)
					addx_ker<T, M>::eval(alpha * x[j], a + lda * j, y);
			}
		}


		BCS_ENSURE_INLINE
		static void eval_b0_(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y)
		{
			if (alpha == 1)
			{
				mul_ker<T, M>::eval(x[0], a, y);

				for (int j = 1; j < N; ++j)
					addx_ker<T, M>::eval(x[j * incx], a + lda * j, y);
			}
			else
			{
				mul_ker<T, M>::eval(alpha * x[0], a, y);

				for (int j = 1; j < N; ++j)
					addx_ker<T, M>::eval(alpha * x[j * incx], a + lda * j, y);
			}
		}


		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
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
				copy_ker<T, M>::eval(cache_y.data, y, incy);
			}
		}


		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					addx_ker<T, M>::eval(x[j], a + lda * j, y);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					addx_ker<T, M>::eval(alpha * x[j], a + lda * j, y);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b1_(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					addx_ker<T, M>::eval(x[j * incx], a + lda * j, y);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					addx_ker<T, M>::eval(alpha * x[j * incx], a + lda * j, y);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
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
				add_ker<T, M>::eval(cache_y.data, y, incy);
			}
		}
	};


	template<typename T, int M, int N>
	struct small_gemv_t_MN
	{
		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					y[j] = dot_ker<T, M>::eval(a + lda * j, x);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					y[j] = alpha * dot_ker<T, M>::eval(a + lda * j, x);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b0_(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y, const int incy)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					y[j * incy] = dot_ker<T, M>::eval(a + lda * j, x);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					y[j * incy] = alpha * dot_ker<T, M>::eval(a + lda * j, x);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (incx == 1)
			{
				eval_b0_(alpha, a, lda, x, y, incy);
			}
			else
			{
				small_cache<T, M> cache_x;
				copy_ker<T, M>::eval(x, incx, cache_x.data);
				eval_b0_(alpha, a, lda, cache_x.data, y, incy);
			}

		}


		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					y[j] += dot_ker<T, M>::eval(a + lda * j, x);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					y[j] += alpha * dot_ker<T, M>::eval(a + lda * j, x);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b1_(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y, const int incy)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					y[j * incy] += dot_ker<T, M>::eval(a + lda * j, x);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					y[j * incy] += alpha * dot_ker<T, M>::eval(a + lda * j, x);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			if (incx == 1)
			{
				eval_b1_(alpha, a, lda, x, y, incy);
			}
			else
			{
				small_cache<T, M> cache_x;
				copy_ker<T, M>::eval(x, incx, cache_x.data);
				eval_b1_(alpha, a, lda, cache_x.data, y, incy);
			}
		}
	};


	template<typename T, int M, int N>
	struct small_gemv_n_ker
	{
		typedef typename select_type<N == 1,
					small_gemv_n_M1<T, M>,
					typename select_type<M == 1,
						small_gemv_n_1N<T, N>,
						small_gemv_n_MN<T, M, N>
					>::type
				>::type impl_t;

		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			impl_t::eval_b0(alpha, a, lda, x, y);
		}

		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			impl_t::eval_b0(alpha, a, lda, x, incx, y, incy);
		}


		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			impl_t::eval_b1(alpha, a, lda, x, y);
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			impl_t::eval_b1(alpha, a, lda, x, incx, y, incy);
		}
	};


	template<typename T, int M, int N>
	struct small_gemv_t_ker
	{
		typedef typename select_type<N == 1,
					small_gemv_t_M1<T, M>,
					typename select_type<M == 1,
						small_gemv_t_1N<T, N>,
						small_gemv_t_MN<T, M, N>
					>::type
				>::type impl_t;

		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			impl_t::eval_b0(alpha, a, lda, x, y);
		}

		BCS_ENSURE_INLINE
		static void eval_b0(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			impl_t::eval_b0(alpha, a, lda, x, incx, y, incy);
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, T* __restrict__ y)
		{
			impl_t::eval_b1(alpha, a, lda, x, y);
		}

		BCS_ENSURE_INLINE
		static void eval_b1(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx, T* __restrict__ y, const int incy)
		{
			impl_t::eval_b1(alpha, a, lda, x, incx, y, incy);
		}
	};


	template<typename T, int M, int N>
	struct small_gemv_n
	{
		inline
		static void eval(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx,
				const T beta, T* __restrict__ y, const int incy)
		{
			if (beta == 0)
			{
				if (incx == 1 && incy == 1)
					small_gemv_n_ker<T, M, N>::eval_b0(alpha, a, lda, x, y);
				else
					small_gemv_n_ker<T, M, N>::eval_b0(alpha, a, lda, x, incx, y, incy);
			}
			else
			{
				if (beta != 1)
				{
					if (incy == 1) mul_ker<T, M>::eval(beta, y);
					else mul_ker<T, M>::eval(beta, y, incy);
				}

				if (incx == 1 && incy == 1)
					small_gemv_n_ker<T, M, N>::eval_b1(alpha, a, lda, x, y);
				else
					small_gemv_n_ker<T, M, N>::eval_b1(alpha, a, lda, x, incx, y, incy);
			}
		}
	};

	template<typename T, int M, int N>
	struct small_gemv_t
	{
		inline
		static void eval(const T alpha, const T* __restrict__ a, const int lda,
				const T* __restrict__ x, const int incx,
				const T beta, T* __restrict__ y, const int incy)
		{
			if (beta == 0)
			{
				if (incx == 1 && incy == 1)
					small_gemv_t_ker<T, M, N>::eval_b0(alpha, a, lda, x, y);
				else
					small_gemv_t_ker<T, M, N>::eval_b0(alpha, a, lda, x, incx, y, incy);
			}
			else
			{
				if (beta != 1)
				{
					if (incy == 1) mul_ker<T, N>::eval(beta, y);
					else mul_ker<T, N>::eval(beta, y, incy);
				}

				if (incx == 1 && incy == 1)
					small_gemv_t_ker<T, M, N>::eval_b1(alpha, a, lda, x, y);
				else
					small_gemv_t_ker<T, M, N>::eval_b1(alpha, a, lda, x, incx, y, incy);
			}
		}
	};



	/********************************************
	 *
	 *  small ger
	 *
	 ********************************************/

	template<typename T>
	struct small_ger_11
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
	struct small_ger_1N
	{
		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
				addx_ker<T, N>::eval(alpha * x[0], y, a);
			else
				addx_ker<T, N>::eval(alpha * x[0], y, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
				addx_ker<T, N>::eval(alpha * x[0], y, a);
			else
				addx_ker<T, N>::eval(alpha * x[0], y, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
				addx_ker<T, N>::eval(alpha * x[0], y, incy, a);
			else
				addx_ker<T, N>::eval(alpha * x[0], y, incy, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
				addx_ker<T, N>::eval(alpha * x[0], y, incy, a);
			else
				addx_ker<T, N>::eval(alpha * x[0], y, incy, a, lda);
		}


		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
				mul_ker<T, N>::eval(alpha * x[0], y, a);
			else
				mul_ker<T, N>::eval(alpha * x[0], y, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
				mul_ker<T, N>::eval(alpha * x[0], y, a);
			else
				mul_ker<T, N>::eval(alpha * x[0], y, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
				mul_ker<T, N>::eval(alpha * x[0], y, incy, a);
			else
				mul_ker<T, N>::eval(alpha * x[0], y, incy, a, lda);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			if (lda == 1)
				mul_ker<T, N>::eval(alpha * x[0], y, incy, a);
			else
				mul_ker<T, N>::eval(alpha * x[0], y, incy, a, lda);
		}
	};


	template<typename T, int M>
	struct small_ger_M1
	{
		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			addx_ker<T, M>::eval(alpha * y[0], x, a);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			addx_ker<T, M>::eval(alpha * y[0], x, incx, a);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y, const int,
				T *__restrict__ a, const int lda)
		{
			addx_ker<T, M>::eval(alpha * y[0], x, a);
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y, const int,
				T *__restrict__ a, const int lda)
		{
			addx_ker<T, M>::eval(alpha * y[0], x, incx, a);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			mul_ker<T, M>::eval(alpha * y[0], x, a);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			mul_ker<T, M>::eval(alpha * y[0], x, incx, a);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y, const int,
				T *__restrict__ a, const int lda)
		{
			mul_ker<T, M>::eval(alpha * y[0], x, a);
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y, const int,
				T *__restrict__ a, const int lda)
		{
			mul_ker<T, M>::eval(alpha * y[0], x, incx, a);
		}
	};


	template<typename T, int M, int N>
	struct small_ger_MN
	{
		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
					addx_ker<T, M>::eval(y[j], x, a + lda * j);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					addx_ker<T, M>::eval(alpha * y[j], x, a + lda * j);
			}
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			small_cache<T, M> cache_x;
			copy_ker<T, M>::eval(x, incx, cache_x.data);
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
					addx_ker<T, M>::eval(y[j * incy], x, a + lda * j);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					addx_ker<T, M>::eval(alpha * y[j * incy], x, a + lda * j);
			}
		}

		BCS_ENSURE_INLINE
		static void eval(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			small_cache<T, M> cache_x;
			copy_ker<T, M>::eval(x, incx, cache_x.data);
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
					mul_ker<T, M>::eval(y[j], x, a + lda * j);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					mul_ker<T, M>::eval(alpha * y[j], x, a + lda * j);
			}
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y,
				T *__restrict__ a, const int lda)
		{
			small_cache<T, M> cache_x;
			copy_ker<T, M>::eval(x, incx, cache_x.data);
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
					mul_ker<T, M>::eval(y[j * incy], x, a + lda * j);
			}
			else
			{
				for (int j = 0; j < N; ++j)
					mul_ker<T, M>::eval(alpha * y[j * incy], x, a + lda * j);
			}
		}

		BCS_ENSURE_INLINE
		static void eval0(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			small_cache<T, M> cache_x;
			copy_ker<T, M>::eval(x, incx, cache_x.data);
			eval0(alpha, cache_x.data, y, incy, a, lda);
		}
	};


	template<typename T, int M, int N>
	struct small_ger_ker
	{
		typedef typename select_type<M == 1,
					typename select_type<N == 1,
						small_ger_11<T>,
						small_ger_1N<T, N>
					>::type,
					typename select_type<N == 1,
						small_ger_M1<T, M>,
						small_ger_MN<T, M, N>
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


	template<typename T, int M, int N>
	struct small_ger
	{
		static void eval(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			if (incx == 1)
			{
				if (incy == 1)
					small_ger_ker<T, M, N>::eval(alpha, x, y, a, lda);
				else
					small_ger_ker<T, M, N>::eval(alpha, x, y, incy, a, lda);
			}
			else
			{
				if (incy == 1)
					small_ger_ker<T, M, N>::eval(alpha, x, incx, y, a, lda);
				else
					small_ger_ker<T, M, N>::eval(alpha, x, incx, y, incy, a, lda);
			}
		}

		static void eval0(const T alpha,
				const T* __restrict__ x, const int incx, const T* __restrict__ y, const int incy,
				T *__restrict__ a, const int lda)
		{
			if (incx == 1)
			{
				if (incy == 1)
					small_ger_ker<T, M, N>::eval0(alpha, x, y, a, lda);
				else
					small_ger_ker<T, M, N>::eval0(alpha, x, y, incy, a, lda);
			}
			else
			{
				if (incy == 1)
					small_ger_ker<T, M, N>::eval0(alpha, x, incx, y, a, lda);
				else
					small_ger_ker<T, M, N>::eval0(alpha, x, incx, y, incy, a, lda);
			}
		}
	};



} }

#endif /* SMALL_BLASL2_H_ */
