/**
 * @file small_blasL3.h
 *
 *  
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef SMALL_BLASL3_H_
#define SMALL_BLASL3_H_

#include "small_blasL2.h"

namespace bcs { namespace engine {

	template<typename T, int K>
	struct small_gemm_11K  // (1 x K) * (K x 1)
	{
		BCS_ENSURE_INLINE
		static void eval_nn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (lda == 1)
			{
				c[0] = alpha * dot_ker<T, K>::eval(a, b);
			}
			else
			{
				c[0] = alpha * dot_ker<T, K>::eval(a, lda, b);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_nn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (lda == 1)
			{
				c[0] += alpha * dot_ker<T, K>::eval(a, b);
			}
			else
			{
				c[0] += alpha * dot_ker<T, K>::eval(a, lda, b);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_nt_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			c[0] = alpha * small_dot<T, K>::eval(a, lda, b, ldb);
		}

		BCS_ENSURE_INLINE
		static void eval_nt_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			c[0] += alpha * small_dot<T, K>::eval(a, lda, b, ldb);
		}

		BCS_ENSURE_INLINE
		static void eval_tn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			c[0] = alpha * dot_ker<T, K>::eval(a, b);
		}

		BCS_ENSURE_INLINE
		static void eval_tn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			c[0] += alpha * dot_ker<T, K>::eval(a, b);
		}

		BCS_ENSURE_INLINE
		static void eval_tt_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (ldb == 1)
			{
				c[0] = alpha * dot_ker<T, K>::eval(a, b);
			}
			else
			{
				c[0] = alpha * dot_ker<T, K>::eval(a, b, ldb);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_tt_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (ldb == 1)
			{
				c[0] += alpha * dot_ker<T, K>::eval(a, b);
			}
			else
			{
				c[0] += alpha * dot_ker<T, K>::eval(a, b, ldb);
			}
		}
	};



	template<typename T, int N, int K>
	struct small_gemm_1NK  // (1 x K) * (K x N)
	{
		BCS_ENSURE_INLINE
		static void eval_nn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (lda == 1 && ldc == 1)
			{
				small_gemv_t_MN<T, K, N>::eval_b0(alpha, b, ldb, a, c);
			}
			else
			{
				small_gemv_t_MN<T, K, N>::eval_b0(alpha, b, ldb, a, lda, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_nn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (lda == 1 && ldc == 1)
			{
				small_gemv_t_MN<T, K, N>::eval_b1(alpha, b, ldb, a, c);
			}
			else
			{
				small_gemv_t_MN<T, K, N>::eval_b1(alpha, b, ldb, a, lda, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_nt_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (lda == 1 && ldc == 1)
			{
				small_gemv_n_MN<T, N, K>::eval_b0(alpha, b, ldb, a, c);
			}
			else
			{
				small_gemv_n_MN<T, N, K>::eval_b0(alpha, b, ldb, a, lda, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_nt_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (lda == 1 && ldc == 1)
			{
				small_gemv_n_MN<T, N, K>::eval_b1(alpha, b, ldb, a, c);
			}
			else
			{
				small_gemv_n_MN<T, N, K>::eval_b1(alpha, b, ldb, a, lda, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_tn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (ldc == 1)
			{
				small_gemv_t_MN<T, K, N>::eval_b0(alpha, b, ldb, a, c);
			}
			else
			{
				small_gemv_t_MN<T, K, N>::eval_b0_(alpha, b, ldb, a, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_tn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (ldc == 1)
			{
				small_gemv_t_MN<T, K, N>::eval_b1(alpha, b, ldb, a, c);
			}
			else
			{
				small_gemv_t_MN<T, K, N>::eval_b1_(alpha, b, ldb, a, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_tt_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (ldc == 1)
			{
				small_gemv_n_MN<T, N, K>::eval_b0(alpha, b, ldb, a, c);
			}
			else
			{
				small_gemv_n_MN<T, N, K>::eval_b0(alpha, b, ldb, a, 1, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_tt_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (ldc == 1)
			{
				small_gemv_n_MN<T, N, K>::eval_b1(alpha, b, ldb, a, c);
			}
			else
			{
				small_gemv_n_MN<T, N, K>::eval_b1(alpha, b, ldb, a, 1, c, ldc);
			}
		}
	};



	template<typename T, int M, int K>
	struct small_gemm_M1K  // (M x K) * (K x 1)
	{
		BCS_ENSURE_INLINE
		static void eval_nn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			small_gemv_n_MN<T, M, K>::eval_b0(alpha, a, lda, b, c);
		}

		BCS_ENSURE_INLINE
		static void eval_nn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			small_gemv_n_MN<T, M, K>::eval_b1(alpha, a, lda, b, c);
		}

		BCS_ENSURE_INLINE
		static void eval_nt_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (ldb == 1)
			{
				small_gemv_n_MN<T, M, K>::eval_b0(alpha, a, lda, b, c);
			}
			else
			{
				small_gemv_n_MN<T, M, K>::eval_b0_(alpha, a, lda, b, ldb, c);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_nt_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (ldb == 1)
			{
				small_gemv_n_MN<T, M, K>::eval_b1(alpha, a, lda, b, c);
			}
			else
			{
				small_gemv_n_MN<T, M, K>::eval_b1_(alpha, a, lda, b, ldb, c);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_tn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			small_gemv_t_MN<T, K, M>::eval_b0(alpha, a, lda, b, c);
		}

		BCS_ENSURE_INLINE
		static void eval_tn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			small_gemv_t_MN<T, K, M>::eval_b1(alpha, a, lda, b, c);
		}

		BCS_ENSURE_INLINE
		static void eval_tt_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (ldb == 1)
			{
				small_gemv_t_MN<T, K, M>::eval_b0(alpha, a, lda, b, c);
			}
			else
			{
				small_gemv_t_MN<T, K, M>::eval_b0(alpha, a, lda, b, ldb, c, 1);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_tt_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (ldb == 1)
			{
				small_gemv_t_MN<T, K, M>::eval_b1(alpha, a, lda, b, c);
			}
			else
			{
				small_gemv_t_MN<T, K, M>::eval_b1(alpha, a, lda, b, ldb, c, 1);
			}
		}
	};



	template<typename T, int M, int N, int K>
	struct small_gemm_MNK  // (M x K) * (K x N)
	{
		BCS_ENSURE_INLINE
		static void eval_nn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			for (int j = 0; j < N; ++j)
			{
				small_gemv_n_MN<T, M, K>::eval_b0(alpha, a, lda, b + j * ldb, c + j * ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_nn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			for (int j = 0; j < N; ++j)
			{
				small_gemv_n_MN<T, M, K>::eval_b1(alpha, a, lda, b + j * ldb, c + j * ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_nt_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			small_ger_MN<T, M, N>::eval0(alpha, a, b, c, ldc);

			for (int k = 1; k < K; ++k)
			{
				small_ger_MN<T, M, N>::eval(alpha, a + k * lda, b + k * ldb, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_nt_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			for (int k = 0; k < K; ++k)
			{
				small_ger_MN<T, M, N>::eval(alpha, a + k * lda, b + k * ldb, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_tn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			for (int j = 0; j < N; ++j)
			{
				small_gemv_t_MN<T, K, M>::eval_b0(alpha, a, lda, b + j * ldb, c + j * ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_tn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			for (int j = 0; j < N; ++j)
			{
				small_gemv_t_MN<T, K, M>::eval_b1(alpha, a, lda, b + j * ldb, c + j * ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_tt_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			small_cache<T, M> cache_a;

			copy_ker<T, M>::eval(a, lda, cache_a.data);
			small_ger_MN<T, M, N>::eval0(alpha, cache_a.data, b, c, ldc);

			for (int k = 1; k < K; ++k)
			{
				copy_ker<T, M>::eval(a + k, lda, cache_a.data);
				small_ger_MN<T, M, N>::eval(alpha, cache_a.data, b + k * ldb, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_tt_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			small_cache<T, M> cache_a;

			for (int k = 0; k < K; ++k)
			{
				copy_ker<T, M>::eval(a + k, lda, cache_a.data);
				small_ger_MN<T, M, N>::eval(alpha, cache_a.data, b + k * ldb, c, ldc);
			}
		}
	};


	template<typename T, int M, int N>
	struct small_gemm_MN1  // (M x 1) * (1 x N)
	{
		BCS_ENSURE_INLINE
		static void eval_nn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (ldb == 1)
			{
				small_ger_MN<T, M, N>::eval0(alpha, a, b, c, ldc);
			}
			else
			{
				small_ger_MN<T, M, N>::eval0(alpha, a, b, ldb, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_nn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (ldb == 1)
			{
				small_ger_MN<T, M, N>::eval(alpha, a, b, c, ldc);
			}
			else
			{
				small_ger_MN<T, M, N>::eval(alpha, a, b, ldb, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_nt_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			small_ger_MN<T, M, N>::eval0(alpha, a, b, c, ldc);
		}

		BCS_ENSURE_INLINE
		static void eval_nt_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			small_ger_MN<T, M, N>::eval(alpha, a, b, c, ldc);
		}

		BCS_ENSURE_INLINE
		static void eval_tn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (lda == 1)
			{
				if (ldb == 1)
				{
					small_ger_MN<T, M, N>::eval0(alpha, a, b, c, ldc);
				}
				else
				{
					small_ger_MN<T, M, N>::eval0(alpha, a, b, ldb, c, ldc);
				}
			}
			else
			{
				if (ldb == 1)
				{
					small_ger_MN<T, M, N>::eval0(alpha, a, lda, b, c, ldc);
				}
				else
				{
					small_ger_MN<T, M, N>::eval0(alpha, a, lda, b, ldb, c, ldc);
				}
			}
		}

		BCS_ENSURE_INLINE
		static void eval_tn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (lda == 1)
			{
				if (ldb == 1)
				{
					small_ger_MN<T, M, N>::eval(alpha, a, b, c, ldc);
				}
				else
				{
					small_ger_MN<T, M, N>::eval(alpha, a, b, ldb, c, ldc);
				}
			}
			else
			{
				if (ldb == 1)
				{
					small_ger_MN<T, M, N>::eval(alpha, a, lda, b, c, ldc);
				}
				else
				{
					small_ger_MN<T, M, N>::eval(alpha, a, lda, b, ldb, c, ldc);
				}
			}
		}

		BCS_ENSURE_INLINE
		static void eval_tt_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (lda == 1)
			{
				small_ger_MN<T, M, N>::eval0(alpha, a, b, c, ldc);
			}
			else
			{
				small_ger_MN<T, M, N>::eval0(alpha, a, lda, b, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		static void eval_tt_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (lda == 1)
			{
				small_ger_MN<T, M, N>::eval(alpha, a, b, c, ldc);
			}
			else
			{
				small_ger_MN<T, M, N>::eval(alpha, a, lda, b, c, ldc);
			}
		}
	};



	template<typename T, int M, int N, int K>
	struct small_gemm_ker
	{
		typedef typename select_type<M == 1,
					typename select_type<N == 1,
						small_gemm_11K<T, K>,
						small_gemm_1NK<T, N, K>
					>::type,
					typename select_type<N == 1,
						small_gemm_M1K<T, M, K>,
						typename select_type<K == 1,
							small_gemm_MN1<T, M, N>,
							small_gemm_MNK<T, M, N, K>
						>::type
					>::type
				>::type impl_t;

		BCS_ENSURE_INLINE
		static void eval_nn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			impl_t::eval_nn_b0(alpha, a, lda, b, ldb, c, ldc);
		}

		BCS_ENSURE_INLINE
		static void eval_nn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			impl_t::eval_nn_b1(alpha, a, lda, b, ldb, c, ldc);
		}

		BCS_ENSURE_INLINE
		static void eval_nt_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			impl_t::eval_nt_b0(alpha, a, lda, b, ldb, c, ldc);
		}

		BCS_ENSURE_INLINE
		static void eval_nt_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			impl_t::eval_nt_b1(alpha, a, lda, b, ldb, c, ldc);
		}

		BCS_ENSURE_INLINE
		static void eval_tn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			impl_t::eval_tn_b0(alpha, a, lda, b, ldb, c, ldc);
		}

		BCS_ENSURE_INLINE
		static void eval_tn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			impl_t::eval_tn_b1(alpha, a, lda, b, ldb, c, ldc);
		}

		BCS_ENSURE_INLINE
		static void eval_tt_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			impl_t::eval_tt_b0(alpha, a, lda, b, ldb, c, ldc);
		}

		BCS_ENSURE_INLINE
		static void eval_tt_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			impl_t::eval_tt_b1(alpha, a, lda, b, ldb, c, ldc);
		}
	};


	template<typename T, int M, int N, int K>
	struct small_gemm
	{
		static void eval_nn(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				const T beta,
				T* __restrict__ c, const int ldc)
		{
			if (beta == 0)
			{
				small_gemm_ker<T, M, N, K>::eval_nn_b0(alpha, a, lda, b, ldb, c, ldc);
			}
			else
			{
				if (beta != 1)
				{
					for (int j = 0; j < N; ++j) mul_ker<T, M>::eval(beta, c + ldc * j);
				}

				small_gemm_ker<T, M, N, K>::eval_nn_b1(alpha, a, lda, b, ldb, c, ldc);
			}
		}

		static void eval_nt(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				const T beta,
				T* __restrict__ c, const int ldc)
		{
			if (beta == 0)
			{
				small_gemm_ker<T, M, N, K>::eval_nt_b0(alpha, a, lda, b, ldb, c, ldc);
			}
			else
			{
				if (beta != 1)
				{
					for (int j = 0; j < N; ++j) mul_ker<T, M>::eval(beta, c + ldc * j);
				}

				small_gemm_ker<T, M, N, K>::eval_nt_b1(alpha, a, lda, b, ldb, c, ldc);
			}
		}

		static void eval_tn(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				const T beta,
				T* __restrict__ c, const int ldc)
		{
			if (beta == 0)
			{
				small_gemm_ker<T, M, N, K>::eval_tn_b0(alpha, a, lda, b, ldb, c, ldc);
			}
			else
			{
				if (beta != 1)
				{
					for (int j = 0; j < N; ++j) mul_ker<T, M>::eval(beta, c + ldc * j);
				}

				small_gemm_ker<T, M, N, K>::eval_tn_b1(alpha, a, lda, b, ldb, c, ldc);
			}
		}

		static void eval_tt(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				const T beta,
				T* __restrict__ c, const int ldc)
		{
			if (beta == 0)
			{
				small_gemm_ker<T, M, N, K>::eval_tt_b0(alpha, a, lda, b, ldb, c, ldc);
			}
			else
			{
				if (beta != 1)
				{
					for (int j = 0; j < N; ++j) mul_ker<T, M>::eval(beta, c + ldc * j);
				}

				small_gemm_ker<T, M, N, K>::eval_tt_b1(alpha, a, lda, b, ldb, c, ldc);
			}
		}

	};


} }


#endif /* SMALL_BLASL3_H_ */
