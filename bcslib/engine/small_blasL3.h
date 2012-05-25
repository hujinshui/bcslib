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
	struct smallgemm_11K  // (1 x K) * (K x 1)
	{
		BCS_ENSURE_INLINE
		void eval_nn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (lda == 1)
			{
				c[0] = alpha * smalldot<T, K>::eval(a, b);
			}
			else
			{
				c[0] = alpha * smalldot<T, K>::eval(a, lda, b);
			}
		}

		BCS_ENSURE_INLINE
		void eval_nn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (lda == 1)
			{
				c[0] += alpha * smalldot<T, K>::eval(a, b);
			}
			else
			{
				c[0] += alpha * smalldot<T, K>::eval(a, lda, b);
			}
		}
	};


	template<typename T, int N, int K>
	struct smallgemm_1NK  // (1 x K) * (K x N)
	{
		BCS_ENSURE_INLINE
		void eval_nn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (lda == 1 && ldc == 1)
			{
				smallgemv_MN<T, K, N>::eval_t_b0(alpha, b, ldb, a, c);
			}
			else
			{
				smallgemv_MN<T, K, N>::eval_t_b0(alpha, b, ldb, a, lda, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		void eval_nn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (lda == 1 && ldc == 1)
			{
				smallgemv_MN<T, K, N>::eval_t_b1(alpha, b, ldb, a, c);
			}
			else
			{
				smallgemv_MN<T, K, N>::eval_t_b1(alpha, b, ldb, a, lda, c, ldc);
			}
		}
	};


	template<typename T, int M, int K>
	struct smallgemm_M1K  // (M x K) * (K x 1)
	{
		BCS_ENSURE_INLINE
		void eval_nn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			smallgemv_MN<T, M, K>::eval_b0(alpha, a, lda, b, c);
		}

		BCS_ENSURE_INLINE
		void eval_nn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			smallgemv_MN<T, M, K>::eval_b1(alpha, a, lda, b, c);
		}
	};


	template<typename T, int M, int N, int K>
	struct smallgemm_MNK  // (M x K) * (K x N)
	{
		BCS_ENSURE_INLINE
		void eval_nn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			for (int j = 0; j < N; ++j)
			{
				smallgemv_MN<T, M, K>::eval_b0(alpha, a, lda, b + j * ldb, c + j * ldc);
			}
		}

		BCS_ENSURE_INLINE
		void eval_nn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			for (int j = 0; j < N; ++j)
			{
				smallgemv_MN<T, M, K>::eval_b1(alpha, a, lda, b + j * ldb, c + j * ldc);
			}
		}
	};


	template<typename T, int M, int N>
	struct smallgemm_MN1  // (M x 1) * (1 x N)
	{
		BCS_ENSURE_INLINE
		void eval_nn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (ldb == 1)
			{
				smallger_MN<T, M, N>::eval0(alpha, a, b, c, ldc);
			}
			else
			{
				smallger_MN<T, M, N>::eval0(alpha, a, b, ldb, c, ldc);
			}
		}

		BCS_ENSURE_INLINE
		void eval_nn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			if (ldb == 1)
			{
				smallger_MN<T, M, N>::eval(alpha, a, b, c, ldc);
			}
			else
			{
				smallger_MN<T, M, N>::eval(alpha, a, b, ldb, c, ldc);
			}
		}
	};


	template<typename T, int M, int N, int K>
	struct smallgemm
	{
		typedef typename select_type<M == 1,
					typename select_type<N == 1,
						smallgemm_11K<T, K>,
						smallgemm_1NK<T, N, K>
					>::type,
					typename select_type<N == 1,
						smallgemm_M1K<T, M, K>,
						typename select_type<K == 1,
							smallgemm_MN1<T, M, N>,
							smallgemm_MNK<T, M, N, K>
						>::type
					>::type
				>::type impl_t;

		BCS_ENSURE_INLINE
		void eval_nn_b0(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			impl_t::eval_nn_b0(alpha, a, lda, b, ldb, c, ldc);
		}

		BCS_ENSURE_INLINE
		void eval_nn_b1(const T alpha,
				const T* __restrict__ a, const int lda,
				const T* __restrict__ b, const int ldb,
				T* __restrict__ c, const int ldc)
		{
			impl_t::eval_nn_b1(alpha, a, lda, b, ldb, c, ldc);
		}
	};

} }


#endif /* SMALL_BLASL3_H_ */
