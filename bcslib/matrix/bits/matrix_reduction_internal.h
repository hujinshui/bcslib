/**
 * @file matrix_reduction_internal.h
 *
 * Internal implementation of full reduction on matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef MATRIX_REDUCTION_INTERNAL_H_
#define MATRIX_REDUCTION_INTERNAL_H_

#include <bcslib/matrix/dense_matrix.h>
#include <bcslib/matrix/column_traverser.h>

namespace bcs { namespace detail {

	// generic reduction

	template<typename Reductor, class Mat>
	struct unary_reduction_eval_helper
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_unary_reduction_functor<Reductor>::value,
				"Reductor must be a unary reduction functor");
#endif

		static inline typename Reductor::result_type
		run(Reductor reduc,
				const IMatrixXpr<Mat, typename Reductor::argument_type>& A)
		{
			typedef typename Reductor::accum_type accum_t;

			if (!is_empty(A))
			{
				index_t m = A.nrows();
				index_t n = A.ncolumns();

				column_traverser<Mat> vec_a(A.derived());

				if (m > 1)
				{
					accum_t s = reduc(vec_a[0]);
					for (index_t i = 1; i < m; ++i) s = reduc(s, vec_a[i]);

					if (n > 1)
					{
						++ vec_a;
						for (index_t j = 1; j < n; ++j, ++vec_a)
						{
							for (index_t i = 0; i < m; ++i) s = reduc(s, vec_a[i]);
						}
					}

					return reduc.get(s, m * n);
				}
				else // m == 1
				{
					accum_t s = reduc(vec_a[0]);

					if (n > 1)
					{
						++ vec_a;
						for (index_t j = 1; j < n; ++j, ++vec_a) s = reduc(s, vec_a[0]);
					}

					return reduc.get(s, n);
				}
			}
			else
			{
				return bcs::empty_reduc_result_of<Reductor>::get(reduc);
			}
		}
	};


	template<typename Reductor, class LMat, class RMat>
	struct binary_reduction_eval_helper
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_binary_reduction_functor<Reductor>::value,
				"Reductor must be a binary reduction functor");
#endif

		static inline typename Reductor::result_type
		run(Reductor reduc,
				const IMatrixXpr<LMat, typename Reductor::argument_type>& A,
				const IMatrixXpr<RMat, typename Reductor::argument_type>& B)
		{
			typedef typename Reductor::accum_type accum_t;

			if (!is_empty(A))
			{
				index_t m = A.nrows();
				index_t n = A.ncolumns();

				column_traverser<LMat> vec_a(A.derived());
				column_traverser<RMat> vec_b(B.derived());

				if (m > 1)
				{
					accum_t s = reduc(vec_a[0], vec_b[0]);
					for (index_t i = 1; i < m; ++i) s = reduc(s, vec_a[i], vec_b[i]);

					if (n > 1)
					{
						++ vec_a;
						++ vec_b;
						for (index_t j = 1; j < n; ++j, ++vec_a, ++vec_b)
						{
							for (index_t i = 0; i < m; ++i) s = reduc(s, vec_a[i], vec_b[i]);
						}
					}

					return reduc.get(s, m * n);
				}
				else // m == 1
				{
					accum_t s = reduc(vec_a[0], vec_b[0]);

					if (n > 1)
					{
						++ vec_a;
						++ vec_b;
						for (index_t j = 1; j < n; ++j, ++vec_a, ++vec_b) s = reduc(s, vec_a[0], vec_b[0]);
					}

					return reduc.get(s, n);
				}
			}
			else
			{
				return bcs::empty_reduc_result_of<Reductor>::get(reduc);
			}
		}
	};



} }

#endif /* MATRIX_REDUCTION_INTERNAL_H_ */




