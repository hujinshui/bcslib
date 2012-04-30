/**
 * @file ewise_matrix_eval.h
 *
 * Internal implementation for element-wise evaluation
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_EWISE_MATRIX_EVAL_INTERNAL_H_
#define BCSLIB_EWISE_MATRIX_EVAL_INTERNAL_H_

#include <bcslib/matrix/vector_proxy.h>

namespace bcs { namespace detail {


	/********************************************
	 *
	 *  Evaluation
	 *
	 ********************************************/

	template<class Expr, class DMat, bool DoLinear>
	struct ewise_evaluator;


	template<class Expr, class DMat>
	struct ewise_evaluator<Expr, DMat, false>
	{
		typedef typename matrix_traits<Expr>::value_type value_type;

		static void run(const Expr& expr, DMat& dst)
		{
			vecwise_reader<Expr> reader(expr);
			vecwise_writer<DMat> writer(dst);

			index_t m = expr.nrows();
			index_t n = expr.ncolumns();

			if (n == 1)
			{
				copy_vec(m, reader, writer);
			}
			else
			{
				for (index_t j = 0; j < n; ++j, ++reader, ++writer)
				{
					copy_vec(m, reader, writer);
				}
			}
		}

	};


	template<class Expr, class DMat>
	struct ewise_evaluator<Expr, DMat, true>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_accessible_as_vector<Expr>::value, "Expr must be accessible-as-vector");
		static_assert(matrix_traits<DMat>::is_linear_indexable, "DMat must be linearly indexable");
#endif

		typedef typename matrix_traits<Expr>::value_type value_type;

		static void run(const Expr& expr, DMat& dst)
		{
			vec_reader<Expr> reader(expr);
			vec_writer<DMat> writer(dst);

			copy_vec(expr.nelems(), reader, writer);
		}

	};


} }

#endif



