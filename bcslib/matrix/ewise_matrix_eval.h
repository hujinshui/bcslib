/**
 * @file ewise_matrix_eval.h
 *
 * Evaluation of element-wise expression
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_EWISE_MATRIX_EVAL_H_
#define BCSLIB_EWISE_MATRIX_EVAL_H_

#include <bcslib/matrix/ewise_matrix_expr.h>
#include <bcslib/matrix/bits/ewise_matrix_eval_internal.h>

namespace bcs
{

	/********************************************
	 *
	 *  Optimization
	 *
	 ********************************************/

	template<typename Fun, class Arg>
	struct expr_optimizer<unary_ewise_expr<Fun, Arg> >
	{
		typedef unary_ewise_expr<Fun, Arg> input_type;

		typedef detail::ewise_expr_optimizer<input_type,
				bcs::is_column_traversable<input_type>::value> optim_t;

		typedef typename optim_t::output_type result_expr_type;
		typedef typename optim_t::return_type return_type;

		BCS_ENSURE_INLINE
		return_type optimize(const input_type& expr)
		{
			return optim_t::optimize(expr);
		}
	};


	template<typename Fun, class LArg, class RArg>
	struct expr_optimizer<binary_ewise_expr<Fun, LArg, RArg> >
	{
		typedef binary_ewise_expr<Fun, LArg, RArg> input_type;

		typedef detail::ewise_expr_optimizer<input_type,
				bcs::is_column_traversable<input_type>::value> optim_t;

		typedef typename optim_t::output_type result_expr_type;
		typedef typename optim_t::return_type return_type;

		BCS_ENSURE_INLINE
		return_type optimize(const input_type& expr)
		{
			return optim_t::optimize(expr);
		}
	};



	/********************************************
	 *
	 *  Evaluation
	 *
	 ********************************************/

	template<typename Fun, class Arg>
	struct expr_evaluator<unary_ewise_expr<Fun, Arg> >
	{
		typedef unary_ewise_expr<Fun, Arg> expr_type;
		typedef typename matrix_traits<expr_type>::value_type value_type;

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_column_traversable<expr_type>::value,
				"The expression is properly optimized");
#endif

		template<class DMat>
		BCS_ENSURE_INLINE
		void evaluate(const expr_type& expr, IRegularMatrix<DMat, value_type>& dst)
		{
			detail::ewise_evaluator<expr_type>::run(expr, dst.derived());
		}
	};


}

#endif
