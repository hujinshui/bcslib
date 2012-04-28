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
	 *  Evaluation
	 *
	 ********************************************/

	template<typename Fun, class Arg>
	struct expr_evaluator<unary_ewise_expr<Fun, Arg> >
	{
		typedef unary_ewise_expr<Fun, Arg> expr_type;
		typedef typename matrix_traits<expr_type>::value_type value_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IRegularMatrix<DMat, value_type>& dst)
		{
			detail::ewise_evaluator<expr_type>::run(expr, dst.derived());
		}
	};


	template<typename Fun, class LArg, class RArg>
	struct expr_evaluator<binary_ewise_expr<Fun, LArg, RArg> >
	{
		typedef binary_ewise_expr<Fun, LArg, RArg> expr_type;
		typedef typename matrix_traits<expr_type>::value_type value_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IRegularMatrix<DMat, value_type>& dst)
		{
			detail::ewise_evaluator<expr_type>::run(expr, dst.derived());
		}
	};

}

#endif
