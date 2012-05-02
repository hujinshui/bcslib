/**
 * @file matrix_xpr.h
 *
 * The basis for matrix expressions
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_XPR_H_
#define BCSLIB_MATRIX_XPR_H_

#include <bcslib/matrix/matrix_concepts.h>

namespace bcs
{


	/******************************************************
	 *
	 *  The concept of expression optimizer
	 *
	 *  - typedef result_expr_type;
	 *  - a static optimize function;
	 *
	 * The concept of expression evaluator
	 *
	 * - the static evaluate function:
	 *
	 * 		evaluate(optimized_expr, dst);
	 *
	 ******************************************************/

	template<class Expr>
	struct expr_optimizer
	{
		typedef Expr result_expr_type;

		BCS_ENSURE_INLINE
		static const result_expr_type& optimize(const Expr& expr)
		{
			return expr;  // by default, do nothing for optimization
		}
	};

	template<typename T, class Expr, class DMat>
	BCS_ENSURE_INLINE
	void evaluate_to(const IMatrixXpr<Expr, T>& expr, IRegularMatrix<DMat, T>& dst)
	{
		typedef typename expr_optimizer<Expr>::result_expr_type optim_expr_type;
		typedef expr_evaluator<optim_expr_type> evaluator_t;

		evaluator_t::evaluate(
				expr_optimizer<Expr>::optimize(expr.derived()),
				dst.derived());
	}

	// forward declaration of generic expression types

	template<typename Fun, class Arg> struct unary_ewise_expr;
	template<typename Fun, class LArg, class RArg> struct binary_ewise_expr;

	template<typename Fun, class Arg> struct unary_colwise_redux_expr;
	template<typename Fun, class LArg, class RArg> struct binary_colwise_redux_expr;

	template<typename Fun, class Arg> struct unary_rowwise_redux_expr;
	template<typename Fun, class LArg, class RArg> struct binary_rowwise_redux_expr;

}

#endif
