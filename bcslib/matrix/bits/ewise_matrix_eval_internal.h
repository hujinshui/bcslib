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

#include <bcslib/matrix/ewise_matrix_expr.h>
#include <bcslib/matrix/dense_matrix.h>
#include <bcslib/matrix/column_traverser.h>

namespace bcs { namespace detail {


	/********************************************
	 *
	 *  Evaluation
	 *
	 ********************************************/

	template<class Expr>
	struct ewise_evaluator
	{
		typedef column_traverser<Expr> traverser_type;
		typedef typename matrix_traits<Expr>::value_type value_type;

		template<class DMat>
		static void run(const Expr& expr, IRegularMatrix<DMat, value_type>& dst)
		{
			traverser_type evec(expr);

			index_t m = expr.nrows();
			index_t n = expr.ncolumns();

			if (n == 1)
			{
				for (index_t i = 0; i < m; ++i) dst.elem(i, 0) = evec[i];
			}
			else
			{
				for (index_t j = 0; j < n; ++j, ++evec)
				{
					for (index_t i = 0; i < m; ++i) dst.elem(i, j) = evec[i];
				}
			}
		}


		template<class DMat>
		static void run(const Expr& expr, IDenseMatrix<DMat, value_type>& dst)
		{
			traverser_type evec(expr);

			index_t m = expr.nrows();
			index_t n = expr.ncolumns();

			if (n == 1)
			{
				value_type *pd = dst.ptr_data();
				for (index_t i = 0; i < m; ++i) pd[i] = evec[i];
			}
			else
			{
				value_type *pd = dst.ptr_data();
				index_t ldim = dst.lead_dim();

				for (index_t j = 0; j < n; ++j, pd += ldim, ++evec)
				{
					for (index_t i = 0; i < m; ++i) pd[i] = evec[i];
				}
			}
		}

	};  // ewise_evaluator



} }

#endif



