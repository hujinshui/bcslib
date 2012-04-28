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
	 *  Optimization
	 *
	 ********************************************/

	template<class Mat, bool IsEwiseAccessible> struct ewise_expr_optimizer;

	template<class Mat>
	struct ewise_expr_optimizer<Mat, false>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::has_matrix_interface<Mat, IMatrixXpr>::value, "Mat must be a model of IMatrixXpr");
		static_assert(!(bcs::is_column_traversable<Mat>::value), "Mat must NOT be column-traversable");
#endif

		typedef Mat input_type;

		typedef dense_matrix<
				typename matrix_traits<Mat>::value_type,
				ct_rows<Mat>::value,
				ct_cols<Mat>::value> output_type;

		typedef output_type return_type;

		BCS_ENSURE_INLINE
		static return_type optimize(const input_type& expr)
		{
			return expr;
		}
	};


	template<class Mat>
	struct ewise_expr_optimizer<Mat, true>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::has_matrix_interface<Mat, IMatrixXpr>::value, "Mat must be a model of IMatrixXpr");
		static_assert(bcs::is_column_traversable<Mat>::value, "Mat must be column-traversable");
#endif

		typedef Mat input_type;
		typedef Mat output_type;
		typedef const output_type& return_type;

		BCS_ENSURE_INLINE
		static return_type optimize(const input_type& expr)
		{
			return expr;
		}
	};


	template<typename Fun, class Arg>
	struct ewise_expr_optimizer<unary_ewise_expr<Fun, Arg>, false>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(!(bcs::is_column_traversable<Arg>::value), "Arg must be NOT column-traversable");
#endif

		typedef ewise_expr_optimizer<Arg, false> arg_op;
		typedef typename arg_op::output_type TArg;

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_column_traversable<TArg>::value, "TArg should be column-traversable");
#endif

		typedef unary_ewise_expr<Fun, Arg> input_type;
		typedef unary_ewise_expr<Fun, TArg> output_type;
		typedef output_type return_type;

		BCS_ENSURE_INLINE
		static return_type optimize(const input_type& expr)
		{
			return output_type(expr.fun, arg_op::optimize(expr.arg));
		}
	};


	template<typename Fun, class LArg, class RArg>
	struct ewise_expr_optimizer<binary_ewise_expr<Fun, LArg, RArg>, false>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(!(
				bcs::is_column_traversable<LArg>::value &&
				bcs::is_column_traversable<RArg>::value),
				"Either LArg or RArg is expected to be NOT column-traversable");
#endif

		typedef ewise_expr_optimizer<LArg, false> left_arg_op;
		typedef ewise_expr_optimizer<RArg, false> right_arg_op;

		typedef typename left_arg_op::output_type LTArg;
		typedef typename right_arg_op::output_type RTArg;

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_column_traversable<LTArg>::value, "LTArg should be column-traversable");
		static_assert(bcs::is_column_traversable<RTArg>::value, "RTArg should be column-traversable");
#endif

		typedef binary_ewise_expr<Fun, LArg, RArg> input_type;
		typedef binary_ewise_expr<Fun, LTArg, RTArg> output_type;
		typedef output_type return_type;

		BCS_ENSURE_INLINE
		static return_type optimize(const input_type& expr)
		{
			return output_type(expr.fun,
					left_arg_op::optimize(expr.left_arg),
					right_arg_op::optimize(expr.right_arg));
		}
	};


	/********************************************
	 *
	 *  Evaluation
	 *
	 ********************************************/

	template<class Expr>
	struct ewise_evaluator
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_column_traversable<Expr>::value, "Expr must be column-traversable");
#endif

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



