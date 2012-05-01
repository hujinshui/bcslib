/**
 * @file matrix_transpose.h
 *
 * Expression to represent matrix/vector transposition
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_TRANSPOSE_H_
#define BCSLIB_MATRIX_TRANSPOSE_H_

#include <bcslib/matrix/matrix_xpr.h>
#include <bcslib/matrix/bits/matrix_transpose_internal.h>

#include <bcslib/matrix/dense_matrix.h>
#include <bcslib/matrix/ref_matrix.h>

namespace bcs
{

	/********************************************
	 *
	 *  Generic Matrix transpose expression
	 *
	 ********************************************/

	template<class Mat> class matrix_transpose_expr;

	template<class Arg>
	struct matrix_traits<matrix_transpose_expr<Arg> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_cols<Arg>::value;
		static const int compile_time_num_cols = ct_rows<Arg>::value;

		static const bool is_linear_indexable = false;
		static const bool is_continuous = false;
		static const bool is_sparse = false;
		static const bool is_readonly = true;

		typedef typename matrix_traits<Arg>::value_type value_type;
		typedef index_t index_type;
	};


	template<class Arg>
	struct matrix_transpose_expr
	: public IMatrixXpr<matrix_transpose_expr<Arg>, typename matrix_traits<Arg>::value_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(has_matrix_interface<Arg, IMatrixXpr>::value, "Arg must be an matrix expression.");
#endif

		typedef Arg arg_type;
		BCS_MAT_TRAITS_CDEFS(typename matrix_traits<Arg>::value_type)

		const Arg& arg;

		matrix_transpose_expr(const Arg& a)
		: arg(a)
		{
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return arg.nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return arg.size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return arg.ncolumns();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return arg.nrows();
		}
	};


	template<class Arg>
	struct expr_evaluator<matrix_transpose_expr<Arg> >
	{
		typedef matrix_transpose_expr<Arg> SExpr;
		typedef typename matrix_traits<SExpr>::value_type value_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const SExpr& expr, IRegularMatrix<DMat, value_type>& dst)
		{
			detail::matrix_transposer<SExpr, DMat,
				bcs::has_matrix_interface<DMat, IDenseMatrix>::value>::run(expr, dst.derived());
		}
	};


	template<class Arg>
	struct matrix_transpose_helper
	{
		typedef matrix_transpose_expr<Arg> result_type;

		BCS_ENSURE_INLINE
		result_type transpose(const Arg& arg)
		{
			return matrix_transpose_expr<Arg>(arg);
		}
	};

	template<class Arg, typename T>
	BCS_ENSURE_INLINE
	typename matrix_transpose_helper<Arg>::result_type
	transpose(const IMatrixXpr<Arg, T>& arg)
	{
		return matrix_transpose_helper<Arg>::transpose(arg.derived());
	}


	/********************************************
	 *
	 *  Special cases
	 *
	 ********************************************/





}

#endif /* MATRIX_TRANSPOSE_H_ */
